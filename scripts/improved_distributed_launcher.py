#!/usr/bin/env python3
"""
改进的分布式推理启动器
使用torch.multiprocessing解决进程管理问题
"""

import os
import sys
import argparse
import subprocess
import time
import json
import signal
from pathlib import Path
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_environment_for_rank(rank: int, world_size: int, master_port: int = 29500):
    """为指定rank设置环境变量"""
    env_vars = {
        'RANK': str(rank),
        'LOCAL_RANK': str(rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': str(master_port),
        'CUDA_VISIBLE_DEVICES': str(rank),
        'TOKENIZERS_PARALLELISM': 'false',
        'NCCL_DEBUG': 'WARN',  # 减少NCCL日志输出
        'NCCL_TIMEOUT': '1800',  # 30分钟超时
    }
    
    # 设置环境变量
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars

def run_single_gpu_worker(rank: int, args: argparse.Namespace):
    """运行单个GPU工作进程"""
    try:
        # 设置环境变量
        env_vars = setup_environment_for_rank(rank, args.world_size)
        
        print(f"[Rank {rank}] 启动工作进程...")
        print(f"[Rank {rank}] 环境变量: {env_vars}")
          # 构建工作进程命令
        python_exe = sys.executable
        worker_script = str(Path(__file__).parent / "fixed_distributed_worker.py")
        cmd = [
            python_exe,
            worker_script,
            "--strategy", args.strategy,
            "--model_config", args.model_config,
            "--batch_size", str(args.batch_size),
            "--data_path", args.data_path,
            "--num_iterations", str(args.num_iterations),
            "--num_samples", str(args.num_samples)
        ]
        
        if args.no_cuda:
            cmd.append("--no_cuda")
        
        print(f"[Rank {rank}] 执行命令: {' '.join(cmd)}")
        
        # 启动子进程
        process = subprocess.Popen(
            cmd,
            env={**os.environ, **env_vars},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        # 等待进程完成
        stdout, stderr = process.communicate()
        
        print(f"[Rank {rank}] 进程完成，返回码: {process.returncode}")
        
        if process.returncode == 0:
            print(f"[Rank {rank}] 执行成功")
            if stdout:
                print(f"[Rank {rank}] 输出:\n{stdout}")
        else:
            print(f"[Rank {rank}] 执行失败")
            if stderr:
                print(f"[Rank {rank}] 错误输出:\n{stderr}")
        
        return {
            'rank': rank,
            'returncode': process.returncode,
            'stdout': stdout,
            'stderr': stderr
        }
        
    except Exception as e:
        print(f"[Rank {rank}] 异常: {e}")
        import traceback
        traceback.print_exc()
        return {
            'rank': rank,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }

def launch_distributed_inference(args):
    """启动分布式推理"""
    print("=== 改进的分布式推理启动器 ===")
    print(f"策略: {args.strategy}")
    print(f"GPU数量: {args.world_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"数据路径: {args.data_path}")
    
    # 检查GPU可用性
    if not args.no_cuda:
        if not torch.cuda.is_available():
            print("CUDA不可用，切换到CPU模式")
            args.no_cuda = True
        else:
            available_gpus = torch.cuda.device_count()
            print(f"检测到 {available_gpus} 个GPU")
            
            if available_gpus < args.world_size:
                print(f"警告: 可用GPU数量 ({available_gpus}) 少于请求的world_size ({args.world_size})")
                args.world_size = min(args.world_size, available_gpus)
                print(f"调整world_size为: {args.world_size}")
    
    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 创建日志目录
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    if args.world_size == 1:
        print("单GPU模式")
        result = run_single_gpu_worker(0, args)
        results = [result]
    else:
        print(f"多GPU模式，启动 {args.world_size} 个进程")
        
        # 使用子进程池
        with mp.Pool(processes=args.world_size) as pool:
            # 为每个rank启动工作进程
            async_results = []
            for rank in range(args.world_size):
                async_result = pool.apply_async(run_single_gpu_worker, (rank, args))
                async_results.append(async_result)
                time.sleep(0.5)  # 避免启动竞争
            
            print(f"已提交 {len(async_results)} 个任务到进程池")
            
            # 等待所有任务完成
            results = []
            for i, async_result in enumerate(async_results):
                try:
                    result = async_result.get(timeout=3600)  # 1小时超时
                    results.append(result)
                    print(f"Rank {i} 任务完成")
                except mp.TimeoutError:
                    print(f"Rank {i} 任务超时")
                    results.append({
                        'rank': i,
                        'returncode': -2,
                        'stdout': '',
                        'stderr': 'Timeout'
                    })
                except Exception as e:
                    print(f"Rank {i} 任务异常: {e}")
                    results.append({
                        'rank': i,
                        'returncode': -3,
                        'stdout': '',
                        'stderr': str(e)
                    })
    
    # 分析结果
    success_count = sum(1 for r in results if r['returncode'] == 0)
    print(f"\n=== 执行结果摘要 ===")
    print(f"成功进程: {success_count}/{len(results)}")
    
    for result in results:
        rank = result['rank']
        returncode = result['returncode']
        if returncode == 0:
            print(f"Rank {rank}: 成功")
        else:
            print(f"Rank {rank}: 失败 (返回码: {returncode})")
            if result['stderr']:
                print(f"  错误: {result['stderr'][:200]}...")
    
    if success_count == len(results):
        print("\n所有进程执行成功!")
        merge_distributed_results(args)
    else:
        print(f"\n{len(results) - success_count} 个进程执行失败")
        
        # 保存失败日志
        failure_log = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': args.strategy,
            'world_size': args.world_size,
            'results': results
        }
        
        with open(logs_dir / f"failure_log_{args.strategy}_{int(time.time())}.json", 'w') as f:
            json.dump(failure_log, f, indent=2, ensure_ascii=False)

def merge_distributed_results(args):
    """合并分布式推理结果"""
    print("\n=== 合并分布式结果 ===")
    
    results_dir = Path("results")
    strategy_files = list(results_dir.glob(f"distributed_{args.strategy}_rank_*.json"))
    
    if not strategy_files:
        print("未找到结果文件")
        return
    
    print(f"找到 {len(strategy_files)} 个结果文件")
    
    # 合并所有rank的结果
    merged_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': args.strategy,
        'world_size': args.world_size,
        'batch_size': args.batch_size,
        'rank_results': {},
        'overall_metrics': {}
    }
    
    total_throughput = 0
    total_samples = 0
    total_time = 0
    memory_usage_list = []
    gpu_utilization_list = []
    
    for file_path in sorted(strategy_files):
        try:
            rank = int(file_path.stem.split('_')[-1])
            
            with open(file_path, 'r', encoding='utf-8') as f:
                rank_data = json.load(f)
            
            merged_results['rank_results'][rank] = rank_data
            
            # 累计指标
            if 'metrics' in rank_data:
                metrics = rank_data['metrics']
                total_throughput += metrics.get('throughput_tokens_per_sec', 0)
                total_samples += metrics.get('total_samples', 0)
                total_time = max(total_time, metrics.get('total_time', 0))
                
                # GPU指标
                if 'gpu_memory_used_mb' in metrics:
                    memory_usage_list.append(metrics['gpu_memory_used_mb'])
                if 'gpu_utilization_avg' in metrics:
                    gpu_utilization_list.append(metrics['gpu_utilization_avg'])
            
            print(f"已处理Rank {rank}结果")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    # 计算整体指标
    merged_results['overall_metrics'] = {
        'total_throughput_tokens_per_sec': total_throughput,
        'average_throughput_per_gpu': total_throughput / args.world_size if args.world_size > 0 else 0,
        'total_samples': total_samples,
        'total_time': total_time,
        'average_latency': total_time / max(total_samples, 1),
        'speedup_ratio': total_throughput / (total_throughput / args.world_size) if total_throughput > 0 else 0,
        'parallel_efficiency': (total_throughput / args.world_size) / (total_throughput / args.world_size) if args.world_size > 1 else 1.0
    }
    
    # GPU资源利用率
    if memory_usage_list:
        merged_results['overall_metrics']['total_memory_used_mb'] = sum(memory_usage_list)
        merged_results['overall_metrics']['average_memory_per_gpu_mb'] = sum(memory_usage_list) / len(memory_usage_list)
    
    if gpu_utilization_list:
        merged_results['overall_metrics']['average_gpu_utilization'] = sum(gpu_utilization_list) / len(gpu_utilization_list)
    
    # 保存合并结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    merged_file = results_dir / f"merged_{args.strategy}_gpu{args.world_size}_{timestamp}.json"
    
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)
    
    print(f"合并结果已保存到: {merged_file}")
    
    # 打印性能摘要
    metrics = merged_results['overall_metrics']
    print(f"\n=== {args.strategy} ({args.world_size}GPU) 性能摘要 ===")
    print(f"总吞吐量: {metrics['total_throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"单GPU平均吞吐量: {metrics['average_throughput_per_gpu']:.2f} tokens/sec")
    print(f"加速比: {metrics['speedup_ratio']:.2f}x")
    print(f"并行效率: {metrics['parallel_efficiency']:.2%}")
    print(f"总样本数: {metrics['total_samples']}")
    print(f"总时间: {metrics['total_time']:.2f}s")
    print(f"平均延迟: {metrics['average_latency']:.3f}s")
    
    if 'total_memory_used_mb' in metrics:
        print(f"总内存使用: {metrics['total_memory_used_mb']:.1f} MB")
        print(f"单GPU平均内存: {metrics['average_memory_per_gpu_mb']:.1f} MB")
    
    if 'average_gpu_utilization' in metrics:
        print(f"平均GPU利用率: {metrics['average_gpu_utilization']:.1f}%")

def main():
    # 设置multiprocessing启动方法
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="改进的分布式推理启动器")
    
    # 模型参数
    parser.add_argument("--model_config", type=str, 
                        default="config/model_config.yaml",
                        help="模型配置文件路径")
    parser.add_argument("--strategy", type=str,
                        choices=['pure_data_parallel', 'tensor_data_hybrid', 
                               'pipeline_data_hybrid', 'full_model_parallel'],
                        default='pure_data_parallel',
                        help="并行策略")
    
    # 推理参数
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--data_path", type=str,
                        default="data/test_prompts.jsonl",
                        help="测试数据路径")
    parser.add_argument("--num_iterations", type=int, default=3,                        help="测试迭代次数")
    
    # 分布式参数
    parser.add_argument("--world_size", type=int, default=2,
                        help="进程数量 (GPU数量)")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="GPU数量 (与world_size相同)")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="测试样本数量")
    
    # 其他参数
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 确保num_gpus与world_size一致
    if args.num_gpus != args.world_size:
        print(f"警告: num_gpus ({args.num_gpus}) 与 world_size ({args.world_size}) 不一致，使用 num_gpus")
        args.world_size = args.num_gpus
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 启动分布式推理
    try:
        launch_distributed_inference(args)
    except KeyboardInterrupt:
        print("\n接收到中断信号，退出...")
    except Exception as e:
        print(f"启动分布式推理时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
