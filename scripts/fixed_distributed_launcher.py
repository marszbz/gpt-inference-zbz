#!/usr/bin/env python3
"""
修复版分布式推理启动器
解决DeepSpeed LOCAL_RANK环境变量问题
"""

import os
import sys
import argparse
import subprocess
import time
import json
import multiprocessing as mp
from pathlib import Path
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_environment_for_worker(rank: int, world_size: int, master_port: int = 29500):
    """为工作进程设置环境变量"""
    env = os.environ.copy()
    env['RANK'] = str(rank)
    env['LOCAL_RANK'] = str(rank)  # 关键: 设置LOCAL_RANK
    env['WORLD_SIZE'] = str(world_size)
    env['MASTER_ADDR'] = 'localhost'
    env['MASTER_PORT'] = str(master_port)
    
    # DeepSpeed环境变量
    env['CUDA_VISIBLE_DEVICES'] = str(rank)
    
    return env

def run_worker_subprocess(rank: int, world_size: int, args):
    """在子进程中运行工作进程"""
    env = setup_environment_for_worker(rank, world_size)
    
    # 构建命令
    cmd = [
        sys.executable,
        "scripts/fixed_distributed_worker.py",
        "--strategy", args.strategy,
        "--model_config", args.model_config,
        "--batch_size", str(args.batch_size),
        "--data_path", args.data_path,
        "--num_iterations", str(args.num_iterations)
    ]
    
    if args.no_cuda:
        cmd.append("--no_cuda")
    
    print(f"启动Rank {rank} 进程: {' '.join(cmd)}")
    
    # 启动子进程
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    
    return process

def launch_distributed_inference(args):
    """启动分布式推理"""
    print("启动修复版分布式推理系统...")
    print(f"策略: {args.strategy}")
    print(f"GPU数量: {args.world_size}")
    print(f"批次大小: {args.batch_size}")
    
    if args.world_size == 1:
        # 单GPU模式，直接运行
        env = setup_environment_for_worker(0, 1)
        os.environ.update(env)
        
        # 导入并运行工作函数
        from scripts.fixed_distributed_worker import run_distributed_inference_worker
        
        worker_args = argparse.Namespace(**vars(args))
        run_distributed_inference_worker(worker_args)
        return
    
    # 多GPU模式，启动多个子进程
    processes = []
    
    try:
        # 启动所有工作进程
        for rank in range(args.world_size):
            process = run_worker_subprocess(rank, args.world_size, args)
            processes.append(process)
            time.sleep(1)  # 避免竞争条件
        
        print(f"已启动 {len(processes)} 个工作进程")
        
        # 等待所有进程完成
        results = []
        for i, process in enumerate(processes):
            print(f"等待Rank {i} 进程完成...")
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"Rank {i} 完成成功")
                if stdout:
                    print(f"Rank {i} 输出:\n{stdout}")
            else:
                print(f"Rank {i} 出错 (返回码: {process.returncode})")
                if stderr:
                    print(f"Rank {i} 错误:\n{stderr}")
            
            results.append((process.returncode, stdout, stderr))
        
        # 检查结果
        success_count = sum(1 for code, _, _ in results if code == 0)
        print(f"\n分布式推理完成: {success_count}/{len(processes)} 进程成功")
        
        if success_count == len(processes):
            print("所有进程执行成功!")
            # 合并结果
            merge_distributed_results(args)
        else:
            print("部分进程执行失败，请检查日志")
    
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在终止所有进程...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    
    except Exception as e:
        print(f"启动分布式推理时出错: {e}")
        # 清理进程
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        raise

def merge_distributed_results(args):
    """合并分布式推理结果"""
    results_dir = Path("results")
    strategy_files = list(results_dir.glob(f"distributed_{args.strategy}_rank_*.json"))
    
    if not strategy_files:
        print("未找到结果文件")
        return
    
    # 合并所有rank的结果
    merged_results = {
        'strategy': args.strategy,
        'world_size': args.world_size,
        'batch_size': args.batch_size,
        'rank_results': {},
        'overall_metrics': {}
    }
    
    total_throughput = 0
    total_samples = 0
    total_time = 0
    
    for file_path in strategy_files:
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
    
    # 计算整体指标
    merged_results['overall_metrics'] = {
        'total_throughput_tokens_per_sec': total_throughput,
        'total_samples': total_samples,
        'total_time': total_time,
        'average_latency': total_time / max(total_samples, 1),
        'parallel_efficiency': total_throughput / (args.world_size * (total_throughput / args.world_size)) if total_throughput > 0 else 0
    }
    
    # 保存合并结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    merged_file = results_dir / f"merged_{args.strategy}_{timestamp}.json"
    
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)
    
    print(f"合并结果已保存到: {merged_file}")
    
    # 打印摘要
    print(f"\n=== {args.strategy} 策略性能摘要 ===")
    print(f"总吞吐量: {total_throughput:.2f} tokens/sec")
    print(f"总样本数: {total_samples}")
    print(f"总时间: {total_time:.2f}s")
    print(f"平均延迟: {merged_results['overall_metrics']['average_latency']:.2f}s")
    print(f"并行效率: {merged_results['overall_metrics']['parallel_efficiency']:.2%}")

def main():
    parser = argparse.ArgumentParser(description="修复版分布式推理启动器")
    
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
    parser.add_argument("--num_iterations", type=int, default=3,
                        help="测试迭代次数")
    
    # 分布式参数
    parser.add_argument("--world_size", type=int, default=4,
                        help="进程数量 (GPU数量)")
    
    # 其他参数
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not args.no_cuda and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU模式")
        args.no_cuda = True
    
    if not args.no_cuda:
        available_gpus = torch.cuda.device_count()
        if available_gpus < args.world_size:
            print(f"警告: 可用GPU数量 ({available_gpus}) 少于请求的world_size ({args.world_size})")
            args.world_size = available_gpus
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 启动分布式推理
    launch_distributed_inference(args)

if __name__ == "__main__":
    main()
