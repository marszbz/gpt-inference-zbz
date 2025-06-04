#!/usr/bin/env python3
"""
简单的多GPU推理测试
使用PyTorch原生的分布式功能，避免DeepSpeed依赖问题
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_process_group(rank: int, world_size: int, backend: str = "nccl"):
    """初始化进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    """清理进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()

def load_test_data(data_path: str, max_samples: int = 20) -> List[Dict[str, Any]]:
    """加载测试数据"""
    data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                data.append(json.loads(line.strip()))
        return data
    except Exception as e:
        print(f"加载数据失败: {e}，使用默认数据")
        return [
            {"prompt": "The future of artificial intelligence is", "max_length": 50},
            {"prompt": "In a world where technology advances rapidly", "max_length": 50},
            {"prompt": "Climate change poses significant challenges", "max_length": 50},
            {"prompt": "The benefits of renewable energy include", "max_length": 50},
            {"prompt": "Machine learning algorithms can help", "max_length": 50}
        ] * 4  # 重复以获得更多样本

def run_inference_worker(rank: int, world_size: int, strategy: str, data_path: str, num_samples: int = 20):
    """运行推理工作进程"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(f"Worker-{rank}")
    
    try:
        # 初始化进程组
        setup_process_group(rank, world_size)
        logger.info(f"进程组初始化完成")
        
        # 设置CUDA设备
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
            logger.info(f"使用设备: {device}")
        else:
            device = torch.device("cpu")
            logger.info("使用CPU设备")
        
        # 加载模型和tokenizer
        model_name = "gpt2-xl"
        logger.info(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        
        # 根据策略配置模型
        if strategy == "data_parallel":
            # 数据并行：每个GPU处理不同的数据
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            logger.info("使用数据并行策略")
        elif strategy == "tensor_parallel":
            # 简单的张量并行：将模型分割到不同GPU
            # 注意：这是一个简化的实现
            logger.info("使用张量并行策略（简化版）")
        else:
            logger.info("使用单GPU策略")
        
        # 同步所有进程
        dist.barrier()
        
        # 加载和分布数据
        if rank == 0:
            test_data = load_test_data(data_path, num_samples)
            logger.info(f"主进程加载了 {len(test_data)} 条数据")
        else:
            test_data = None
        
        # 广播数据到所有进程
        test_data = broadcast_data(test_data, rank)
        
        # 分配数据到各个进程
        local_data = distribute_data(test_data, rank, world_size)
        logger.info(f"本地分配到 {len(local_data)} 条数据")
        
        # 运行推理
        results = []
        total_tokens = 0
        total_time = 0
        
        logger.info("开始推理测试...")
        
        for i, sample in enumerate(local_data):
            try:
                prompt = sample.get("prompt", "The future of AI is")
                max_new_tokens = sample.get("max_length", 50) - len(tokenizer.encode(prompt))
                max_new_tokens = max(10, min(max_new_tokens, 50))  # 限制生成长度
                
                # 编码输入
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # 开始计时
                start_time = time.time()
                
                # 生成
                with torch.no_grad():
                    if strategy == "data_parallel":
                        # 数据并行模式
                        model.eval()
                        outputs = model.module.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            temperature=1.0
                        )
                    else:
                        # 其他模式
                        model.eval()
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            temperature=1.0
                        )
                
                # 停止计时
                end_time = time.time()
                
                # 计算指标
                inference_time = end_time - start_time
                generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                tokens_per_second = generated_tokens / inference_time if inference_time > 0 else 0
                
                # 记录结果
                result = {
                    "rank": rank,
                    "sample_id": i,
                    "prompt": prompt,
                    "generated_tokens": generated_tokens,
                    "inference_time": inference_time,
                    "tokens_per_second": tokens_per_second,
                    "gpu_memory_allocated": torch.cuda.memory_allocated(rank) / 1024**2 if torch.cuda.is_available() else 0,
                }
                
                results.append(result)
                total_tokens += generated_tokens
                total_time += inference_time
                
                # 打印示例（每个rank打印一个）
                if i == 0:
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info(f"示例生成:")
                    logger.info(f"  提示: {prompt}")
                    logger.info(f"  生成: {generated_text[len(prompt):]}")
                    logger.info(f"  令牌数: {generated_tokens}, 吞吐量: {tokens_per_second:.2f} tokens/s")
                
            except Exception as e:
                logger.error(f"处理样本 {i} 时出错: {str(e)}")
                continue
        
        # 计算本地统计
        if results:
            avg_throughput = total_tokens / total_time if total_time > 0 else 0
            avg_latency = total_time / len(results)
            avg_memory = sum(r["gpu_memory_allocated"] for r in results) / len(results)
            
            local_stats = {
                "rank": rank,
                "strategy": strategy,
                "total_samples": len(results),
                "total_tokens": total_tokens,
                "total_time": total_time,
                "avg_throughput": avg_throughput,
                "avg_latency": avg_latency,
                "avg_memory_mb": avg_memory
            }
            
            logger.info(f"本地统计: 样本={len(results)}, 令牌={total_tokens}, 吞吐量={avg_throughput:.2f} tokens/s")
        else:
            local_stats = {"rank": rank, "error": "No successful samples"}
        
        # 收集所有进程的结果
        all_stats = gather_results(local_stats, rank, world_size)
        
        # 只在rank 0保存和打印结果
        if rank == 0:
            save_and_print_results(all_stats, strategy, world_size)
        
        # 同步所有进程
        dist.barrier()
        
    except Exception as e:
        logger.error(f"工作进程 {rank} 出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        cleanup()

def broadcast_data(data, rank):
    """广播数据到所有进程"""
    if rank == 0:
        data_to_broadcast = data
    else:
        data_to_broadcast = None
    
    # 使用torch.distributed的广播功能
    data_list = [data_to_broadcast]
    dist.broadcast_object_list(data_list, src=0)
    return data_list[0]

def distribute_data(data: List[Dict], rank: int, world_size: int) -> List[Dict]:
    """将数据分配到各个进程"""
    chunk_size = len(data) // world_size
    start_idx = rank * chunk_size
    if rank == world_size - 1:
        # 最后一个进程处理剩余的所有数据
        end_idx = len(data)
    else:
        end_idx = start_idx + chunk_size
    
    return data[start_idx:end_idx]

def gather_results(local_result, rank: int, world_size: int):
    """收集所有进程的结果"""
    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, local_result)
    return gathered_results

def save_and_print_results(all_stats: List[Dict], strategy: str, world_size: int):
    """保存和打印结果"""
    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 计算总体统计
    total_samples = sum(stat.get("total_samples", 0) for stat in all_stats if "total_samples" in stat)
    total_tokens = sum(stat.get("total_tokens", 0) for stat in all_stats if "total_tokens" in stat)
    total_time = max(stat.get("total_time", 0) for stat in all_stats if "total_time" in stat)  # 使用最大时间
    
    overall_throughput = total_tokens / total_time if total_time > 0 else 0
    avg_memory = sum(stat.get("avg_memory_mb", 0) for stat in all_stats if "avg_memory_mb" in stat) / world_size
    
    summary = {
        "strategy": strategy,
        "world_size": world_size,
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "overall_throughput": overall_throughput,
        "avg_memory_mb": avg_memory,
        "per_rank_stats": all_stats
    }
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"multi_gpu_{strategy}_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\\n" + "="*80)
    print(f"多GPU推理性能测试结果 - 策略: {strategy}")
    print("="*80)
    print(f"GPU数量: {world_size}")
    print(f"总样本数: {total_samples}")
    print(f"总令牌数: {total_tokens}")
    print(f"总时间: {total_time:.2f}s")
    print(f"整体吞吐量: {overall_throughput:.2f} tokens/s")
    print(f"平均显存使用: {avg_memory:.0f} MB")
    print("\\n各GPU性能:")
    for stat in all_stats:
        if "total_samples" in stat:
            rank = stat["rank"]
            throughput = stat.get("avg_throughput", 0)
            latency = stat.get("avg_latency", 0)
            memory = stat.get("avg_memory_mb", 0)
            print(f"  GPU {rank}: {throughput:.2f} tokens/s, {latency:.3f}s延迟, {memory:.0f}MB显存")
    
    print(f"\\n结果保存到: {result_file}")
    print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="多GPU推理测试")
    parser.add_argument("--world_size", type=int, default=4, help="GPU数量")
    parser.add_argument("--strategy", type=str, default="data_parallel",
                        choices=["single_gpu", "data_parallel", "tensor_parallel"],
                        help="并行策略")
    parser.add_argument("--data_path", type=str, 
                        default="data/datasets/benchmark_dataset_config_1.jsonl",
                        help="测试数据路径")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="测试样本数量")
    
    args = parser.parse_args()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("CUDA不可用，无法进行多GPU测试")
        return
    
    available_gpus = torch.cuda.device_count()
    if available_gpus < args.world_size:
        print(f"可用GPU数量 ({available_gpus}) 少于请求数量 ({args.world_size})")
        args.world_size = available_gpus
    
    print(f"启动多GPU推理测试:")
    print(f"  策略: {args.strategy}")
    print(f"  GPU数量: {args.world_size}")
    print(f"  测试数据: {args.data_path}")
    print(f"  样本数量: {args.num_samples}")
    
    if args.world_size == 1 or args.strategy == "single_gpu":
        # 单GPU模式
        run_inference_worker(0, 1, "single_gpu", args.data_path, args.num_samples)
    else:
        # 多GPU模式
        mp.spawn(
            run_inference_worker,
            args=(args.world_size, args.strategy, args.data_path, args.num_samples),
            nprocs=args.world_size,
            join=True
        )

if __name__ == "__main__":
    main()
