"""
分布式推理启动脚本
支持多种并行策略和4张RTX 3080 GPU
"""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import yaml
import time
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_manager import ModelManager
from src.models.parallel_strategy import ParallelStrategyManager
from src.inference.inference_engine import DistributedInferenceEngine
from src.utils.data_loader import DataLoader
from src.evaluation.performance_evaluator import PerformanceEvaluator

def setup_logging(rank: int) -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger(f"DistributedInference-Rank{rank}")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 只有rank 0输出到控制台
        if rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 每个rank都写入自己的日志文件
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"rank_{rank}.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def run_inference_worker(rank: int, world_size: int, args):
    """运行推理工作进程"""
    logger = setup_logging(rank)
    logger.info(f"启动推理工作进程 rank={rank}, world_size={world_size}")
    
    try:
        # 1. 初始化模型管理器
        model_manager = ModelManager(args.model_config)
        
        # 2. 设置分布式环境和并行策略
        model_manager.setup_distributed(rank, world_size, args.strategy)
        
        # 3. 加载模型
        model_manager.load_model(rank)
        
        # 4. 初始化推理引擎
        inference_engine = DistributedInferenceEngine(
            model_manager=model_manager,
            config_path=args.inference_config
        )
        
        # 5. 预热模型
        if rank == 0:
            logger.info("开始模型预热...")
        inference_engine.warmup()
        
        # 6. 加载测试数据
        data_loader = DataLoader(args.data_config)
        test_data = data_loader.load_test_data(args.data_path)
        
        # 7. 分配数据到各个进程
        local_data = distribute_data(test_data, rank, world_size)
        logger.info(f"Rank {rank} 分配到 {len(local_data)} 条测试数据")
        
        # 8. 运行推理测试
        results = inference_engine.run_inference_benchmark(
            test_data=local_data,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations
        )
        
        # 9. 收集结果
        all_results = gather_results(results, rank, world_size)
        
        # 10. 评估性能（只在rank 0执行）
        if rank == 0:
            evaluator = PerformanceEvaluator()
            metrics = evaluator.evaluate_performance(all_results)
            
            # 保存结果
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"inference_results_{args.strategy}_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'strategy': args.strategy,
                    'world_size': world_size,
                    'batch_size': args.batch_size,
                    'metrics': metrics,
                    'detailed_results': [r.__dict__ for r in all_results]
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"推理测试完成，结果保存到: {results_file}")
            
            # 打印关键指标
            print_performance_summary(metrics, args.strategy)
    
    except Exception as e:
        logger.error(f"Rank {rank} 推理过程出错: {str(e)}", exc_info=True)
        raise
    finally:
        # 清理资源
        if 'model_manager' in locals():
            model_manager.cleanup()

def distribute_data(data, rank: int, world_size: int):
    """将数据分配到各个进程"""
    chunk_size = len(data) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(data)
    return data[start_idx:end_idx]

def gather_results(local_results, rank: int, world_size: int):
    """收集所有进程的结果"""
    # 使用all_gather收集结果
    if dist.is_initialized():
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
        
        # 展平结果列表
        all_results = []
        for results in gathered_results:
            if results:
                all_results.extend(results)
        
        return all_results
    else:
        return local_results

def print_performance_summary(metrics: dict, strategy: str):
    """打印性能摘要"""
    print("\\n" + "="*80)
    print(f"分布式推理性能测试结果 - 策略: {strategy}")
    print("="*80)
    
    print(f"平均吞吐量: {metrics.get('avg_throughput', 0):.2f} tokens/s")
    print(f"平均延迟: {metrics.get('avg_latency', 0):.3f} s")
    print(f"首令牌延迟: {metrics.get('avg_first_token_latency', 0):.3f} s")
    print(f"GPU利用率: {metrics.get('avg_gpu_utilization', 0):.1f}%")
    print(f"显存使用: {metrics.get('avg_memory_usage', {}).get('gpu_memory_allocated_mb', 0):.0f} MB")
    
    communication_times = metrics.get('avg_communication_times', {})
    if communication_times:
        print("\\n通信开销:")
        for comm_type, time_ms in communication_times.items():
            print(f"  {comm_type}: {time_ms:.2f} ms")
    
    print("="*80 + "\\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分布式GPT推理测试")
    
    # 基本配置
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml",
                        help="模型配置文件路径")
    parser.add_argument("--inference_config", type=str, default="config/inference_config.yaml", 
                        help="推理配置文件路径")
    parser.add_argument("--data_config", type=str, default="config/data_config.yaml",
                        help="数据配置文件路径")
    
    # 推理参数
    parser.add_argument("--data_path", type=str, default="data/processed/test_dataset.jsonl",
                        help="测试数据路径")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="测试迭代次数")
    
    # 分布式参数
    parser.add_argument("--world_size", type=int, default=4,
                        help="GPU数量")
    parser.add_argument("--strategy", type=str, default="tensor_data_hybrid",
                        choices=["pure_data_parallel", "tensor_data_hybrid", 
                                "pipeline_data_hybrid", "full_model_parallel"],
                        help="并行策略")
    
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
    
    # 显示配置信息
    print("分布式推理配置:")
    print(f"  - 并行策略: {args.strategy}")
    print(f"  - GPU数量: {args.world_size}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 测试数据: {args.data_path}")
    print(f"  - 迭代次数: {args.num_iterations}")
    
    # 检查可用策略
    strategy_manager = ParallelStrategyManager(args.model_config)
    available_strategies = strategy_manager.get_available_strategies()
    print(f"\\n可用的并行策略:")
    for name, desc in available_strategies.items():
        print(f"  - {name}: {desc}")
    
    if args.strategy not in available_strategies:
        print(f"错误: 策略 '{args.strategy}' 不可用")
        return
    
    # 启动分布式推理
    print(f"\\n启动分布式推理进程...")
    
    if args.world_size == 1:
        # 单GPU模式
        run_inference_worker(0, 1, args)
    else:
        # 多GPU模式
        mp.spawn(
            run_inference_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )

if __name__ == "__main__":
    main()
