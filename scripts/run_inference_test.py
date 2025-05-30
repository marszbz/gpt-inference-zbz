"""
分布式推理测试脚本
支持单卡和多卡分布式推理性能测试
"""

import sys
import os
import argparse
import logging
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
import json
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import ModelManager
from src.inference import DistributedInferenceEngine
from src.utils import DataLoader
import jsonlines

def setup_logging(rank: int = 0, log_level: str = "INFO") -> None:
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/inference_test_rank_{rank}.log')
        ]
    )

def load_test_samples(dataset_path: str, config_ids: list = None, max_samples: int = None):
    """加载测试样本"""
    data_loader = DataLoader(dataset_path)
    samples = data_loader.load_samples(config_ids)
    
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    
    # 转换为字典格式
    sample_dicts = []
    for sample in samples:
        sample_dict = {
            'id': sample.id,
            'config_id': sample.config_id,
            'prompt': sample.prompt,
            'prompt_length': sample.prompt_length,
            'generation_length': sample.generation_length,
            'source_type': sample.source_type,
            'metadata': sample.metadata
        }
        sample_dicts.append(sample_dict)
    
    return sample_dicts

def run_single_gpu_inference(args):
    """运行单GPU推理测试"""
    logger = logging.getLogger(__name__)
    logger.info("开始单GPU推理测试")
    
    # 初始化模型管理器
    model_manager = ModelManager(args.model_config)
    model_manager.load_model()
    
    # 初始化推理引擎
    inference_engine = DistributedInferenceEngine(
        model_manager, 
        args.inference_config
    )
    
    # 加载测试数据
    logger.info(f"加载测试数据: {args.dataset}")
    samples = load_test_samples(
        args.dataset, 
        args.config_ids, 
        args.max_samples
    )
    
    # 运行推理测试
    logger.info(f"开始推理测试，共 {len(samples)} 个样本")
    results = inference_engine.run_performance_benchmark(samples)
    
    # 清理资源
    inference_engine.cleanup()
    
    logger.info("单GPU推理测试完成")
    return results

def run_distributed_worker(rank: int, world_size: int, args):
    """分布式推理工作进程"""
    # 设置日志
    setup_logging(rank, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"启动分布式工作进程: rank={rank}, world_size={world_size}")
    
    try:
        # 初始化模型管理器
        model_manager = ModelManager(args.model_config)
        model_manager.setup_distributed(rank, world_size)
        model_manager.load_model(local_rank=rank)
        
        # 如果启用DeepSpeed
        if hasattr(args, 'use_deepspeed') and args.use_deepspeed:
            model_manager.setup_deepspeed(args)
        
        # 初始化推理引擎
        inference_engine = DistributedInferenceEngine(
            model_manager, 
            args.inference_config
        )
        
        # 只在主进程加载数据
        if rank == 0:
            logger.info(f"加载测试数据: {args.dataset}")
            samples = load_test_samples(
                args.dataset, 
                args.config_ids, 
                args.max_samples
            )
            
            # 将样本数量广播给所有进程
            sample_count = torch.tensor(len(samples), device=f'cuda:{rank}')
        else:
            samples = []
            sample_count = torch.tensor(0, device=f'cuda:{rank}')
        
        # 广播样本数量
        dist.broadcast(sample_count, src=0)
        
        # 分发样本数据（简化实现，实际可能需要更复杂的数据分发）
        if rank != 0:
            # 非主进程创建空样本列表
            samples = [None] * sample_count.item()
        
        # 同步所有进程
        dist.barrier()
        
        # 运行推理测试
        logger.info(f"开始分布式推理测试，共 {len(samples)} 个样本")
        if rank == 0:
            results = inference_engine.run_performance_benchmark(samples)
        else:
            # 非主进程也需要参与推理，但不返回完整结果
            inference_engine.run_performance_benchmark(samples[:10])  # 运行少量样本
            results = {}
        
        # 清理资源
        inference_engine.cleanup()
        
        logger.info(f"Rank {rank} 推理测试完成")
        
        if rank == 0:
            return results
        
    except Exception as e:
        logger.error(f"Rank {rank} 出现错误: {e}")
        raise

def run_distributed_inference(args):
    """运行分布式推理测试"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始分布式推理测试，GPU数量: {args.world_size}")
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # 启动多进程
    mp.spawn(
        run_distributed_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )

def main():
    parser = argparse.ArgumentParser(description='GPT分布式推理性能测试')
    
    # 基本参数
    parser.add_argument('--dataset', type=str, 
                       default='data/datasets/benchmark_dataset.jsonl',
                       help='测试数据集路径')
    parser.add_argument('--model-config', type=str, 
                       default='config/model_config.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--inference-config', type=str, 
                       default='config/inference_config.yaml',
                       help='推理配置文件路径')
    
    # 分布式参数
    parser.add_argument('--distributed', action='store_true',
                       help='启用分布式推理')
    parser.add_argument('--world-size', type=int, default=1,
                       help='分布式进程数量')
    parser.add_argument('--master-addr', type=str, default='localhost',
                       help='主节点地址')
    parser.add_argument('--master-port', type=str, default='12355',
                       help='主节点端口')
    
    # DeepSpeed参数
    parser.add_argument('--use-deepspeed', action='store_true',
                       help='启用DeepSpeed')
    parser.add_argument('--deepspeed-config', type=str,
                       default='config/deepspeed_config.json',
                       help='DeepSpeed配置文件')
    
    # 测试参数
    parser.add_argument('--config-ids', type=int, nargs='+',
                       help='指定测试的配置ID列表')
    parser.add_argument('--max-samples', type=int,
                       help='最大测试样本数量')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(0, args.log_level)
    logger = logging.getLogger(__name__)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，无法运行GPU推理测试")
        sys.exit(1)
    
    # 检查数据集文件
    if not Path(args.dataset).exists():
        logger.error(f"数据集文件不存在: {args.dataset}")
        logger.info("请先运行 'python scripts/generate_dataset.py' 生成数据集")
        sys.exit(1)
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        if args.distributed and args.world_size > 1:
            # 分布式推理
            results = run_distributed_inference(args)
        else:
            # 单GPU推理
            results = run_single_gpu_inference(args)
        
        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"推理测试完成，总耗时: {total_time:.2f} 秒")
        
        # 输出结果摘要
        if results:
            logger.info("=== 测试结果摘要 ===")
            for config_id, data in results.items():
                config = data['config']
                stats = data['statistics']
                logger.info(f"配置 {config_id} (P{config['prompt_length']}_G{config['generation_length']}): "
                          f"吞吐量={stats['throughput']['mean']:.2f} tokens/s, "
                          f"延迟={stats['latency']['total_time']['mean']:.2f} ms")
        
    except Exception as e:
        logger.error(f"推理测试失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
