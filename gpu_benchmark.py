#!/usr/bin/env python3
"""
GPU推理性能基准测试
专门用于GPU环境下的分布式推理性能测试
支持单GPU和多GPU并行策略
"""

import sys
import os
import argparse
import logging
import torch
import time
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging(rank: int = 0, log_level: str = "INFO") -> None:
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'[GPU-{rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/gpu_inference_test_rank_{rank}.log')
        ]
    )

def validate_gpu_environment(min_gpus: int = 1):
    """验证GPU环境"""
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用！")
        logger.error("请确保：")
        logger.error("1. 安装了支持CUDA的PyTorch版本")
        logger.error("2. 系统中有可用的NVIDIA GPU")
        logger.error("3. 正确安装了CUDA驱动和工具包")
        return False
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < min_gpus:
        logger.error(f"❌ 需要至少 {min_gpus} 个GPU，但只检测到 {gpu_count} 个")
        return False
    
    logger.info(f"✅ GPU环境验证通过，检测到 {gpu_count} 个GPU：")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return True

def load_test_dataset(dataset_dir: str, max_samples_per_config: int = None):
    """加载测试数据集"""
    logger = logging.getLogger(__name__)
    
    from src.utils import DataLoader
    
    dataset_files = list(Path(dataset_dir).glob("benchmark_dataset_config_*.jsonl"))
    if not dataset_files:
        logger.error(f"未在 {dataset_dir} 中找到数据集文件")
        return None
    
    logger.info(f"找到 {len(dataset_files)} 个配置文件")
    
    all_samples = []
    config_stats = {}
    
    for config_file in dataset_files:
        try:
            data_loader = DataLoader(str(config_file))
            samples = data_loader.load_samples()
            
            if max_samples_per_config and len(samples) > max_samples_per_config:
                samples = samples[:max_samples_per_config]
                logger.info(f"配置文件 {config_file.name}：限制为 {max_samples_per_config} 个样本")
            
            config_id = samples[0].config_id if samples else "unknown"
            config_stats[config_id] = len(samples)
            all_samples.extend(samples)
            
        except Exception as e:
            logger.warning(f"加载配置文件 {config_file} 失败: {e}")
    
    logger.info(f"总共加载 {len(all_samples)} 个测试样本")
    for config_id, count in config_stats.items():
        logger.info(f"  配置 {config_id}: {count} 个样本")
    
    return all_samples

def run_single_gpu_benchmark(args):
    """运行单GPU性能基准测试"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 开始单GPU推理性能测试")
    
    try:
        from src.models import ModelManager
        from src.inference import DistributedInferenceEngine
        
        # 初始化模型管理器
        logger.info("初始化模型管理器...")
        model_manager = ModelManager(args.model_config)
        
        # 设置GPU设备
        torch.cuda.set_device(0)
        model_manager.device = torch.device("cuda:0")
        
        # 加载模型
        logger.info("加载模型到GPU...")
        start_time = time.time()
        model_manager.load_model(local_rank=0)
        load_time = time.time() - start_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f} 秒")
        
        # 初始化推理引擎
        logger.info("初始化推理引擎...")
        inference_engine = DistributedInferenceEngine(
            model_manager, 
            args.inference_config
        )
        
        # 加载测试数据
        logger.info("加载测试数据...")
        samples = load_test_dataset(args.dataset, args.max_samples_per_config)
        if not samples:
            logger.error("测试数据加载失败")
            return None
        
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
        
        # 运行性能基准测试
        logger.info(f"开始推理基准测试，共 {len(sample_dicts)} 个样本")
        logger.info("这可能需要几分钟时间...")
        
        benchmark_start = time.time()
        results = inference_engine.run_performance_benchmark(sample_dicts)
        benchmark_time = time.time() - benchmark_start
        
        logger.info(f"推理基准测试完成，总耗时: {benchmark_time:.2f} 秒")
        
        # 清理资源
        inference_engine.cleanup()
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        logger.error(f"单GPU基准测试失败: {e}")
        return None

def run_multi_gpu_benchmark(args):
    """运行多GPU性能基准测试"""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 开始多GPU推理性能测试 (使用 {args.num_gpus} 个GPU)")
    
    try:
        # TODO: 实现多GPU分布式推理测试
        # 这里需要使用torch.multiprocessing或torch.distributed
        logger.info("多GPU基准测试功能开发中...")
        logger.info("当前版本支持单GPU测试，多GPU版本将在后续更新中提供")
        
        return None
        
    except Exception as e:
        logger.error(f"多GPU基准测试失败: {e}")
        return None

def analyze_benchmark_results(results):
    """分析基准测试结果"""
    logger = logging.getLogger(__name__)
    
    if not results:
        logger.warning("没有结果数据可分析")
        return
    
    logger.info("\n" + "="*60)
    logger.info("📊 GPU推理性能基准测试结果分析")
    logger.info("="*60)
    
    try:
        from src.evaluation import PerformanceEvaluator
        
        evaluator = PerformanceEvaluator("config/inference_config.yaml")
        
        # 按配置分组分析
        for config_id, config_data in results.items():
            logger.info(f"\n📋 配置 {config_id} 性能统计:")
            
            config_info = config_data['config']
            stats = config_data['statistics']
            
            logger.info(f"   提示长度: {config_info['prompt_length']} tokens")
            logger.info(f"   生成长度: {config_info['generation_length']} tokens")
            logger.info(f"   样本数量: {len(config_data['samples'])}")
            
            # 延迟统计
            logger.info(f"   总延迟: {stats['latency']['total_time']['mean']:.2f} ± {stats['latency']['total_time']['std']:.2f} ms")
            logger.info(f"   首Token: {stats['latency']['first_token_time']['mean']:.2f} ± {stats['latency']['first_token_time']['std']:.2f} ms")
            
            # 吞吐量统计
            logger.info(f"   吞吐量: {stats['throughput']['mean']:.2f} ± {stats['throughput']['std']:.2f} tokens/s")
            
            # 资源使用统计
            if 'resource_utilization' in stats:
                resource_stats = stats['resource_utilization']
                if 'gpu_memory_allocated_mb' in resource_stats:
                    logger.info(f"   GPU内存: {resource_stats['gpu_memory_allocated_mb']['mean']:.1f} ± {resource_stats['gpu_memory_allocated_mb']['std']:.1f} MB")
                if 'gpu_utilization' in resource_stats:
                    logger.info(f"   GPU利用率: {resource_stats['gpu_utilization']['mean']:.1f} ± {resource_stats['gpu_utilization']['std']:.1f} %")
        
        # 总体性能总结
        logger.info(f"\n🎯 总体性能总结:")
        
        all_throughputs = []
        all_latencies = []
        
        for config_data in results.values():
            stats = config_data['statistics']
            all_throughputs.append(stats['throughput']['mean'])
            all_latencies.append(stats['latency']['total_time']['mean'])
        
        if all_throughputs:
            avg_throughput = sum(all_throughputs) / len(all_throughputs)
            max_throughput = max(all_throughputs)
            avg_latency = sum(all_latencies) / len(all_latencies)
            min_latency = min(all_latencies)
            
            logger.info(f"   平均吞吐量: {avg_throughput:.2f} tokens/s")
            logger.info(f"   最大吞吐量: {max_throughput:.2f} tokens/s")
            logger.info(f"   平均延迟: {avg_latency:.2f} ms")
            logger.info(f"   最低延迟: {min_latency:.2f} ms")
        
        # 保存详细结果
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"gpu_benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 详细结果已保存到: {results_file}")
        
    except Exception as e:
        logger.error(f"结果分析失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='GPU推理性能基准测试')
    
    # 基本参数
    parser.add_argument('--dataset', type=str, 
                       default='data/datasets',
                       help='测试数据集目录路径')
    parser.add_argument('--model-config', type=str, 
                       default='config/model_config.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--inference-config', type=str, 
                       default='config/inference_config.yaml',
                       help='推理配置文件路径')
    
    # GPU参数
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='使用的GPU数量')
    parser.add_argument('--gpu-ids', type=int, nargs='+',
                       help='指定使用的GPU ID列表')
    
    # 测试参数
    parser.add_argument('--max-samples-per-config', type=int, default=50,
                       help='每个配置的最大样本数量')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='预热步数')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    # 性能优化参数
    parser.add_argument('--use-fp16', action='store_true',
                       help='使用FP16推理')
    parser.add_argument('--use-torch-compile', action='store_true',
                       help='使用torch.compile优化')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(0, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("🎯 GPU推理性能基准测试启动")
    logger.info("="*60)
    
    # 验证GPU环境
    if not validate_gpu_environment(args.num_gpus):
        sys.exit(1)
    
    # 检查数据集
    if not Path(args.dataset).exists():
        logger.error(f"数据集目录不存在: {args.dataset}")
        logger.info("请先运行 'python scripts/generate_dataset.py' 生成数据集")
        sys.exit(1)
    
    # 检查配置文件
    config_files = [args.model_config, args.inference_config]
    for config_file in config_files:
        if not Path(config_file).exists():
            logger.error(f"配置文件不存在: {config_file}")
            sys.exit(1)
    
    try:
        # 记录开始时间
        total_start_time = time.time()
        
        # 运行基准测试
        if args.num_gpus == 1:
            results = run_single_gpu_benchmark(args)
        else:
            results = run_multi_gpu_benchmark(args)
        
        # 记录结束时间
        total_time = time.time() - total_start_time
        
        if results:
            # 分析结果
            analyze_benchmark_results(results)
            
            logger.info(f"\n🎉 GPU推理性能基准测试完成！")
            logger.info(f"总耗时: {total_time:.2f} 秒")
        else:
            logger.error("基准测试失败，未获得有效结果")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"基准测试发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
