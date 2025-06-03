#!/usr/bin/env python3
"""
快速GPU测试启动脚本
自动检测GPU环境并运行适合的测试
"""

import sys
import os
import logging
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_requirements():
    """检查基本要求"""
    logger = logging.getLogger(__name__)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用，无法运行GPU测试")
        logger.error("请安装支持CUDA的PyTorch版本")
        return False
    
    # 检查数据集
    dataset_dir = Path("data/datasets")
    if not dataset_dir.exists() or not list(dataset_dir.glob("benchmark_dataset_config_*.jsonl")):
        logger.error("❌ 未找到测试数据集")
        logger.info("正在生成数据集...")
        
        try:
            # 运行数据集生成
            os.system("python scripts/generate_dataset.py")
            
            # 再次检查
            if not list(dataset_dir.glob("benchmark_dataset_config_*.jsonl")):
                logger.error("数据集生成失败")
                return False
            
            logger.info("✅ 数据集生成成功")
            
        except Exception as e:
            logger.error(f"数据集生成失败: {e}")
            return False
    
    # 检查配置文件
    config_files = [
        "config/model_config.yaml",
        "config/inference_config.yaml",
        "config/data_config.yaml"
    ]
    
    for config_file in config_files:
        if not Path(config_file).exists():
            logger.error(f"❌ 配置文件不存在: {config_file}")
            return False
    
    logger.info("✅ 基本要求检查通过")
    return True

def detect_gpu_setup():
    """检测GPU配置"""
    logger = logging.getLogger(__name__)
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"检测到 {gpu_count} 个GPU:")
    
    gpu_info = []
    total_memory = 0
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        total_memory += gpu_memory
        
        gpu_info.append({
            'id': i,
            'name': gpu_name,
            'memory_gb': gpu_memory
        })
        
        logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    logger.info(f"总GPU内存: {total_memory:.1f} GB")
    
    return gpu_info

def recommend_test_strategy(gpu_info):
    """推荐测试策略"""
    logger = logging.getLogger(__name__)
    
    gpu_count = len(gpu_info)
    total_memory = sum(gpu['memory_gb'] for gpu in gpu_info)
    
    logger.info(f"\n🎯 推荐的测试策略:")
    
    if gpu_count == 1:
        if gpu_info[0]['memory_gb'] >= 8:
            logger.info("✅ 单GPU配置，内存充足")
            logger.info("推荐：运行完整的单GPU性能基准测试")
            return "single_gpu_full"
        else:
            logger.info("⚠️  单GPU配置，内存有限")
            logger.info("推荐：运行轻量级的单GPU测试")
            return "single_gpu_light"
    
    elif gpu_count >= 2:
        logger.info(f"✅ 多GPU配置 ({gpu_count} 个GPU)")
        logger.info("推荐：运行分布式推理性能测试")
        if total_memory >= 32:
            logger.info("内存充足，可运行完整测试")
            return "multi_gpu_full"
        else:
            logger.info("内存有限，建议轻量级测试")
            return "multi_gpu_light"
    
    return "single_gpu_light"

def run_test_strategy(strategy):
    """执行测试策略"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n🚀 执行测试策略: {strategy}")
    
    if strategy == "single_gpu_full":
        logger.info("运行完整单GPU基准测试...")
        cmd = "python gpu_benchmark.py --max-samples-per-config 100"
        
    elif strategy == "single_gpu_light":
        logger.info("运行轻量级单GPU测试...")
        cmd = "python gpu_benchmark.py --max-samples-per-config 20"
        
    elif strategy == "multi_gpu_full":
        gpu_count = torch.cuda.device_count()
        logger.info(f"运行完整多GPU基准测试 ({gpu_count} GPUs)...")
        cmd = f"python gpu_benchmark.py --num-gpus {gpu_count} --max-samples-per-config 50"
        
    elif strategy == "multi_gpu_light":
        gpu_count = torch.cuda.device_count()
        logger.info(f"运行轻量级多GPU测试 ({gpu_count} GPUs)...")
        cmd = f"python gpu_benchmark.py --num-gpus {gpu_count} --max-samples-per-config 10"
        
    else:
        logger.error(f"未知的测试策略: {strategy}")
        return False
    
    logger.info(f"执行命令: {cmd}")
    
    try:
        result = os.system(cmd)
        return result == 0
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return False

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("🎯 GPU推理测试快速启动")
    logger.info("="*60)
    
    # 检查基本要求
    if not check_requirements():
        logger.error("基本要求检查失败，无法继续")
        sys.exit(1)
    
    # 检测GPU配置
    gpu_info = detect_gpu_setup()
    
    # 推荐测试策略
    strategy = recommend_test_strategy(gpu_info)
    
    # 询问用户确认
    print(f"\n是否执行推荐的测试策略？ (y/n): ", end="")
    try:
        user_input = input().strip().lower()
        if user_input not in ['y', 'yes', '']:
            logger.info("用户取消测试")
            sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\n用户取消测试")
        sys.exit(0)
    
    # 执行测试
    success = run_test_strategy(strategy)
    
    if success:
        logger.info("\n🎉 GPU推理测试完成！")
        logger.info("查看 results/ 目录获取详细结果")
        logger.info("查看 logs/ 目录获取详细日志")
    else:
        logger.error("\n❌ GPU推理测试失败")
        logger.info("请检查日志文件了解详细错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
