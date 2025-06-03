#!/usr/bin/env python3
"""
CPU兼容性推理测试脚本
用于验证系统在CPU环境下的基本功能
"""

import sys
import os
import logging
import torch
from pathlib import Path
import json
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

def test_cpu_compatibility():
    """测试CPU兼容性"""
    logger = logging.getLogger(__name__)
    
    logger.info("=== CPU兼容性测试开始 ===")
    
    # 1. 检查设备可用性
    logger.info("1. 检查计算设备...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f"✓ CUDA可用，检测到 {torch.cuda.device_count()} 个GPU")
        device = torch.device("cuda:0")
    else:
        logger.info("✓ CUDA不可用，使用CPU")
        device = torch.device("cpu")
    
    # 2. 测试基本模块导入
    logger.info("2. 测试模块导入...")
    try:
        from src.models import ModelManager
        from src.inference import DistributedInferenceEngine
        from src.utils import DataLoader, PerformanceMonitor
        logger.info("✓ 所有模块导入成功")
    except Exception as e:
        logger.error(f"✗ 模块导入失败: {e}")
        return False
    
    # 3. 测试数据加载
    logger.info("3. 测试数据加载...")
    try:
        data_loader = DataLoader("data/datasets")
        samples = data_loader.load_samples(config_ids=[0])  # 只加载配置0
        if samples:
            logger.info(f"✓ 数据加载成功，加载了 {len(samples)} 个样本")
            # 显示第一个样本信息
            sample = samples[0]
            logger.info(f"  样本示例: prompt_length={sample.prompt_length}, "
                       f"generation_length={sample.generation_length}, source={sample.source_type}")
        else:
            logger.warning("! 没有加载到数据样本")
    except Exception as e:
        logger.error(f"✗ 数据加载失败: {e}")
        return False
    
    # 4. 测试简单的Transformer模型加载（使用更小的模型）
    logger.info("4. 测试轻量级模型加载...")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # 使用GPT2-small进行测试（更适合CPU）
        model_name = "gpt2"  # 124M参数，适合CPU测试
        
        logger.info(f"加载模型: {model_name}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = model.to(device)
        model.eval()
        
        logger.info("✓ 模型加载成功")
        
        # 5. 测试简单推理
        logger.info("5. 测试简单推理...")
        test_prompt = "The future of artificial intelligence is"
        
        # 编码输入
        inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
        
        # 生成文本
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=10,  # 少量token以加快速度
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id
            )
        
        inference_time = time.time() - start_time
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"✓ 推理成功")
        logger.info(f"  输入: {test_prompt}")
        logger.info(f"  输出: {generated_text}")
        logger.info(f"  推理时间: {inference_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"✗ 模型测试失败: {e}")
        logger.warning("这可能是因为网络问题或模型下载失败")
        logger.info("提示: 可以手动下载模型到本地缓存目录")
        return False
    
    # 6. 测试性能监控
    logger.info("6. 测试性能监控...")
    try:
        monitor = PerformanceMonitor()
        system_info = monitor.get_system_info()
        logger.info("✓ 性能监控正常")
        logger.info(f"  CPU核心数: {system_info.get('cpu_count', 'N/A')}")
        logger.info(f"  内存: {system_info.get('memory_total_gb', 'N/A'):.1f}GB")
        if cuda_available:
            logger.info(f"  GPU: {system_info.get('gpu_info', {}).get('name', 'N/A')}")
    except Exception as e:
        logger.error(f"✗ 性能监控测试失败: {e}")
        return False
    
    logger.info("=== CPU兼容性测试完成 ===")
    logger.info("✓ 所有测试通过，系统可以在CPU环境下运行")
    
    return True

def main():
    """主函数"""
    setup_logging()
    
    success = test_cpu_compatibility()
    
    if success:
        print("\n🎉 CPU兼容性测试成功！")
        print("现在可以运行完整的推理测试脚本了")
        return 0
    else:
        print("\n❌ CPU兼容性测试失败")
        print("请检查错误信息并修复问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
