#!/usr/bin/env python3
"""
GPU推理功能测试
验证GPU环境下的推理功能，包括多GPU分布式推理
"""

import sys
import os
import logging
import torch
import time
from pathlib import Path

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

def check_gpu_environment():
    """检查GPU环境"""
    logger = logging.getLogger(__name__)
    logger.info("=== 检查GPU环境 ===")
    
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，无法进行GPU推理测试")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"检测到 {gpu_count} 个GPU设备")
    
    for i in range(gpu_count):
        try:
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} - {memory_total:.1f} GB")
            
            # 测试GPU内存分配
            torch.cuda.set_device(i)
            test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)
            logger.info(f"  内存测试通过，已分配: {memory_allocated:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"GPU {i} 测试失败: {e}")
            return False
    
    return True

def test_model_loading_gpu():
    """测试GPU上的模型加载"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试GPU模型加载 ===")
    
    try:
        from src.models import ModelManager
        
        # 检查配置文件
        config_path = "config/model_config.yaml"
        if not Path(config_path).exists():
            logger.error(f"配置文件不存在: {config_path}")
            return False
        
        # 初始化模型管理器
        model_manager = ModelManager(config_path)
        logger.info("模型管理器初始化成功")
        
        # 设置GPU设备
        device = torch.device("cuda:0")
        model_manager.device = device
        
        logger.info("准备加载模型到GPU...")
        logger.info("注意：这可能需要几分钟时间下载模型文件")
        
        # 记录开始时间
        start_time = time.time()
        
        # 加载模型（这里可能需要下载模型文件）
        try:
            model_manager.load_model(local_rank=0)
            load_time = time.time() - start_time
            logger.info(f"模型加载成功，耗时: {load_time:.2f} 秒")
            
            # 验证模型在GPU上
            if hasattr(model_manager.model, 'device'):
                model_device = next(model_manager.model.parameters()).device
                logger.info(f"模型设备: {model_device}")
            
            # 测试简单推理
            logger.info("测试GPU推理...")
            test_prompt = "Hello, this is a test prompt for"
            inputs = model_manager.prepare_inputs([test_prompt])
            
            with torch.no_grad():
                outputs = model_manager.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.8
                )
            
            generated_text = model_manager.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            logger.info(f"生成文本: {generated_text}")
            logger.info("GPU推理测试成功")
            
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.info("可能的原因：")
            logger.info("1. 网络连接问题，无法下载模型")
            logger.info("2. GPU内存不足")
            logger.info("3. 模型配置文件路径错误")
            return False
            
    except Exception as e:
        logger.error(f"GPU模型测试失败: {e}")
        return False

def test_gpu_memory_management():
    """测试GPU内存管理"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试GPU内存管理 ===")
    
    try:
        device = torch.device("cuda:0")
        
        # 获取初始内存状态
        initial_memory = torch.cuda.memory_allocated(0) / (1024**2)
        logger.info(f"初始GPU内存使用: {initial_memory:.1f} MB")
        
        # 分配一些内存
        test_tensors = []
        for i in range(5):
            tensor = torch.randn(1024, 1024, device=device)
            test_tensors.append(tensor)
            current_memory = torch.cuda.memory_allocated(0) / (1024**2)
            logger.info(f"分配张量 {i+1}，当前内存: {current_memory:.1f} MB")
        
        # 清理内存
        del test_tensors
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(0) / (1024**2)
        logger.info(f"清理后GPU内存使用: {final_memory:.1f} MB")
        
        logger.info("GPU内存管理测试通过")
        return True
        
    except Exception as e:
        logger.error(f"GPU内存管理测试失败: {e}")
        return False

def test_multi_gpu_detection():
    """测试多GPU检测和配置"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试多GPU检测 ===")
    
    try:
        gpu_count = torch.cuda.device_count()
        
        if gpu_count == 1:
            logger.info("检测到单GPU环境")
            logger.info("建议配置：使用数据并行或推理优化")
        elif gpu_count >= 2:
            logger.info(f"检测到多GPU环境：{gpu_count} 个GPU")
            logger.info("可用配置选项：")
            logger.info("1. 数据并行 (DataParallel)")
            logger.info("2. 分布式数据并行 (DistributedDataParallel)")
            logger.info("3. 张量并行 (Tensor Parallelism)")
            logger.info("4. 流水线并行 (Pipeline Parallelism)")
            
            # 测试多GPU通信
            if gpu_count >= 2:
                logger.info("测试多GPU通信...")
                try:
                    tensor1 = torch.randn(100, 100, device='cuda:0')
                    tensor2 = tensor1.to('cuda:1')
                    result = tensor2.sum()
                    logger.info("多GPU通信测试通过")
                except Exception as e:
                    logger.warning(f"多GPU通信测试失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"多GPU检测测试失败: {e}")
        return False

def test_performance_monitoring():
    """测试GPU性能监控"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试GPU性能监控 ===")
    
    try:
        from src.utils import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # 获取系统信息
        system_info = monitor.get_system_info()
        logger.info("系统信息获取成功：")
        logger.info(f"  CPU: {system_info['cpu']['model']}")
        logger.info(f"  CPU核心数: {system_info['cpu']['cores']}")
        logger.info(f"  内存: {system_info['memory']['total_gb']:.1f} GB")
        
        if 'gpu' in system_info:
            for i, gpu_info in enumerate(system_info['gpu']):
                logger.info(f"  GPU {i}: {gpu_info['name']} - {gpu_info['memory_total_gb']:.1f} GB")
        
        logger.info("GPU性能监控测试通过")
        return True
        
    except Exception as e:
        logger.error(f"GPU性能监控测试失败: {e}")
        return False

def test_inference_engine_gpu():
    """测试GPU推理引擎"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试GPU推理引擎 ===")
    
    try:
        from src.models import ModelManager
        from src.inference import DistributedInferenceEngine
        
        # 初始化模型管理器
        model_manager = ModelManager("config/model_config.yaml")
        
        # 初始化推理引擎
        inference_engine = DistributedInferenceEngine(
            model_manager, 
            "config/inference_config.yaml"
        )
        
        logger.info("GPU推理引擎初始化成功")
        
        # 测试GPU监控功能
        if hasattr(inference_engine, 'measure_gpu_utilization'):
            gpu_util = inference_engine.measure_gpu_utilization()
            logger.info(f"GPU利用率监控: {gpu_util}%")
        
        # 测试内存监控
        memory_info = inference_engine.measure_memory_usage()
        if 'gpu_memory_allocated_mb' in memory_info:
            logger.info(f"GPU内存监控: {memory_info['gpu_memory_allocated_mb']:.1f} MB")
        
        logger.info("GPU推理引擎测试通过")
        return True
        
    except Exception as e:
        logger.error(f"GPU推理引擎测试失败: {e}")
        return False

def run_gpu_comprehensive_test():
    """运行GPU综合测试"""
    logger = logging.getLogger(__name__)
    logger.info("开始GPU推理功能测试")
    
    # 首先检查GPU环境
    if not check_gpu_environment():
        logger.error("GPU环境检查失败，无法继续测试")
        return False
    
    tests = [
        ("GPU内存管理", test_gpu_memory_management),
        ("多GPU检测", test_multi_gpu_detection),
        ("GPU性能监控", test_performance_monitoring),
        ("GPU推理引擎", test_inference_engine_gpu),
        ("GPU模型加载", test_model_loading_gpu),  # 放在最后，因为可能需要下载模型
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        logger.info(f"\n--- 开始测试: {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"测试结果: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"测试异常: {e}")
    
    end_time = time.time()
    
    # 输出总结
    logger.info("\n" + "="*50)
    logger.info("GPU推理功能测试总结")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name:20s}: {status}")
    
    logger.info(f"\n测试结果: {passed}/{total} 通过")
    logger.info(f"总耗时: {end_time - start_time:.2f} 秒")
    
    if passed == total:
        logger.info("🎉 所有GPU功能测试通过！")
        logger.info("现在可以运行完整的GPU推理性能测试")
        return True
    else:
        logger.warning(f"⚠️  有 {total - passed} 个测试失败")
        return False

if __name__ == "__main__":
    setup_logging()
    
    try:
        success = run_gpu_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"测试发生严重错误: {e}")
        sys.exit(1)
