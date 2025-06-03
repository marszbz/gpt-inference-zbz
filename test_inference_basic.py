#!/usr/bin/env python3
"""
基础推理功能测试
验证推理引擎的基本功能，无需GPU环境
"""

import sys
import os
import logging
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

def test_data_loading():
    """测试数据加载功能"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试数据加载功能 ===")
    
    try:
        from src.utils import DataLoader
        
        # 检查数据集文件
        dataset_files = list(Path("data/datasets").glob("benchmark_dataset_config_*.jsonl"))
        if not dataset_files:
            logger.error("未找到数据集文件")
            return False
        
        # 测试加载第一个配置文件
        test_file = dataset_files[0]
        logger.info(f"测试加载数据集: {test_file}")
        
        data_loader = DataLoader(str(test_file))
        samples = data_loader.load_samples()
        
        logger.info(f"成功加载 {len(samples)} 个样本")
        
        # 验证样本格式
        if samples:
            sample = samples[0]
            logger.info(f"样本示例 - ID: {sample.id}, 配置: {sample.config_id}, "
                       f"提示长度: {sample.prompt_length}, 生成长度: {sample.generation_length}")
        
        return True
        
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}")
        return False

def test_model_manager_loading():
    """测试模型管理器加载"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试模型管理器加载 ===")
    
    try:
        from src.models import ModelManager
        
        # 测试配置文件加载
        model_manager = ModelManager("config/model_config.yaml")
        logger.info("模型管理器初始化成功")
        
        # 测试模型配置
        config = model_manager.config
        logger.info(f"模型名称: {config['model']['name']}")
        logger.info(f"模型路径: {config['model']['path']}")
        logger.info(f"缓存目录: {config['model']['cache_dir']}")
        
        return True
        
    except Exception as e:
        logger.error(f"模型管理器测试失败: {e}")
        return False

def test_inference_engine_init():
    """测试推理引擎初始化"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试推理引擎初始化 ===")
    
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
        
        logger.info("推理引擎初始化成功")
        
        # 测试性能监控器
        performance_monitor = inference_engine.performance_monitor
        system_info = performance_monitor.get_system_info()
        
        logger.info(f"系统信息获取成功: CPU核心数 = {system_info['cpu']['cores']}")
        
        return True
        
    except Exception as e:
        logger.error(f"推理引擎测试失败: {e}")
        return False

def test_performance_evaluator():
    """测试性能评估器"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试性能评估器 ===")
    
    try:
        from src.evaluation import PerformanceEvaluator
        
        # 初始化性能评估器
        evaluator = PerformanceEvaluator("config/inference_config.yaml")
        logger.info("性能评估器初始化成功")
        
        # 创建模拟结果数据
        mock_results = [
            {
                'sample_id': 'test_1',
                'total_time': 100.0,
                'first_token_time': 50.0,
                'throughput': 25.0,
                'prompt_tokens': 32,
                'generated_tokens': 32,
                'memory_usage': {'gpu_memory_allocated_mb': 1024.0},
                'gpu_utilization': 85.0
            },
            {
                'sample_id': 'test_2',
                'total_time': 120.0,
                'first_token_time': 60.0,
                'throughput': 22.0,
                'prompt_tokens': 128,
                'generated_tokens': 64,
                'memory_usage': {'gpu_memory_allocated_mb': 1200.0},
                'gpu_utilization': 90.0
            }
        ]
        
        # 测试统计计算
        stats = evaluator.calculate_statistics(mock_results)
        logger.info(f"统计计算成功: 平均延迟 = {stats['latency']['total_time']['mean']:.2f} ms")
        
        return True
        
    except Exception as e:
        logger.error(f"性能评估器测试失败: {e}")
        return False

def test_parallel_strategy_manager():
    """测试并行策略管理器"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试并行策略管理器 ===")
    
    try:
        from src.models import ParallelStrategyManager
        
        # 初始化策略管理器
        strategy_manager = ParallelStrategyManager()
        logger.info("并行策略管理器初始化成功")
        
        # 测试策略选择
        strategy = strategy_manager.select_strategy(
            model_size=1.5,
            num_gpus=1,
            memory_per_gpu=8
        )
        
        logger.info(f"策略选择成功: {strategy['name']}")
        logger.info(f"策略描述: {strategy['description']}")
        
        return True
        
    except Exception as e:
        logger.error(f"并行策略管理器测试失败: {e}")
        return False

def test_config_files():
    """测试配置文件"""
    logger = logging.getLogger(__name__)
    logger.info("=== 测试配置文件 ===")
    
    config_files = [
        "config/model_config.yaml",
        "config/data_config.yaml", 
        "config/inference_config.yaml",
        "config/deepspeed_config.json"
    ]
    
    all_valid = True
    
    for config_file in config_files:
        try:
            if config_file.endswith('.yaml'):
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            logger.info(f"✓ {config_file} 加载成功")
            
        except Exception as e:
            logger.error(f"✗ {config_file} 加载失败: {e}")
            all_valid = False
    
    return all_valid

def run_comprehensive_test():
    """运行综合测试"""
    logger = logging.getLogger(__name__)
    logger.info("开始基础推理功能测试")
    
    tests = [
        ("配置文件", test_config_files),
        ("数据加载", test_data_loading),
        ("模型管理器", test_model_manager_loading),
        ("并行策略管理器", test_parallel_strategy_manager),
        ("推理引擎初始化", test_inference_engine_init),
        ("性能评估器", test_performance_evaluator),
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
    logger.info("基础推理功能测试总结")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name:20s}: {status}")
    
    logger.info(f"\n测试结果: {passed}/{total} 通过")
    logger.info(f"总耗时: {end_time - start_time:.2f} 秒")
    
    if passed == total:
        logger.info("🎉 所有基础功能测试通过！")
        return True
    else:
        logger.warning(f"⚠️  有 {total - passed} 个测试失败")
        return False

if __name__ == "__main__":
    setup_logging()
    
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"测试发生严重错误: {e}")
        sys.exit(1)
