#!/usr/bin/env python3
"""
系统集成测试脚本
验证各个模块的导入和基本功能
"""

import sys
import os
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试数据生成模块
        from src.data_generation import DatasetGenerator
        print("✅ 数据生成模块导入成功")
        
        # 测试模型管理模块
        from src.models import ModelManager, ParallelStrategyManager
        print("✅ 模型管理模块导入成功")
        
        # 测试推理引擎
        from src.inference import DistributedInferenceEngine
        print("✅ 推理引擎模块导入成功")
        
        # 测试评估模块
        from src.evaluation import PerformanceEvaluator
        print("✅ 性能评估模块导入成功")
        
        # 测试工具模块
        from src.utils import PerformanceMonitor, DataLoader, CommunicationProfiler
        print("✅ 工具模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n🔍 测试配置文件加载...")
    
    try:
        import yaml
        import json
        
        # 测试YAML配置文件
        config_files = [
            "config/model_config.yaml",
            "config/data_config.yaml", 
            "config/inference_config.yaml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"✅ {config_file} 加载成功")
            else:
                print(f"⚠️ {config_file} 文件不存在")
        
        # 测试JSON配置文件
        json_config = "config/deepspeed_config.json"
        if os.path.exists(json_config):
            with open(json_config, 'r') as f:
                config = json.load(f)
            print(f"✅ {json_config} 加载成功")
        else:
            print(f"⚠️ {json_config} 文件不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n🔍 测试基本功能...")
    
    try:
        # 测试并行策略管理器
        from src.models.parallel_strategy import ParallelStrategyManager
        
        strategy_manager = ParallelStrategyManager()
        available_strategies = strategy_manager.get_available_strategies()
        print(f"✅ 可用并行策略: {available_strategies}")
        
        # 测试策略配置
        for strategy in available_strategies:
            config = strategy_manager.get_strategy_config(strategy)
            print(f"   - {strategy}: {config}")
        
        # 测试数据生成器初始化
        from src.data_generation.dataset_generator import DatasetGenerator
        
        generator = DatasetGenerator()
        print("✅ 数据生成器初始化成功")
        
        # 测试性能监控器
        from src.utils.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        memory_info = monitor.get_system_info()
        print(f"✅ 系统信息获取成功: CPU核心数={memory_info.get('cpu_count', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        traceback.print_exc()
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n🔍 测试GPU可用性...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ 检测到 {gpu_count} 个GPU")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
            return True
        else:
            print("⚠️ 未检测到可用的GPU")
            return False
            
    except Exception as e:
        print(f"❌ GPU检测失败: {e}")
        return False

def create_test_directories():
    """创建测试所需的目录"""
    print("\n🔍 创建测试目录...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/datasets",
        "logs",
        "results",
        "cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 目录 {directory} 已准备")

def main():
    """主测试函数"""
    print("🚀 GPT-1.5B 分布式推理性能测试系统 - 集成测试")
    print("=" * 60)
    
    # 创建必要的目录
    create_test_directories()
    
    # 运行各项测试
    tests = [
        ("模块导入测试", test_imports),
        ("配置文件加载测试", test_config_loading), 
        ("基本功能测试", test_basic_functionality),
        ("GPU可用性测试", test_gpu_availability)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*50}")
        print(f"执行: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # 输出测试结果总结
    print(f"\n{'='*60}")
    print("📊 测试结果总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(tests)} 项测试通过")
    
    if passed == len(tests):
        print("🎉 所有测试通过！系统准备就绪。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查相关配置。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
