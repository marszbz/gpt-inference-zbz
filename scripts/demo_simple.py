"""
简单的分布式推理演示脚本
用于快速测试系统功能
"""

import os
import sys
import time
import torch
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_manager import ModelManager
from src.models.parallel_strategy import ParallelStrategyManager
from src.data_generation.dataset_generator import DatasetGenerator

def test_model_loading():
    """测试模型加载"""
    print("="*60)
    print("测试1: 模型加载")
    print("="*60)
    
    try:
        # 初始化模型管理器
        model_manager = ModelManager()
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            print(f"发现 {torch.cuda.device_count()} 张GPU:")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("没有可用的GPU，将使用CPU")
        
        # 加载模型
        print("\\n加载模型...")
        model_manager.load_model()
        
        # 获取模型信息
        model_info = model_manager.get_model_info()
        print(f"模型信息:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        print("✓ 模型加载测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型加载测试失败: {str(e)}")
        return False

def test_parallel_strategies():
    """测试并行策略"""
    print("\\n" + "="*60)
    print("测试2: 并行策略")
    print("="*60)
    
    try:
        # 初始化策略管理器
        strategy_manager = ParallelStrategyManager()
        
        # 获取可用策略
        strategies = strategy_manager.get_available_strategies()
        print(f"可用的并行策略:")
        for name, desc in strategies.items():
            print(f"  {name}: {desc}")
        
        # 测试策略验证
        print("\\n测试策略验证:")
        for strategy_name in strategies.keys():
            config = strategy_manager.get_strategy_config(strategy_name)
            is_valid = strategy_manager.validate_strategy(config)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {strategy_name}: {'有效' if is_valid else '无效'}")
        
        # 推荐策略
        recommended = strategy_manager.recommend_strategy(model_size_gb=6.0, batch_size=1)
        print(f"\\n推荐策略: {recommended}")
        
        print("✓ 并行策略测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 并行策略测试失败: {str(e)}")
        return False

def test_data_generation():
    """测试数据生成"""
    print("\\n" + "="*60)
    print("测试3: 数据生成")
    print("="*60)
    
    try:
        # 初始化数据生成器
        generator = DatasetGenerator()
        
        # 生成小量测试数据
        print("生成测试数据...")
        test_config = {
            'prompt_lengths': [32, 128],
            'generation_lengths': [32],
            'samples_per_config': 5,
            'data_sources': ['synthetic']  # 只使用合成数据以避免下载
        }
        
        # 生成数据
        dataset = []
        for prompt_len in test_config['prompt_lengths']:
            for gen_len in test_config['generation_lengths']:
                samples = generator.generate_synthetic_samples(
                    num_samples=test_config['samples_per_config'],
                    prompt_length=prompt_len,
                    generation_length=gen_len
                )
                dataset.extend(samples)
        
        print(f"生成了 {len(dataset)} 条测试数据")
        
        # 保存测试数据
        test_data_dir = Path("data/test")
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = test_data_dir / "demo_dataset.jsonl"
        with open(test_file, 'w', encoding='utf-8') as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + '\\n')
        
        print(f"测试数据保存到: {test_file}")
        print("✓ 数据生成测试通过")
        return True, test_file
        
    except Exception as e:
        print(f"✗ 数据生成测试失败: {str(e)}")
        return False, None

def test_simple_inference():
    """测试简单推理"""
    print("\\n" + "="*60)
    print("测试4: 简单推理")
    print("="*60)
    
    try:
        # 初始化模型管理器
        model_manager = ModelManager()
        model_manager.load_model()
        
        # 准备测试输入
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly",
            "Machine learning algorithms can help us"
        ]
        
        print(f"测试推理，共 {len(test_prompts)} 条样本...")
        
        results = []
        for i, prompt in enumerate(test_prompts):
            print(f"  处理样本 {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            start_time = time.time()
            
            # 准备输入
            inputs = model_manager.prepare_inputs([prompt])
            
            # 执行推理
            with torch.no_grad():
                outputs = model_manager.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.8
                )
            
            # 解码输出
            generated_text = model_manager.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            inference_time = time.time() - start_time
            
            # 计算token数量
            prompt_tokens = inputs['input_ids'].shape[1]
            total_tokens = outputs.shape[1]
            generated_tokens = total_tokens - prompt_tokens
            
            result = {
                'prompt': prompt,
                'generated_text': generated_text,
                'prompt_tokens': prompt_tokens,
                'generated_tokens': generated_tokens,
                'inference_time': inference_time,
                'throughput': generated_tokens / inference_time if inference_time > 0 else 0
            }
            
            results.append(result)
            print(f"    生成文本: {generated_text[len(prompt):].strip()[:100]}...")
            print(f"    推理时间: {inference_time:.3f}s, 吞吐量: {result['throughput']:.2f} tokens/s")
        
        # 计算平均性能
        avg_time = sum(r['inference_time'] for r in results) / len(results)
        avg_throughput = sum(r['throughput'] for r in results) / len(results)
        
        print(f"\\n推理性能摘要:")
        print(f"  平均推理时间: {avg_time:.3f}s")
        print(f"  平均吞吐量: {avg_throughput:.2f} tokens/s")
        
        print("✓ 简单推理测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 简单推理测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_monitoring():
    """测试内存监控"""
    print("\\n" + "="*60)
    print("测试5: 内存监控")
    print("="*60)
    
    try:
        from src.utils.performance_monitor import PerformanceMonitor
        
        # 初始化性能监控器
        monitor = PerformanceMonitor()
        
        print("开始内存监控...")
        monitor.start_monitoring()
        
        # 模拟一些计算
        if torch.cuda.is_available():
            # 创建一些张量进行计算
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            
            time.sleep(2)  # 等待2秒
            
            # 清理
            del x, y, z
            torch.cuda.empty_cache()
        
        time.sleep(1)
        monitor.stop_monitoring()
        
        # 获取监控结果
        stats = monitor.get_resource_stats()
        
        print("内存监控结果:")
        if 'gpu_memory_usage' in stats:
            gpu_stats = stats['gpu_memory_usage']
            print(f"  GPU显存:")
            print(f"    最大使用: {max(gpu_stats):.2f} MB")
            print(f"    平均使用: {sum(gpu_stats)/len(gpu_stats):.2f} MB")
        
        if 'cpu_memory_usage' in stats:
            cpu_stats = stats['cpu_memory_usage']
            print(f"  CPU内存:")
            print(f"    最大使用: {max(cpu_stats):.2f} MB")
            print(f"    平均使用: {sum(cpu_stats)/len(cpu_stats):.2f} MB")
        
        print("✓ 内存监控测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 内存监控测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("开始GPT分布式推理系统演示")
    print("="*80)
    
    # 检查环境
    print("环境检查:")
    print(f"  Python版本: {sys.version}")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
    
    # 运行测试
    tests = [
        ("模型加载", test_model_loading),
        ("并行策略", test_parallel_strategies), 
        ("数据生成", test_data_generation),
        ("简单推理", test_simple_inference),
        ("内存监控", test_memory_monitoring)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            if test_name == "数据生成":
                success, data_file = test_func()
                results[test_name] = success
            else:
                results[test_name] = test_func()
        except KeyboardInterrupt:
            print(f"\\n用户中断测试")
            break
        except Exception as e:
            print(f"\\n测试 {test_name} 出现异常: {str(e)}")
            results[test_name] = False
    
    # 总结结果
    print("\\n" + "="*80)
    print("测试结果总结")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常。")
        print("\\n下一步可以运行:")
        print("  python scripts/run_distributed_inference.py --strategy tensor_data_hybrid")
    else:
        print("⚠️  部分测试失败，请检查错误信息并修复相关问题。")
    
    print("="*80)

if __name__ == "__main__":
    main()
