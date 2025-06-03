#!/usr/bin/env python3
"""
简化的多GPU推理测试
测试数据并行策略的性能
"""

import os
import sys
import json
import torch
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_single_vs_multi_gpu():
    """对比单GPU和多GPU性能"""
    
    # 检查GPU环境
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA GPU")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU")
    
    # 测试配置
    test_configs = [
        {"name": "单GPU", "use_multi_gpu": False, "batch_size": 4},
    ]
    
    if gpu_count >= 2:
        test_configs.append({"name": "多GPU数据并行", "use_multi_gpu": True, "batch_size": 8})
    
    # 加载测试样本
    from src.utils.data_loader import DataLoader
    
    dataset_files = list(Path("data/datasets").glob("benchmark_dataset_config_*.jsonl"))
    if not dataset_files:
        print("错误：未找到数据集文件")
        return
    
    # 加载第一个配置的数据
    data_loader = DataLoader(str(dataset_files[0]))
    samples = data_loader.load_samples()[:20]  # 使用20个样本进行快速测试
    
    print(f"加载了 {len(samples)} 个测试样本")
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"测试配置: {config['name']}")
        print(f"批次大小: {config['batch_size']}")
        print(f"{'='*50}")
        
        try:
            # 导入模型管理器
            from src.models.model_manager import ModelManager
            
            # 初始化模型管理器
            model_manager = ModelManager()
            
            # 加载模型
            print("正在加载模型...")
            model_manager.load_model()
            
            # 如果使用多GPU，包装模型
            if config["use_multi_gpu"] and gpu_count > 1:
                model_manager.model = torch.nn.DataParallel(
                    model_manager.model, 
                    device_ids=list(range(min(gpu_count, 4)))  # 最多使用4个GPU
                )
                print(f"启用DataParallel，使用 {min(gpu_count, 4)} 个GPU")
            
            # 预热
            print("模型预热中...")
            with torch.no_grad():
                for i in range(3):
                    test_prompt = "This is a test prompt for warmup"
                    inputs = model_manager.prepare_inputs([test_prompt])
                    _ = model_manager.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=10
                    )
            
            # 开始性能测试
            batch_size = config["batch_size"]
            total_tokens = 0
            total_samples = 0
            
            start_time = time.time()
            
            # 按批次处理样本
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i+batch_size]
                batch_prompts = []
                
                for sample in batch_samples:
                    prompt = sample.prompt if hasattr(sample, 'prompt') else sample['prompt']
                    batch_prompts.append(prompt)
                
                batch_start = time.time()
                
                # 批量推理
                with torch.no_grad():
                    for prompt in batch_prompts:
                        inputs = model_manager.prepare_inputs([prompt])
                        outputs = model_manager.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_new_tokens=32,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.9
                        )
                        
                        # 计算生成的token数量
                        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                        total_tokens += len(generated_tokens)
                        total_samples += 1
                
                batch_time = time.time() - batch_start
                print(f"批次 {i//batch_size + 1}: {len(batch_prompts)} 样本, 用时 {batch_time:.2f}s")
            
            total_time = time.time() - start_time
            
            # 计算性能指标
            throughput = total_tokens / total_time
            avg_latency = total_time / total_samples
            samples_per_sec = total_samples / total_time
            
            # 获取GPU内存使用情况
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
            result = {
                "config": config,
                "performance": {
                    "total_time": total_time,
                    "total_tokens": total_tokens,
                    "total_samples": total_samples,
                    "throughput_tokens_per_sec": throughput,
                    "avg_latency_sec": avg_latency,
                    "samples_per_sec": samples_per_sec,
                    "gpu_memory_mb": gpu_memory_mb
                },
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            print(f"\n✅ {config['name']} 测试完成:")
            print(f"   总时间: {total_time:.2f}s")
            print(f"   吞吐量: {throughput:.2f} tokens/s")
            print(f"   平均延迟: {avg_latency:.3f}s")
            print(f"   样本处理速度: {samples_per_sec:.2f} samples/s")
            print(f"   GPU内存使用: {gpu_memory_mb:.1f} MB")
            
            # 清理内存
            del model_manager
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ {config['name']} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 生成对比报告
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print("🚀 多GPU性能对比报告")
        print(f"{'='*70}")
        
        single_gpu_result = results[0]
        multi_gpu_result = results[1]
        
        single_throughput = single_gpu_result["performance"]["throughput_tokens_per_sec"]
        multi_throughput = multi_gpu_result["performance"]["throughput_tokens_per_sec"]
        
        speedup = multi_throughput / single_throughput
        efficiency = speedup / gpu_count * 100
        
        print(f"单GPU吞吐量:     {single_throughput:.2f} tokens/s")
        print(f"多GPU吞吐量:     {multi_throughput:.2f} tokens/s")
        print(f"加速比:         {speedup:.2f}x")
        print(f"并行效率:       {efficiency:.1f}%")
        
        single_latency = single_gpu_result["performance"]["avg_latency_sec"]
        multi_latency = multi_gpu_result["performance"]["avg_latency_sec"]
        latency_improvement = (single_latency - multi_latency) / single_latency * 100
        
        print(f"单GPU平均延迟:   {single_latency:.3f}s")
        print(f"多GPU平均延迟:   {multi_latency:.3f}s")
        print(f"延迟改善:       {latency_improvement:.1f}%")
    
    # 保存结果
    results_file = f"results/multi_gpu_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 结果已保存到: {results_file}")

if __name__ == "__main__":
    print("🔥 开始多GPU推理性能对比测试")
    test_single_vs_multi_gpu()
    print("✅ 测试完成")
