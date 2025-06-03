#!/usr/bin/env python3
"""
多GPU数据并行测试
专门测试数据并行策略的性能
"""

import os
import sys
import json
import torch
import time
from datetime import datetime
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from models.model_manager import ModelManager
from inference.inference_engine import InferenceEngine
from utils.performance_monitor import PerformanceMonitor

def test_multi_gpu_data_parallel():
    """测试多GPU数据并行推理"""
    
    # 检查GPU环境
    if not torch.cuda.is_available():
        print("错误：未检测到CUDA GPU")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU")
    
    # 设置测试参数
    test_configs = [
        {"num_gpus": 1, "batch_size": 4, "description": "单GPU基准"},
        {"num_gpus": 2, "batch_size": 8, "description": "双GPU数据并行"},
    ]
    
    if gpu_count >= 4:
        test_configs.append({"num_gpus": 4, "batch_size": 16, "description": "四GPU数据并行"})
    
    # 加载测试数据
    test_samples = []
    dataset_file = "data/datasets/benchmark_dataset_config_0.jsonl"
    
    if not os.path.exists(dataset_file):
        print(f"错误：找不到数据集文件 {dataset_file}")
        return
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 50:  # 使用50个样本
                break
            test_samples.append(json.loads(line.strip()))
    
    print(f"加载了 {len(test_samples)} 个测试样本")
    
    results = []
    
    # 测试每个配置
    for config in test_configs:
        num_gpus = config["num_gpus"]
        batch_size = config["batch_size"]
        description = config["description"]
        
        if num_gpus > gpu_count:
            print(f"跳过 {description}：需要 {num_gpus} 个GPU")
            continue
        
        print(f"\n{'='*50}")
        print(f"测试: {description}")
        print(f"GPU数量: {num_gpus}, 批次大小: {batch_size}")
        print(f"{'='*50}")
        
        try:
            # 初始化组件
            model_manager = ModelManager()
            performance_monitor = PerformanceMonitor()
            
            # 加载模型
            print("正在加载模型...")
            model, tokenizer = model_manager.load_model("gpt2-xl")
            
            # 如果是多GPU，使用DataParallel
            if num_gpus > 1:
                model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
                print(f"启用DataParallel，使用GPU: {list(range(num_gpus))}")
            
            # 初始化推理引擎
            inference_engine = InferenceEngine(
                model=model,
                tokenizer=tokenizer,
                strategy="data_parallel",
                num_gpus=num_gpus
            )
            
            # 开始性能监控
            start_time = time.time()
            performance_monitor.start_monitoring()
            
            # 运行推理测试
            total_tokens = 0
            inference_results = []
            
            for i in range(0, len(test_samples), batch_size):
                batch = test_samples[i:i+batch_size]
                prompts = [sample["prompt"] for sample in batch]
                
                batch_start = time.time()
                
                # 运行批量推理
                outputs = inference_engine.run_batch_inference(
                    prompts=prompts,
                    max_length=32,
                    temperature=0.8
                )
                
                batch_time = time.time() - batch_start
                
                # 计算token数量
                for output in outputs:
                    total_tokens += len(tokenizer.encode(output))
                
                inference_results.extend(outputs)
                
                print(f"批次 {i//batch_size + 1}: {len(outputs)} 个样本, 用时 {batch_time:.2f}s")
            
            # 停止监控
            total_time = time.time() - start_time
            performance_stats = performance_monitor.stop_monitoring()
            
            # 计算性能指标
            throughput = total_tokens / total_time
            avg_latency = total_time / len(inference_results)
            samples_per_sec = len(inference_results) / total_time
            
            # 保存结果
            result = {
                "config": config,
                "performance": {
                    "total_time": total_time,
                    "total_tokens": total_tokens,
                    "total_samples": len(inference_results),
                    "throughput_tokens_per_sec": throughput,
                    "avg_latency": avg_latency,
                    "samples_per_sec": samples_per_sec
                },
                "system_stats": performance_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            print(f"\n结果:")
            print(f"  总时间: {total_time:.2f}s")
            print(f"  吞吐量: {throughput:.2f} tokens/s")
            print(f"  平均延迟: {avg_latency:.3f}s")
            print(f"  样本/秒: {samples_per_sec:.2f}")
            
            if 'gpu_utilization' in performance_stats:
                print(f"  GPU利用率: {performance_stats['gpu_utilization']:.1f}%")
            
            # 清理内存
            del model, inference_engine
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 保存完整结果
    results_file = f"results/multi_gpu_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成对比报告
    print(f"\n{'='*70}")
    print("多GPU性能对比报告")
    print(f"{'='*70}")
    print(f"{'配置':<20} {'吞吐量':<15} {'延迟':<10} {'GPU利用率':<10} {'加速比':<10}")
    print(f"{'-'*70}")
    
    baseline_throughput = None
    for result in results:
        config = result["config"]
        perf = result["performance"]
        
        throughput = perf["throughput_tokens_per_sec"]
        latency = perf["avg_latency"]
        
        gpu_util = "N/A"
        if 'gpu_utilization' in result["system_stats"]:
            gpu_util = f"{result['system_stats']['gpu_utilization']:.1f}%"
        
        # 计算加速比
        if baseline_throughput is None:
            baseline_throughput = throughput
            speedup = "1.0x"
        else:
            speedup = f"{throughput / baseline_throughput:.1f}x"
        
        print(f"{config['description']:<20} {throughput:<15.2f} {latency:<10.3f} {gpu_util:<10} {speedup:<10}")
    
    print(f"\n结果已保存到: {results_file}")

if __name__ == "__main__":
    test_multi_gpu_data_parallel()
