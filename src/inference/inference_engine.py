"""
分布式推理引擎
支持多种分布式策略和性能测量
"""

import torch
import torch.distributed as dist
import time
import logging
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from ..models import ModelManager
from ..utils import PerformanceMonitor
from ..utils.communication_profiler import CommunicationProfiler

@dataclass
class InferenceResult:
    """推理结果数据类"""
    sample_id: str
    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_time: float
    first_token_time: float
    throughput: float
    memory_usage: Dict[str, float]
    gpu_utilization: float
    communication_time: Dict[str, float]

class DistributedInferenceEngine:
    def __init__(self, 
                 model_manager: ModelManager,
                 config_path: str = "config/inference_config.yaml"):
        """初始化分布式推理引擎"""
        self.model_manager = model_manager
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = self._setup_logger()
        self.performance_monitor = PerformanceMonitor()
        self.communication_profiler = CommunicationProfiler()
        
        # 初始化NVML（如果可用）
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except:
                self.nvml_initialized = False
                self.logger.warning("NVML初始化失败，GPU监控将不可用")
        else:
            self.nvml_initialized = False
        
        # 性能统计
        self.results = []
        self.warmup_completed = False
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        if not logger.handlers:
            # 控制台处理器
            if self.config['logging']['console_output']:
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # 文件处理器
            log_file = Path(self.config['logging']['log_file'])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def warmup(self) -> None:
        """预热模型"""
        self.logger.info("开始模型预热...")
        
        warmup_steps = self.config['performance']['warmup_steps']
        dummy_text = "This is a warmup text for the model. " * 10
        
        for i in range(warmup_steps):
            inputs = self.model_manager.prepare_inputs([dummy_text])
            
            with torch.no_grad():
                _ = self.model_manager.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=32,
                    do_sample=False
                )
            
            if i % 5 == 0:
                self.logger.info(f"预热进度: {i+1}/{warmup_steps}")
        
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        self.warmup_completed = True
        self.logger.info("模型预热完成")
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """测量内存使用情况"""
        memory_info = {}
        
        # CPU内存
        process = psutil.Process()
        memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        memory_info['cpu_memory_percent'] = process.memory_percent()
        
        # GPU内存
        if torch.cuda.is_available():
            memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_info['gpu_memory_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return memory_info
    
    def measure_gpu_utilization(self) -> float:
        """测量GPU利用率"""
        if not self.nvml_initialized:
            return 0.0
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except:
            return 0.0
    
    def measure_communication_overhead(self, operation: str) -> float:
        """测量通信开销"""
        if hasattr(self, 'communication_profiler'):
            return self.communication_profiler.measure_operation(operation)
        return 0.0
    
    def run_inference_benchmark(self,
                                test_data: List[Dict],
                                batch_size: int = 1,
                                num_iterations: int = 100) -> List[InferenceResult]:
        """运行推理基准测试"""
        if not self.warmup_completed:
            self.warmup()
        
        self.logger.info(f"开始推理基准测试，数据量: {len(test_data)}, 批次大小: {batch_size}")
        
        results = []
        total_samples = min(len(test_data), num_iterations)
        
        # 开始性能监控
        self.performance_monitor.start_monitoring()
        
        try:
            for i in range(0, total_samples, batch_size):
                batch_data = test_data[i:i+batch_size]
                batch_results = self._process_batch(batch_data, i)
                results.extend(batch_results)
                
                if (i // batch_size + 1) % 10 == 0:
                    progress = (i + batch_size) / total_samples * 100
                    self.logger.info(f"推理进度: {progress:.1f}% ({i + batch_size}/{total_samples})")
                
                # 定期清理GPU缓存
                if (i // batch_size + 1) % 20 == 0:
                    torch.cuda.empty_cache()
        
        finally:
            # 停止性能监控
            self.performance_monitor.stop_monitoring()
        
        self.logger.info(f"推理基准测试完成，共处理 {len(results)} 个样本")
        return results
    
    def _process_batch(self, batch_data: List[Dict], batch_idx: int) -> List[InferenceResult]:
        """处理单个批次"""
        batch_results = []
        
        for sample_idx, sample in enumerate(batch_data):
            sample_id = f"batch_{batch_idx}_sample_{sample_idx}"
            
            # 开始通信分析
            if hasattr(self, 'communication_profiler'):
                self.communication_profiler.start_profiling()
            
            # 记录开始时间
            start_time = time.time()
            start_memory = self.measure_memory_usage()
            start_gpu_util = self.measure_gpu_utilization()
            
            # 准备输入
            prompt = sample.get('prompt', '')
            max_new_tokens = sample.get('max_new_tokens', 32)
            
            inputs = self.model_manager.prepare_inputs([prompt])
            prompt_tokens = inputs['input_ids'].shape[1]
            
            # 记录首令牌时间
            first_token_start = time.time()
            
            # 执行推理
            with torch.no_grad():
                outputs = self.model_manager.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.model_manager.tokenizer.pad_token_id
                )
            
            first_token_time = time.time() - first_token_start
            
            # 解码生成的文本
            generated_tokens = outputs[0][prompt_tokens:]
            generated_text = self.model_manager.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # 记录结束时间和资源使用
            end_time = time.time()
            total_time = end_time - start_time
            end_memory = self.measure_memory_usage()
            end_gpu_util = self.measure_gpu_utilization()
            
            # 计算性能指标
            generated_token_count = len(generated_tokens)
            throughput = generated_token_count / total_time if total_time > 0 else 0
            
            # 获取通信开销
            communication_times = {}
            if hasattr(self, 'communication_profiler'):
                communication_times = self.communication_profiler.get_communication_times()
                self.communication_profiler.stop_profiling()
            
            # 创建结果对象
            result = InferenceResult(
                sample_id=sample_id,
                prompt=prompt,
                generated_text=generated_text,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_token_count,
                total_time=total_time,
                first_token_time=first_token_time,
                throughput=throughput,
                memory_usage={
                    'start': start_memory,
                    'end': end_memory,
                    'gpu_memory_allocated_mb': end_memory.get('gpu_memory_allocated_mb', 0)
                },
                gpu_utilization=(start_gpu_util + end_gpu_util) / 2,
                communication_time=communication_times
            )
            
            batch_results.append(result)
        
        return batch_results
    
    def measure_communication_overhead(self, operation: str) -> float:
        """测量通信开销"""
        if not dist.is_initialized():
            return 0.0
        
        start_time = time.perf_counter()
        
        if operation == "allreduce":
            # 模拟allreduce操作
            dummy_tensor = torch.randn(1000, device=self.model_manager.device)
            dist.all_reduce(dummy_tensor)
        elif operation == "broadcast":
            # 模拟broadcast操作
            dummy_tensor = torch.randn(1000, device=self.model_manager.device)
            dist.broadcast(dummy_tensor, src=0)
        elif operation == "barrier":
            # 同步屏障
            dist.barrier()
        
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # 转换为毫秒
    
    def run_single_inference(self, 
                           sample: Dict[str, Any],
                           measure_performance: bool = True) -> InferenceResult:
        """运行单次推理"""
        prompt = sample['prompt']
        generation_length = sample['generation_length']
        sample_id = sample['id']
        
        # 准备输入
        inputs = self.model_manager.prepare_inputs([prompt])
        prompt_tokens = inputs['input_ids'].shape[1]
        
        # 性能测量开始
        start_time = time.perf_counter()
        
        # 测量首token时间
        first_token_time = None
        memory_before = self.measure_memory_usage() if measure_performance else {}
        gpu_util_before = self.measure_gpu_utilization() if measure_performance else 0.0
        
        # 生成文本
        with torch.no_grad():
            # 记录首token时间（简化实现）
            first_token_start = time.perf_counter()
            
            outputs = self.model_manager.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=generation_length,
                **self.config['inference']
            )
            
            # 简化的首token时间计算
            first_token_time = (time.perf_counter() - first_token_start) * 1000
        
        # 性能测量结束
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 解码生成的文本
        generated_tokens = outputs.shape[1] - prompt_tokens
        generated_text = self.model_manager.tokenizer.decode(
            outputs[0][prompt_tokens:], 
            skip_special_tokens=True
        )
        
        # 计算吞吐量
        throughput = generated_tokens / (total_time / 1000) if total_time > 0 else 0.0
        
        # 测量性能指标
        memory_after = self.measure_memory_usage() if measure_performance else {}
        gpu_util_after = self.measure_gpu_utilization() if measure_performance else 0.0
        
        # 测量通信开销
        communication_time = {}
        if measure_performance and dist.is_initialized():
            communication_time = {
                'allreduce': self.measure_communication_overhead('allreduce'),
                'broadcast': self.measure_communication_overhead('broadcast'),
                'barrier': self.measure_communication_overhead('barrier')
            }
        
        # 创建结果
        result = InferenceResult(
            sample_id=sample_id,
            prompt=prompt,
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_time=total_time,
            first_token_time=first_token_time,
            throughput=throughput,
            memory_usage=memory_after,
            gpu_utilization=gpu_util_after,
            communication_time=communication_time
        )
        
        return result
    
    def run_batch_inference(self, 
                          samples: List[Dict[str, Any]],
                          batch_size: Optional[int] = None) -> List[InferenceResult]:
        """运行批量推理"""
        if batch_size is None:
            batch_size = self.config['inference']['batch_size']
        
        if not self.warmup_completed:
            self.warmup()
        
        results = []
        total_batches = (len(samples) + batch_size - 1) // batch_size
        
        self.logger.info(f"开始批量推理: {len(samples)} 个样本, 批次大小: {batch_size}")
        
        for batch_idx in range(0, len(samples), batch_size):
            batch_samples = samples[batch_idx:batch_idx + batch_size]
            batch_results = []
            
            for sample in batch_samples:
                result = self.run_single_inference(sample, measure_performance=True)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # 同步所有进程（如果是分布式）
            if dist.is_initialized():
                dist.barrier()
            
            current_batch = batch_idx // batch_size + 1
            self.logger.info(f"完成批次 {current_batch}/{total_batches}")
            
            # 清理内存
            if current_batch % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        self.results.extend(results)
        self.logger.info(f"批量推理完成，总共处理 {len(results)} 个样本")
        
        return results
    
    def run_performance_benchmark(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行性能基准测试"""
        self.logger.info("开始性能基准测试")
        
        # 按配置分组样本
        config_groups = {}
        for sample in samples:
            config_id = sample['config_id']
            if config_id not in config_groups:
                config_groups[config_id] = []
            config_groups[config_id].append(sample)
        
        benchmark_results = {}
        
        for config_id, config_samples in config_groups.items():
            self.logger.info(f"测试配置 {config_id}: {len(config_samples)} 个样本")
            
            # 获取配置信息
            sample_config = config_samples[0]
            prompt_length = sample_config['prompt_length']
            generation_length = sample_config['generation_length']
            
            # 限制测试样本数量
            num_iterations = min(
                len(config_samples),
                self.config['performance']['num_iterations']
            )
            test_samples = config_samples[:num_iterations]
            
            # 运行推理
            results = self.run_batch_inference(test_samples)
            
            # 计算统计指标
            stats = self._calculate_statistics(results)
            
            benchmark_results[config_id] = {
                'config': {
                    'prompt_length': prompt_length,
                    'generation_length': generation_length
                },
                'num_samples': len(results),
                'statistics': stats
            }
        
        # 保存结果
        self._save_benchmark_results(benchmark_results)
        
        self.logger.info("性能基准测试完成")
        return benchmark_results
    
    def _calculate_statistics(self, results: List[InferenceResult]) -> Dict[str, Any]:
        """计算统计指标"""
        if not results:
            return {}
        
        # 提取各项指标
        throughputs = [r.throughput for r in results]
        total_times = [r.total_time for r in results]
        first_token_times = [r.first_token_time for r in results if r.first_token_time]
        gpu_utilizations = [r.gpu_utilization for r in results]
        
        # 内存使用统计
        memory_usages = {}
        if results[0].memory_usage:
            for key in results[0].memory_usage.keys():
                values = [r.memory_usage.get(key, 0) for r in results]
                memory_usages[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'max': float(np.max(values)),
                    'min': float(np.min(values))
                }
        
        # 通信开销统计
        communication_stats = {}
        if results[0].communication_time:
            for key in results[0].communication_time.keys():
                values = [r.communication_time.get(key, 0) for r in results]
                communication_stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'max': float(np.max(values)),
                    'min': float(np.min(values))
                }
        
        stats = {
            'throughput': {
                'mean': float(np.mean(throughputs)),
                'std': float(np.std(throughputs)),
                'max': float(np.max(throughputs)),
                'min': float(np.min(throughputs)),
                'p50': float(np.percentile(throughputs, 50)),
                'p95': float(np.percentile(throughputs, 95)),
                'p99': float(np.percentile(throughputs, 99))
            },
            'latency': {
                'total_time': {
                    'mean': float(np.mean(total_times)),
                    'std': float(np.std(total_times)),
                    'max': float(np.max(total_times)),
                    'min': float(np.min(total_times)),
                    'p50': float(np.percentile(total_times, 50)),
                    'p95': float(np.percentile(total_times, 95)),
                    'p99': float(np.percentile(total_times, 99))
                }
            },
            'gpu_utilization': {
                'mean': float(np.mean(gpu_utilizations)),
                'std': float(np.std(gpu_utilizations)),
                'max': float(np.max(gpu_utilizations)),
                'min': float(np.min(gpu_utilizations))
            },
            'memory_usage': memory_usages,
            'communication_overhead': communication_stats
        }
        
        if first_token_times:
            stats['latency']['first_token_time'] = {
                'mean': float(np.mean(first_token_times)),
                'std': float(np.std(first_token_times)),
                'max': float(np.max(first_token_times)),
                'min': float(np.min(first_token_times)),
                'p50': float(np.percentile(first_token_times, 50)),
                'p95': float(np.percentile(first_token_times, 95)),
                'p99': float(np.percentile(first_token_times, 99))
            }
        
        return stats
    
    def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """保存基准测试结果"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = results_dir / filename
        
        # 添加元数据
        output_data = {
            'metadata': {
                'timestamp': timestamp,
                'model_config': self.model_manager.config,
                'inference_config': self.config,
                'model_info': self.model_manager.get_model_info()
            },
            'results': results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"基准测试结果已保存到: {filepath}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.results:
            return {}
        
        # 计算平均指标
        total_throughput = sum(r.throughput for r in self.results)
        total_latency = sum(r.total_time for r in self.results)
        total_first_token_latency = sum(r.first_token_time for r in self.results)
        total_gpu_util = sum(r.gpu_utilization for r in self.results)
        
        avg_memory = {}
        if self.results[0].memory_usage:
            memory_keys = self.results[0].memory_usage.keys()
            for key in memory_keys:
                values = [r.memory_usage.get(key, 0) for r in self.results if r.memory_usage]
                if values:
                    avg_memory[key] = sum(values) / len(values)
        
        # 计算通信开销
        comm_times = {}
        for result in self.results:
            for comm_type, time_val in result.communication_time.items():
                if comm_type not in comm_times:
                    comm_times[comm_type] = []
                comm_times[comm_type].append(time_val)
        
        avg_comm_times = {}
        for comm_type, times in comm_times.items():
            avg_comm_times[comm_type] = sum(times) / len(times) if times else 0
        
        summary = {
            'total_samples': len(self.results),
            'avg_throughput': total_throughput / len(self.results),
            'avg_latency': total_latency / len(self.results),
            'avg_first_token_latency': total_first_token_latency / len(self.results),
            'avg_gpu_utilization': total_gpu_util / len(self.results),
            'avg_memory_usage': avg_memory,
            'avg_communication_times': avg_comm_times,
            'total_tokens_generated': sum(r.generated_tokens for r in self.results),
            'total_inference_time': total_latency
        }
        
        return summary
    
    def save_results(self, filepath: str) -> None:
        """保存推理结果"""
        results_data = {
            'results': [result.__dict__ for result in self.results],
            'summary': self.get_performance_summary(),
            'config': self.config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存到: {filepath}")
    
    def cleanup(self) -> None:
        """清理资源"""
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.stop_monitoring()
        
        torch.cuda.empty_cache()
        self.logger.info("推理引擎资源清理完成")
