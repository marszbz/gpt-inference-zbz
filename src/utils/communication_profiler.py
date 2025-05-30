"""
通信性能分析器
用于分析分布式训练中的通信开销
"""

import time
import torch
import torch.distributed as dist
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
import threading
from contextlib import contextmanager

class CommunicationProfiler:
    """通信性能分析器"""
    
    def __init__(self):
        """初始化通信分析器"""
        self.logger = self._setup_logger()
        self.communication_times = defaultdict(list)
        self.current_operations = {}
        self.is_profiling = False
        self.lock = threading.Lock()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_profiling(self) -> None:
        """开始性能分析"""
        with self.lock:
            self.is_profiling = True
            self.communication_times.clear()
            self.current_operations.clear()
        
        self.logger.debug("开始通信性能分析")
    
    def stop_profiling(self) -> None:
        """停止性能分析"""
        with self.lock:
            self.is_profiling = False
            # 完成所有未完成的操作
            for op_name in list(self.current_operations.keys()):
                self.end_operation(op_name)
        
        self.logger.debug("停止通信性能分析")
    
    def start_operation(self, operation_name: str) -> None:
        """开始记录操作"""
        if not self.is_profiling:
            return
        
        with self.lock:
            self.current_operations[operation_name] = time.time()
    
    def end_operation(self, operation_name: str) -> float:
        """结束记录操作并返回耗时"""
        if not self.is_profiling:
            return 0.0
        
        with self.lock:
            if operation_name in self.current_operations:
                start_time = self.current_operations.pop(operation_name)
                duration = (time.time() - start_time) * 1000  # 转换为毫秒
                self.communication_times[operation_name].append(duration)
                return duration
            
        return 0.0
    
    @contextmanager
    def measure_operation(self, operation_name: str):
        """上下文管理器，用于测量操作时间"""
        self.start_operation(operation_name)
        try:
            yield
        finally:
            self.end_operation(operation_name)
    
    def measure_allreduce(self, tensor: torch.Tensor, group: Optional[Any] = None) -> float:
        """测量AllReduce操作"""
        if not self.is_profiling or not dist.is_initialized():
            return 0.0
        
        operation_name = f"allreduce_{tensor.numel()}_elements"
        
        with self.measure_operation(operation_name):
            dist.all_reduce(tensor, group=group)
        
        return self.communication_times[operation_name][-1] if self.communication_times[operation_name] else 0.0
    
    def measure_allgather(self, tensor: torch.Tensor, group: Optional[Any] = None) -> float:
        """测量AllGather操作"""
        if not self.is_profiling or not dist.is_initialized():
            return 0.0
        
        operation_name = f"allgather_{tensor.numel()}_elements"
        world_size = dist.get_world_size(group) if group else dist.get_world_size()
        
        # 创建输出张量列表
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        
        with self.measure_operation(operation_name):
            dist.all_gather(tensor_list, tensor, group=group)
        
        return self.communication_times[operation_name][-1] if self.communication_times[operation_name] else 0.0
    
    def measure_broadcast(self, tensor: torch.Tensor, src: int, group: Optional[Any] = None) -> float:
        """测量Broadcast操作"""
        if not self.is_profiling or not dist.is_initialized():
            return 0.0
        
        operation_name = f"broadcast_{tensor.numel()}_elements"
        
        with self.measure_operation(operation_name):
            dist.broadcast(tensor, src=src, group=group)
        
        return self.communication_times[operation_name][-1] if self.communication_times[operation_name] else 0.0
    
    def measure_reduce_scatter(self, output: torch.Tensor, input_list: List[torch.Tensor], 
                              group: Optional[Any] = None) -> float:
        """测量ReduceScatter操作"""
        if not self.is_profiling or not dist.is_initialized():
            return 0.0
        
        total_elements = sum(t.numel() for t in input_list)
        operation_name = f"reduce_scatter_{total_elements}_elements"
        
        with self.measure_operation(operation_name):
            dist.reduce_scatter(output, input_list, group=group)
        
        return self.communication_times[operation_name][-1] if self.communication_times[operation_name] else 0.0
    
    def get_communication_times(self) -> Dict[str, float]:
        """获取平均通信时间"""
        avg_times = {}
        
        with self.lock:
            for op_name, times in self.communication_times.items():
                if times:
                    avg_times[op_name] = sum(times) / len(times)
                else:
                    avg_times[op_name] = 0.0
        
        return avg_times
    
    def get_detailed_stats(self) -> Dict[str, Dict[str, float]]:
        """获取详细统计信息"""
        detailed_stats = {}
        
        with self.lock:
            for op_name, times in self.communication_times.items():
                if times:
                    detailed_stats[op_name] = {
                        'count': len(times),
                        'total_time_ms': sum(times),
                        'avg_time_ms': sum(times) / len(times),
                        'min_time_ms': min(times),
                        'max_time_ms': max(times),
                        'std_time_ms': self._calculate_std(times)
                    }
                else:
                    detailed_stats[op_name] = {
                        'count': 0,
                        'total_time_ms': 0.0,
                        'avg_time_ms': 0.0,
                        'min_time_ms': 0.0,
                        'max_time_ms': 0.0,
                        'std_time_ms': 0.0
                    }
        
        return detailed_stats
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def print_summary(self) -> None:
        """打印通信性能摘要"""
        stats = self.get_detailed_stats()
        
        if not stats:
            self.logger.info("没有收集到通信统计信息")
            return
        
        print("\n" + "="*80)
        print("通信性能分析摘要")
        print("="*80)
        
        total_comm_time = 0.0
        for op_name, stat in stats.items():
            print(f"\n{op_name}:")
            print(f"  调用次数: {stat['count']}")
            print(f"  总时间: {stat['total_time_ms']:.2f} ms")
            print(f"  平均时间: {stat['avg_time_ms']:.2f} ms")
            print(f"  最小时间: {stat['min_time_ms']:.2f} ms")
            print(f"  最大时间: {stat['max_time_ms']:.2f} ms")
            print(f"  标准差: {stat['std_time_ms']:.2f} ms")
            
            total_comm_time += stat['total_time_ms']
        
        print(f"\n总通信时间: {total_comm_time:.2f} ms")
        print("="*80 + "\n")
    
    def export_to_dict(self) -> Dict[str, Any]:
        """导出为字典格式"""
        return {
            'communication_times': dict(self.communication_times),
            'detailed_stats': self.get_detailed_stats(),
            'avg_times': self.get_communication_times()
        }
    
    def reset(self) -> None:
        """重置统计信息"""
        with self.lock:
            self.communication_times.clear()
            self.current_operations.clear()
        
        self.logger.debug("通信统计信息已重置")


class DistributedCommunicationMonitor:
    """分布式通信监控器"""
    
    def __init__(self, profiler: CommunicationProfiler):
        """初始化监控器"""
        self.profiler = profiler
        self.original_functions = {}
        self.is_monitoring = False
    
    def start_monitoring(self) -> None:
        """开始监控分布式通信"""
        if not dist.is_initialized() or self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Hook分布式函数
        self._hook_distributed_functions()
        
        self.profiler.start_profiling()
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # 恢复原始函数
        self._restore_distributed_functions()
        
        self.profiler.stop_profiling()
    
    def _hook_distributed_functions(self) -> None:
        """Hook分布式函数以进行监控"""
        # 保存原始函数
        self.original_functions['all_reduce'] = dist.all_reduce
        self.original_functions['all_gather'] = dist.all_gather
        self.original_functions['broadcast'] = dist.broadcast
        self.original_functions['reduce_scatter'] = dist.reduce_scatter
        
        # 创建包装函数
        def wrapped_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
            if async_op:
                return self.original_functions['all_reduce'](tensor, op, group, async_op)
            
            self.profiler.measure_allreduce(tensor, group)
            return None
        
        def wrapped_all_gather(tensor_list, tensor, group=None, async_op=False):
            if async_op:
                return self.original_functions['all_gather'](tensor_list, tensor, group, async_op)
            
            self.profiler.measure_allgather(tensor, group)
            return None
        
        def wrapped_broadcast(tensor, src, group=None, async_op=False):
            if async_op:
                return self.original_functions['broadcast'](tensor, src, group, async_op)
            
            self.profiler.measure_broadcast(tensor, src, group)
            return None
        
        def wrapped_reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=None, async_op=False):
            if async_op:
                return self.original_functions['reduce_scatter'](output, input_list, op, group, async_op)
            
            self.profiler.measure_reduce_scatter(output, input_list, group)
            return None
        
        # 替换函数
        dist.all_reduce = wrapped_all_reduce
        dist.all_gather = wrapped_all_gather
        dist.broadcast = wrapped_broadcast
        dist.reduce_scatter = wrapped_reduce_scatter
    
    def _restore_distributed_functions(self) -> None:
        """恢复原始分布式函数"""
        for func_name, original_func in self.original_functions.items():
            setattr(dist, func_name, original_func)
        
        self.original_functions.clear()
