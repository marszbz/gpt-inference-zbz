"""
性能监控工具
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

@dataclass
class PerformanceSnapshot:
    """性能快照数据类"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.snapshots = []
        self.monitor_thread = None
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化NVML
        self.nvml_available = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
            except:
                self.logger.warning("NVML初始化失败，GPU监控将不可用")
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.snapshots = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("性能监控已开始")
    
    def stop_monitoring(self) -> List[PerformanceSnapshot]:
        """停止监控并返回结果"""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info(f"性能监控已停止，收集了 {len(self.snapshots)} 个快照")
        return self.snapshots.copy()
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"监控过程中出错: {e}")
    
    def _take_snapshot(self) -> PerformanceSnapshot:
        """获取性能快照"""
        timestamp = time.time()
        
        # CPU和内存使用率
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU使用率
        gpu_usage = 0.0
        gpu_memory = 0.0
        if self.nvml_available:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_usage = float(utilization.gpu)
                gpu_memory = float(memory_info.used) / float(memory_info.total) * 100
            except:
                pass
        
        # 磁盘IO
        disk_io = {}
        try:
            disk_stats = psutil.disk_io_counters()
            if disk_stats:
                disk_io = {
                    'read_bytes': disk_stats.read_bytes,
                    'write_bytes': disk_stats.write_bytes,
                    'read_time': disk_stats.read_time,
                    'write_time': disk_stats.write_time
                }
        except:
            pass
        
        # 网络IO
        network_io = {}
        try:
            net_stats = psutil.net_io_counters()
            if net_stats:
                network_io = {
                    'bytes_sent': net_stats.bytes_sent,
                    'bytes_recv': net_stats.bytes_recv,
                    'packets_sent': net_stats.packets_sent,
                    'packets_recv': net_stats.packets_recv
                }
        except:
            pass
        
        return PerformanceSnapshot(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            disk_io=disk_io,
            network_io=network_io
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        if not self.snapshots:
            return {}
        
        # 提取各项指标
        cpu_usages = [s.cpu_usage for s in self.snapshots]
        memory_usages = [s.memory_usage for s in self.snapshots]
        gpu_usages = [s.gpu_usage for s in self.snapshots]
        gpu_memories = [s.gpu_memory for s in self.snapshots]
        
        stats = {
            'duration': self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            'num_snapshots': len(self.snapshots),
            'cpu_usage': {
                'mean': float(np.mean(cpu_usages)),
                'max': float(np.max(cpu_usages)),
                'min': float(np.min(cpu_usages)),
                'std': float(np.std(cpu_usages))
            },
            'memory_usage': {
                'mean': float(np.mean(memory_usages)),
                'max': float(np.max(memory_usages)),
                'min': float(np.min(memory_usages)),
                'std': float(np.std(memory_usages))
            },
            'gpu_usage': {
                'mean': float(np.mean(gpu_usages)),
                'max': float(np.max(gpu_usages)),
                'min': float(np.min(gpu_usages)),
                'std': float(np.std(gpu_usages))
            },
            'gpu_memory': {
                'mean': float(np.mean(gpu_memories)),
                'max': float(np.max(gpu_memories)),
                'min': float(np.min(gpu_memories)),
                'std': float(np.std(gpu_memories))
            }
        }
        
        return stats
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        system_info = {}
        
        # CPU信息
        system_info['cpu_count'] = psutil.cpu_count()
        system_info['cpu_count_logical'] = psutil.cpu_count(logical=True)
        system_info['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        
        # 内存信息
        memory = psutil.virtual_memory()
        system_info['memory_total_gb'] = memory.total / (1024**3)
        system_info['memory_available_gb'] = memory.available / (1024**3)
        system_info['memory_usage_percent'] = memory.percent
        
        # GPU信息
        system_info['gpu_available'] = self.nvml_available
        if self.nvml_available:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                system_info['gpu_count'] = device_count
                system_info['gpu_info'] = []
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_info = {
                        'index': i,
                        'name': name,
                        'memory_total_gb': memory_info.total / (1024**3),
                        'memory_free_gb': memory_info.free / (1024**3),
                        'memory_used_gb': memory_info.used / (1024**3)
                    }
                    system_info['gpu_info'].append(gpu_info)
            except Exception as e:
                self.logger.warning(f"获取GPU信息失败: {e}")
                system_info['gpu_count'] = 0
                system_info['gpu_info'] = []
        else:
            system_info['gpu_count'] = 0
            system_info['gpu_info'] = []
        
        return system_info


class CommunicationProfiler:
    """通信分析器（用于分布式环境）"""
    
    def __init__(self):
        self.communication_logs = []
        self.logger = logging.getLogger(__name__)
    
    def profile_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """分析通信操作"""
        start_time = time.perf_counter()
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000  # 转换为毫秒
        
        log_entry = {
            'operation': operation_name,
            'duration_ms': duration,
            'timestamp': time.time(),
            'success': success,
            'error': error
        }
        
        self.communication_logs.append(log_entry)
        
        if not success:
            self.logger.error(f"通信操作失败: {operation_name}, 错误: {error}")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        if not self.communication_logs:
            return {}
        
        # 按操作类型分组
        by_operation = defaultdict(list)
        for log in self.communication_logs:
            by_operation[log['operation']].append(log['duration_ms'])
        
        stats = {}
        for operation, durations in by_operation.items():
            stats[operation] = {
                'count': len(durations),
                'total_time_ms': sum(durations),
                'mean_time_ms': np.mean(durations),
                'max_time_ms': np.max(durations),
                'min_time_ms': np.min(durations),
                'std_time_ms': np.std(durations)
            }
        
        # 总体统计
        all_durations = [log['duration_ms'] for log in self.communication_logs]
        success_count = sum(1 for log in self.communication_logs if log['success'])
        
        stats['overall'] = {
            'total_operations': len(self.communication_logs),
            'successful_operations': success_count,
            'success_rate': success_count / len(self.communication_logs),
            'total_time_ms': sum(all_durations),
            'mean_time_ms': np.mean(all_durations),
            'max_time_ms': np.max(all_durations),
            'min_time_ms': np.min(all_durations)
        }
        
        return stats
    
    def clear_logs(self) -> None:
        """清空日志"""
        self.communication_logs = []


class ResourceTracker:
    """资源跟踪器"""
    
    def __init__(self):
        self.resource_history = []
        self.logger = logging.getLogger(__name__)
    
    def track_memory_usage(self) -> Dict[str, float]:
        """跟踪内存使用"""
        import torch
        
        memory_info = {}
        
        # 系统内存
        memory = psutil.virtual_memory()
        memory_info['system_memory_used_gb'] = memory.used / (1024**3)
        memory_info['system_memory_percent'] = memory.percent
        
        # GPU内存
        if torch.cuda.is_available():
            memory_info['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            memory_info['gpu_memory_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        
        # 记录历史
        self.resource_history.append({
            'timestamp': time.time(),
            'memory': memory_info
        })
        
        return memory_info
    
    def get_peak_memory_usage(self) -> Dict[str, float]:
        """获取峰值内存使用"""
        if not self.resource_history:
            return {}
        
        peak_usage = {}
        
        # 提取所有内存记录
        memory_keys = set()
        for record in self.resource_history:
            memory_keys.update(record['memory'].keys())
        
        for key in memory_keys:
            values = [record['memory'].get(key, 0) for record in self.resource_history]
            peak_usage[f'peak_{key}'] = max(values)
        
        return peak_usage
