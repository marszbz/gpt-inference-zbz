"""
并行策略管理器
支持多种分布式并行策略，针对RTX 3080进行优化
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import yaml
from pathlib import Path

class ParallelStrategy(Enum):
    """并行策略枚举"""
    PURE_DATA_PARALLEL = "pure_data_parallel"
    TENSOR_DATA_HYBRID = "tensor_data_hybrid" 
    PIPELINE_DATA_HYBRID = "pipeline_data_hybrid"
    FULL_MODEL_PARALLEL = "full_model_parallel"
    CUSTOM = "custom"

class ParallelStrategyManager:
    """并行策略管理器"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """初始化并行策略管理器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = self._setup_logger()
        self.world_size = self.config['distributed']['world_size']
        self.current_strategy = None
        self.device_placement = {}
        
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
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略配置"""
        if strategy_name in self.config['parallel_strategy']['custom_strategies']:
            return self.config['parallel_strategy']['custom_strategies'][strategy_name]
        else:
            # 使用默认策略
            return self.config['parallel_strategy']
    
    def validate_strategy(self, strategy_config: Dict[str, Any]) -> bool:
        """验证并行策略是否合理"""
        tp_size = strategy_config.get('tensor_parallel_size', 1)
        pp_size = strategy_config.get('pipeline_parallel_size', 1) 
        dp_enabled = strategy_config.get('data_parallel', True)
        
        # 检查总GPU数量
        total_model_parallel = tp_size * pp_size
        if total_model_parallel > self.world_size:
            self.logger.error(f"模型并行度 {total_model_parallel} 超过可用GPU数量 {self.world_size}")
            return False
        
        # 检查数据并行的合理性
        if dp_enabled and self.world_size % total_model_parallel != 0:
            self.logger.error(f"GPU数量 {self.world_size} 不能被模型并行度 {total_model_parallel} 整除")
            return False
        
        return True
    
    def calculate_device_placement(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """计算设备放置策略"""
        tp_size = strategy_config.get('tensor_parallel_size', 1)
        pp_size = strategy_config.get('pipeline_parallel_size', 1)
        dp_enabled = strategy_config.get('data_parallel', True)
        
        placement = {
            'tensor_parallel_size': tp_size,
            'pipeline_parallel_size': pp_size,
            'data_parallel_enabled': dp_enabled,
            'device_map': {}
        }
        
        if dp_enabled:
            dp_size = self.world_size // (tp_size * pp_size)
            placement['data_parallel_size'] = dp_size
        else:
            placement['data_parallel_size'] = 1
        
        # 为每个rank分配设备角色
        for rank in range(self.world_size):
            # 计算在模型并行组内的位置
            model_parallel_rank = rank % (tp_size * pp_size)
            tp_rank = model_parallel_rank % tp_size
            pp_rank = model_parallel_rank // tp_size
            
            # 计算数据并行组
            if dp_enabled:
                dp_rank = rank // (tp_size * pp_size)
            else:
                dp_rank = 0
            
            placement['device_map'][rank] = {
                'tensor_parallel_rank': tp_rank,
                'pipeline_parallel_rank': pp_rank,
                'data_parallel_rank': dp_rank,
                'gpu_id': rank
            }
        
        return placement
    
    def setup_parallel_groups(self, rank: int, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """设置并行进程组"""
        placement = self.calculate_device_placement(strategy_config)
        
        tp_size = placement['tensor_parallel_size']
        pp_size = placement['pipeline_parallel_size']
        dp_size = placement['data_parallel_size']
        
        groups = {
            'tensor_parallel_group': None,
            'pipeline_parallel_group': None,
            'data_parallel_group': None
        }
        
        # 创建张量并行组
        if tp_size > 1:
            for i in range(0, self.world_size, tp_size):
                tp_ranks = list(range(i, min(i + tp_size, self.world_size)))
                group = dist.new_group(tp_ranks)
                if rank in tp_ranks:
                    groups['tensor_parallel_group'] = group
        
        # 创建流水线并行组
        if pp_size > 1:
            for dp_group in range(dp_size):
                for tp_group in range(tp_size):
                    pp_ranks = []
                    for pp_rank in range(pp_size):
                        global_rank = dp_group * (tp_size * pp_size) + pp_rank * tp_size + tp_group
                        if global_rank < self.world_size:
                            pp_ranks.append(global_rank)
                    
                    if len(pp_ranks) > 1:
                        group = dist.new_group(pp_ranks)
                        if rank in pp_ranks:
                            groups['pipeline_parallel_group'] = group
        
        # 创建数据并行组
        if dp_size > 1:
            for model_parallel_rank in range(tp_size * pp_size):
                dp_ranks = []
                for dp_rank in range(dp_size):
                    global_rank = dp_rank * (tp_size * pp_size) + model_parallel_rank
                    if global_rank < self.world_size:
                        dp_ranks.append(global_rank)
                
                if len(dp_ranks) > 1:
                    group = dist.new_group(dp_ranks)
                    if rank in dp_ranks:
                        groups['data_parallel_group'] = group
        
        self.logger.info(f"Rank {rank} 并行组设置完成:")
        self.logger.info(f"  - 张量并行组: {groups['tensor_parallel_group'] is not None}")
        self.logger.info(f"  - 流水线并行组: {groups['pipeline_parallel_group'] is not None}")
        self.logger.info(f"  - 数据并行组: {groups['data_parallel_group'] is not None}")
        
        return groups
    
    def get_memory_optimization_config(self) -> Dict[str, Any]:
        """获取RTX 3080显存优化配置"""
        gpu_memory_gb = self.config.get('hardware', {}).get('gpu_memory_gb', 10)
        
        # RTX 3080特定的显存优化策略
        optimization_config = {
            'gradient_checkpointing': True,  # 激活检查点
            'cpu_offload': True,  # CPU卸载
            'fp16_enabled': True,  # 半精度
            'max_memory_per_gpu': f"{gpu_memory_gb - 1}GB",  # 保留1GB显存
            'activation_cpu_offload': True,  # 激活值CPU卸载
        }
        
        # 根据并行策略调整优化程度
        if self.current_strategy:
            strategy_config = self.get_strategy_config(self.current_strategy)
            tp_size = strategy_config.get('tensor_parallel_size', 1)
            
            if tp_size > 1:
                # 张量并行需要更多通信，减少CPU卸载
                optimization_config['cpu_offload'] = False
                optimization_config['activation_cpu_offload'] = False
            else:
                # 数据并行可以使用更激进的优化
                optimization_config['zero_stage'] = 3
        
        return optimization_config
    
    def apply_strategy(self, strategy_name: str, rank: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """应用并行策略"""
        self.current_strategy = strategy_name
        strategy_config = self.get_strategy_config(strategy_name)
        
        if not self.validate_strategy(strategy_config):
            raise ValueError(f"无效的并行策略: {strategy_name}")
        
        self.logger.info(f"应用并行策略: {strategy_name}")
        self.logger.info(f"策略配置: {strategy_config}")
        
        # 设置并行组
        parallel_groups = self.setup_parallel_groups(rank, strategy_config)
        
        # 计算设备放置
        device_placement = self.calculate_device_placement(strategy_config)
        
        # 获取显存优化配置
        memory_config = self.get_memory_optimization_config()
        
        return {
            'parallel_groups': parallel_groups,
            'device_placement': device_placement,
            'memory_config': memory_config,
            'strategy_config': strategy_config
        }, strategy_config
    
    def get_available_strategies(self) -> Dict[str, str]:
        """获取可用的并行策略列表"""
        strategies = {}
        
        # 预定义策略
        custom_strategies = self.config['parallel_strategy']['custom_strategies']
        for name, config in custom_strategies.items():
            tp = config.get('tensor_parallel_size', 1)
            pp = config.get('pipeline_parallel_size', 1)
            dp = config.get('data_parallel', True)
            
            description = f"TP={tp}, PP={pp}, DP={'是' if dp else '否'}"
            strategies[name] = description
        
        return strategies
    
    def recommend_strategy(self, model_size_gb: float, batch_size: int) -> str:
        """根据模型大小和批次大小推荐策略"""
        gpu_memory_gb = self.config.get('hardware', {}).get('gpu_memory_gb', 10)
        
        # 估算单GPU能否容纳模型
        model_memory_with_overhead = model_size_gb * 1.5  # 包含优化器状态等开销
        
        if model_memory_with_overhead <= gpu_memory_gb * 0.8:  # 保留20%显存余量
            # 小模型，使用数据并行
            return "pure_data_parallel"
        elif model_memory_with_overhead <= gpu_memory_gb * 1.5:
            # 中等模型，使用张量并行 + 数据并行
            return "tensor_data_hybrid"  
        else:
            # 大模型，使用完全模型并行
            return "full_model_parallel"
