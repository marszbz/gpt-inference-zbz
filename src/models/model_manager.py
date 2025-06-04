"""
模型管理器
负责加载和管理GPT模型，支持分布式推理和自定义并行策略
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    AutoModelForCausalLM, 
    AutoTokenizer
)
import deepspeed
import logging
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import yaml
import json
import os
import time

from .parallel_strategy import ParallelStrategyManager, ParallelStrategy

class ModelManager:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """初始化模型管理器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_distributed = False
        self.deepspeed_engine = None
        self.parallel_strategy_manager = ParallelStrategyManager(config_path)
        self.parallel_groups = None
        self.current_strategy = None
        
        self.logger = self._setup_logger()
        
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
    
    def setup_distributed(self, rank: int, world_size: int, strategy_name: str = "tensor_data_hybrid") -> None:
        """设置分布式环境和并行策略"""
        self.logger.info(f"初始化分布式环境: rank={rank}, world_size={world_size}, strategy={strategy_name}")
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = self.config['distributed']['master_addr']
        os.environ['MASTER_PORT'] = self.config['distributed']['master_port']
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
          # 初始化分布式进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config['distributed']['backend'],
                init_method=self.config['distributed'].get('init_method', 'env://'),                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(minutes=self.config['distributed'].get('timeout_minutes', 30))
            )
        
        self.is_distributed = True
        
        # 在分布式设置中处理设备映射
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            actual_device_id = rank % available_gpus
            self.device = torch.device(f"cuda:{actual_device_id}")
            torch.cuda.set_device(actual_device_id)
        else:
            self.device = torch.device("cpu")
        
        # 应用并行策略
        strategy_result, strategy_config = self.parallel_strategy_manager.apply_strategy(strategy_name, rank)
        self.parallel_groups = strategy_result['parallel_groups']
        self.device_placement = strategy_result['device_placement'] 
        self.memory_config = strategy_result['memory_config']
        self.current_strategy = strategy_name
        
        self.logger.info(f"分布式环境设置完成，设备: {self.device}")
        self.logger.info(f"当前rank在并行组中的角色: {self.device_placement['device_map'][rank]}")
    
    def get_model_size_estimate(self, model_name: str) -> float:
        """估算模型大小（GB）"""
        model_sizes = {
            'gpt2': 0.5,
            'gpt2-medium': 1.3, 
            'gpt2-large': 3.0,
            'gpt2-xl': 6.0,  # GPT2-XL约1.5B参数
        }
        
        # 默认估算：假设每个参数4字节（float32），加上优化器状态
        if model_name in model_sizes:
            return model_sizes[model_name]
        else:
            # 保守估算
            return 6.0
    
    def optimize_for_rtx3080(self) -> Dict[str, Any]:
        """为RTX 3080优化模型配置"""
        optimization_config = {
            'torch_dtype': torch.float16,  # 使用FP16
            'low_cpu_mem_usage': True,     # 低CPU内存使用
            'device_map': None,            # 由分布式策略控制
        }
        
        # 根据并行策略调整
        if self.current_strategy:
            strategy_config = self.parallel_strategy_manager.get_strategy_config(self.current_strategy)
            tp_size = strategy_config.get('tensor_parallel_size', 1)
            
            if tp_size > 1:
                # 张量并行需要确保权重在多GPU间正确分布
                optimization_config['load_in_8bit'] = False  # 避免量化影响张量并行
            else:
                # 数据并行可以使用更激进的优化
                optimization_config['load_in_8bit'] = False  # RTX 3080显存足够，不使用8bit
        
        return optimization_config
    
    def load_model(self, local_rank: Optional[int] = None) -> None:
        """加载模型和分词器"""
        self.logger.info(f"加载模型: {self.config['model']['name']}")
          # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['tokenizer_path'],
            cache_dir="./cache",
            local_files_only=False,
            padding_side='left'  # 修复decoder-only模型的padding警告
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
          # 确定设备
        if local_rank is not None:
            # 在分布式设置中处理设备映射
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                actual_device_id = local_rank % available_gpus
                self.device = torch.device(f"cuda:{actual_device_id}")
            else:
                self.device = torch.device("cpu")
        elif self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取优化配置
        optimization_config = self.optimize_for_rtx3080()
        
        # 加载模型
        self.logger.info("开始加载模型权重...")
        start_time = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['model_path'],
            cache_dir="./cache",
            local_files_only=False,
            **optimization_config
        )
        
        load_time = time.time() - start_time
        self.logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # 应用分布式策略
        if self.is_distributed:
            self._apply_distributed_strategy()
        else:
            self.model = self.model.to(self.device)
        
        # 打印模型信息
        self._print_model_info()
        
    def _apply_distributed_strategy(self) -> None:
        """应用分布式策略"""
        if not self.is_distributed:
            return
        
        deepspeed_enabled = self.config.get('deepspeed', {}).get('enabled', False)
        
        if deepspeed_enabled:
            self._setup_deepspeed()
        else:
            # 使用PyTorch原生分布式
            self._setup_pytorch_distributed()
    
    def _setup_deepspeed(self) -> None:
        """设置DeepSpeed"""
        self.logger.info("使用DeepSpeed初始化模型...")
        
        # 加载DeepSpeed配置
        deepspeed_config_path = self.config['deepspeed']['config_path']
        with open(deepspeed_config_path, 'r') as f:
            deepspeed_config = json.load(f)
        
        # 根据并行策略调整DeepSpeed配置
        if self.current_strategy:
            strategy_config = self.parallel_strategy_manager.get_strategy_config(self.current_strategy)
            tp_size = strategy_config.get('tensor_parallel_size', 1)
            
            if tp_size > 1:
                # 张量并行时调整ZeRO配置
                deepspeed_config['zero_optimization']['stage'] = 1  # 降低ZeRO stage
        
        # 初始化DeepSpeed引擎
        self.deepspeed_engine, _, _, _ = deepspeed.initialize(
            model=self.model,
            config=deepspeed_config,
            dist_init_required=False  # 我们已经初始化了分布式
        )
        
        self.model = self.deepspeed_engine.module
        self.logger.info("DeepSpeed初始化完成")
    
    def _setup_pytorch_distributed(self) -> None:
        """设置PyTorch原生分布式"""
        self.logger.info("使用PyTorch DDP初始化模型...")
        
        self.model = self.model.to(self.device)
        
        # 根据并行策略选择分布式包装方式
        if self.parallel_groups and self.parallel_groups['data_parallel_group']:
            # 使用数据并行组
            self.model = DDP(
                self.model, 
                device_ids=[self.device.index],
                process_group=self.parallel_groups['data_parallel_group'],
                find_unused_parameters=True  # 对于复杂的模型并行场景
            )
        else:
            # 标准DDP
            self.model = DDP(
                self.model,
                device_ids=[self.device.index]
            )
        
        self.logger.info("PyTorch DDP初始化完成")
    
    def _print_model_info(self) -> None:
        """打印模型信息"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            
            self.logger.info(f"GPU {self.device.index} 显存使用:")
            self.logger.info(f"  - 已分配: {memory_allocated:.2f} GB")
            self.logger.info(f"  - 已保留: {memory_reserved:.2f} GB")
        
        # 统计参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型参数统计:")
        self.logger.info(f"  - 总参数: {total_params:,}")
        self.logger.info(f"  - 可训练参数: {trainable_params:,}")
        self.logger.info(f"  - 参数大小估算: {total_params * 4 / 1024**3:.2f} GB (FP32)")
        
        self.logger.info("模型加载完成")
    
    def setup_deepspeed(self, args=None) -> None:
        """设置DeepSpeed"""
        if not self.config.get('deepspeed', {}).get('enabled', False):
            return
        
        self.logger.info("设置DeepSpeed引擎")
        
        # 加载DeepSpeed配置
        ds_config_path = self.config['deepspeed']['config_path']
        with open(ds_config_path, 'r') as f:
            ds_config = json.load(f)
        
        # 初始化DeepSpeed引擎
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=self.model,
            config=ds_config
        )
        
        self.deepspeed_engine = model_engine
        self.model = model_engine.module  # 获取原始模型
        
        self.logger.info("DeepSpeed设置完成")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        model = self.model
        if hasattr(self.model, 'module'):  # DDP包装的模型
            model = self.model.module
        
        num_parameters = sum(p.numel() for p in model.parameters())
        num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            "model_name": self.config['model']['name'],
            "num_parameters": num_parameters,
            "num_trainable_parameters": num_trainable_parameters,
            "model_size_mb": num_parameters * 4 / (1024 * 1024),  # 假设float32
            "device": str(self.device),
            "is_distributed": self.is_distributed,
            "deepspeed_enabled": self.deepspeed_engine is not None
        }
        
        return info
    
    def prepare_inputs(self, texts: list, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """准备输入数据"""
        if max_length is None:
            max_length = self.config['model']['max_length']
        
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # 移动到设备
        for key in encoding:
            encoding[key] = encoding[key].to(self.device)
        
        return encoding
    
    @torch.no_grad()
    def generate(self, 
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 max_new_tokens: int = 50,
                 **generation_kwargs) -> torch.Tensor:
        """生成文本"""
        model = self.deepspeed_engine if self.deepspeed_engine else self.model
        
        # 设置生成参数
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 1.0,
            **generation_kwargs
        }
          # 生成 - 处理DataParallel包装的模型
        if hasattr(model, 'module'):
            # 如果模型被DataParallel包装，使用module属性访问原始模型
            outputs = model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        else:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        
        return outputs
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("资源清理完成")
    
    def initialize_model_with_strategy(self,
                                       strategy: str,
                                       local_rank: int,
                                       world_size: int,
                                       **strategy_config) -> tuple:
        """根据并行策略初始化模型
        
        Args:
            strategy: 并行策略名称
            local_rank: 本地进程rank
            world_size: 总进程数
            **strategy_config: 策略配置参数
            
        Returns:
            tuple: (model, tokenizer)
        """
        self.logger.info(f"使用策略 {strategy} 初始化模型 (rank {local_rank}/{world_size})")
        
        self.current_strategy = strategy
          # 设置设备
        if torch.cuda.is_available():
            # 在分布式设置中，CUDA_VISIBLE_DEVICES可能限制了可见GPU数量
            # 因此需要映射local_rank到实际可用的GPU索引
            available_gpus = torch.cuda.device_count()
            actual_device_id = local_rank % available_gpus
            torch.cuda.set_device(actual_device_id)
            self.device = torch.device(f"cuda:{actual_device_id}")
            self.logger.info(f"设置设备: {self.device}, 可用GPU数量: {available_gpus}, Local Rank: {local_rank}")
        else:
            self.device = torch.device("cpu")
        
        # 加载基础模型和分词器
        self.load_model(local_rank)
        
        # 根据策略设置并行
        if strategy == 'pure_data_parallel':
            self._setup_data_parallel(local_rank, world_size)
            
        elif strategy == 'tensor_data_hybrid':
            self._setup_tensor_data_hybrid(local_rank, world_size, **strategy_config)
            
        elif strategy == 'pipeline_data_hybrid':
            self._setup_pipeline_data_hybrid(local_rank, world_size, **strategy_config)
            
        elif strategy == 'full_model_parallel':
            self._setup_full_model_parallel(local_rank, world_size, **strategy_config)
            
        else:
            raise ValueError(f"未知的并行策略: {strategy}")
        
        self._print_model_info()
        
        return self.model, self.tokenizer
    
    def _setup_data_parallel(self, local_rank: int, world_size: int):
        """设置纯数据并行"""
        self.logger.info("设置数据并行策略")
        
        # 初始化分布式
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                rank=local_rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=300)
            )
        
        # 将模型移动到GPU
        self.model = self.model.to(self.device)
          # 包装为DDP
        actual_device_id = local_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.model = DDP(
            self.model,
            device_ids=[actual_device_id] if torch.cuda.is_available() else None,
            output_device=actual_device_id if torch.cuda.is_available() else None,
            find_unused_parameters=False
        )
        
        self.is_distributed = True
    
    def _setup_tensor_data_hybrid(self, local_rank: int, world_size: int, **config):
        """设置张量并行+数据并行混合策略"""
        self.logger.info("设置张量+数据并行混合策略")
        
        tp_size = config.get('tensor_parallel_size', 2)
        dp_size = world_size // tp_size
        
        # 初始化分布式
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                rank=local_rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=300)
            )
        
        # 创建并行组
        self._create_parallel_groups(local_rank, world_size, tp_size, dp_size)
        
        # 将模型移动到GPU
        self.model = self.model.to(self.device)
          # 简化的张量并行实现（实际应该使用FairScale或Megatron）
        if tp_size > 1:
            self.logger.warning("张量并行需要专门的库支持，当前使用简化实现")
            # 这里应该集成FairScale或Megatron的张量并行
        
        # 数据并行包装
        if dp_size > 1:
            actual_device_id = local_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
            self.model = DDP(
                self.model,
                device_ids=[actual_device_id] if torch.cuda.is_available() else None,
                output_device=actual_device_id if torch.cuda.is_available() else None,
                process_group=self.parallel_groups.get('data_parallel_group')
            )
        
        self.is_distributed = True
    
    def _setup_pipeline_data_hybrid(self, local_rank: int, world_size: int, **config):
        """设置流水线并行+数据并行混合策略"""
        self.logger.info("设置流水线+数据并行混合策略")
        
        pp_size = config.get('pipeline_parallel_size', 2)
        dp_size = world_size // pp_size
        
        # 初始化分布式
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                rank=local_rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=300)
            )
        
        # 创建并行组
        self._create_parallel_groups(local_rank, world_size, 1, dp_size, pp_size)
          # 将模型移动到GPU
        self.model = self.model.to(self.device)
        
        # 简化的流水线并行实现
        if pp_size > 1:
            self.logger.warning("流水线并行需要专门的库支持，当前使用简化实现")
            # 这里应该集成FairScale或GPipe的流水线并行
        
        # 数据并行包装
        if dp_size > 1:
            actual_device_id = local_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
            self.model = DDP(
                self.model,
                device_ids=[actual_device_id] if torch.cuda.is_available() else None,
                output_device=actual_device_id if torch.cuda.is_available() else None,
                process_group=self.parallel_groups.get('data_parallel_group')
            )
        
        self.is_distributed = True
      def _setup_full_model_parallel(self, local_rank: int, world_size: int, **config):
        """设置全模型并行策略"""
        self.logger.info("设置全模型并行策略")
        
        # 根据world_size动态调整并行大小
        tp_size = min(config.get('tensor_parallel_size', 2), world_size)
        pp_size = min(config.get('pipeline_parallel_size', 1), world_size // tp_size)
        
        self.logger.info(f"全模型并行配置: tp_size={tp_size}, pp_size={pp_size}, world_size={world_size}")
        
        # 验证配置
        if tp_size * pp_size > world_size:
            self.logger.warning(f"并行配置超出world_size，调整为: tp_size={world_size}, pp_size=1")
            tp_size = world_size
            pp_size = 1
        
        # 初始化分布式
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                rank=local_rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=300)
            )
        
        # 创建并行组
        self._create_parallel_groups(local_rank, world_size, tp_size, 1, pp_size)
        
        # 将模型移动到GPU
        self.model = self.model.to(self.device)
        
        # 简化的全模型并行实现
        self.logger.warning("全模型并行需要专门的库支持，当前使用简化实现")
        
        self.is_distributed = True
      def _create_parallel_groups(self, rank: int, world_size: int, 
                               tp_size: int = 1, dp_size: int = 1, pp_size: int = 1):
        """创建并行进程组"""
        self.parallel_groups = {}
        
        self.logger.info(f"创建并行组: rank={rank}, world_size={world_size}, tp_size={tp_size}, dp_size={dp_size}, pp_size={pp_size}")
        
        # 验证配置
        total_ranks_needed = tp_size * dp_size * pp_size
        if total_ranks_needed > world_size:
            self.logger.warning(f"所需ranks ({total_ranks_needed}) 超出world_size ({world_size})，使用简化配置")
            # 简化为只使用张量并行
            tp_size = min(tp_size, world_size)
            dp_size = 1
            pp_size = 1
        
        # 数据并行组
        if dp_size > 1:
            for i in range(tp_size * pp_size):
                dp_ranks = [i + j * tp_size * pp_size for j in range(dp_size)]
                # 确保所有ranks都在有效范围内
                dp_ranks = [r for r in dp_ranks if r < world_size]
                if len(dp_ranks) > 1:
                    dp_group = dist.new_group(dp_ranks)
                    if rank in dp_ranks:
                        self.parallel_groups['data_parallel_group'] = dp_group
        
        # 张量并行组
        if tp_size > 1:
            for pp_rank in range(pp_size):
                for dp_rank in range(dp_size):
                    tp_ranks = [pp_rank * tp_size * dp_size + dp_rank * tp_size + tp_rank 
                               for tp_rank in range(tp_size)]
                    # 确保所有ranks都在有效范围内
                    tp_ranks = [r for r in tp_ranks if r < world_size]
                    if len(tp_ranks) > 1:
                        tp_group = dist.new_group(tp_ranks)
                        if rank in tp_ranks:
                            self.parallel_groups['tensor_parallel_group'] = tp_group
        
        # 流水线并行组
        if pp_size > 1:
            for tp_rank in range(tp_size):
                for dp_rank in range(dp_size):
                    pp_ranks = [pp_rank * tp_size * dp_size + dp_rank * tp_size + tp_rank 
                               for pp_rank in range(pp_size)]
                    # 确保所有ranks都在有效范围内
                    pp_ranks = [r for r in pp_ranks if r < world_size]
                    if len(pp_ranks) > 1:
                        pp_group = dist.new_group(pp_ranks)
                        if rank in pp_ranks:
                            self.parallel_groups['pipeline_parallel_group'] = pp_group

class ModelParallelManager(ModelManager):
    """支持模型并行的模型管理器"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        super().__init__(config_path)
        self.tensor_parallel_size = self.config.get('model_parallel', {}).get('tensor_parallel_size', 1)
        self.pipeline_parallel_size = self.config.get('model_parallel', {}).get('pipeline_parallel_size', 1)
    
    def setup_model_parallel(self, rank: int, world_size: int) -> None:
        """设置模型并行"""
        self.logger.info("设置模型并行")
        
        # 这里可以集成Megatron-LM或FairScale等模型并行库
        # 由于复杂性，这里提供一个简化的实现框架
        
        if self.tensor_parallel_size > 1:
            self.logger.info(f"启用张量并行，大小: {self.tensor_parallel_size}")
            # 实现张量并行逻辑
        
        if self.pipeline_parallel_size > 1:
            self.logger.info(f"启用流水线并行，大小: {self.pipeline_parallel_size}")
            # 实现流水线并行逻辑
