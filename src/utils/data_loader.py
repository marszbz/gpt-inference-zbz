"""
数据加载工具
"""

import json
import logging
from typing import List, Dict, Any, Iterator, Optional
from pathlib import Path
import jsonlines
from dataclasses import dataclass

@dataclass
class TestSample:
    """测试样本数据类"""
    id: str
    config_id: int
    prompt: str
    prompt_length: int
    generation_length: int
    source_type: str
    metadata: Dict[str, Any]

class DataLoader:
    """数据加载器"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    def load_samples(self, config_ids: Optional[List[int]] = None) -> List[TestSample]:
        """加载测试样本"""
        samples = []
        
        if self.dataset_path.is_file():
            # 单个文件
            samples.extend(self._load_from_file(self.dataset_path, config_ids))
        else:
            # 目录中的多个文件
            for file_path in self.dataset_path.glob("*.jsonl"):
                samples.extend(self._load_from_file(file_path, config_ids))
        
        self.logger.info(f"加载了 {len(samples)} 个测试样本")
        return samples
    
    def _load_from_file(self, file_path: Path, config_ids: Optional[List[int]]) -> List[TestSample]:
        """从单个文件加载样本"""
        samples = []
        
        try:
            with jsonlines.open(file_path, mode='r') as reader:
                for item in reader:
                    # 过滤配置
                    if config_ids is not None and item.get('config_id') not in config_ids:
                        continue
                    
                    sample = TestSample(
                        id=item['id'],
                        config_id=item['config_id'],
                        prompt=item['prompt'],
                        prompt_length=item['prompt_length'],
                        generation_length=item['generation_length'],
                        source_type=item['source_type'],
                        metadata=item.get('metadata', {})
                    )
                    samples.append(sample)
        
        except Exception as e:
            self.logger.error(f"加载文件 {file_path} 时出错: {e}")
        
        return samples
    
    def get_config_info(self) -> Dict[int, Dict[str, Any]]:
        """获取配置信息"""
        config_info = {}
        samples = self.load_samples()
        
        for sample in samples:
            if sample.config_id not in config_info:
                config_info[sample.config_id] = {
                    'prompt_length': sample.prompt_length,
                    'generation_length': sample.generation_length,
                    'sample_count': 0,
                    'source_types': set()
                }
            
            config_info[sample.config_id]['sample_count'] += 1
            config_info[sample.config_id]['source_types'].add(sample.source_type)
        
        # 转换set为list
        for config_id in config_info:
            config_info[config_id]['source_types'] = list(config_info[config_id]['source_types'])
        
        return config_info


class BatchIterator:
    """批次迭代器"""
    
    def __init__(self, samples: List[TestSample], batch_size: int = 1):
        self.samples = samples
        self.batch_size = batch_size
        self.current_index = 0
    
    def __iter__(self) -> Iterator[List[TestSample]]:
        """迭代器接口"""
        self.current_index = 0
        return self
    
    def __next__(self) -> List[TestSample]:
        """获取下一个批次"""
        if self.current_index >= len(self.samples):
            raise StopIteration
        
        end_index = min(self.current_index + self.batch_size, len(self.samples))
        batch = self.samples[self.current_index:end_index]
        self.current_index = end_index
        
        return batch
    
    def __len__(self) -> int:
        """批次数量"""
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
