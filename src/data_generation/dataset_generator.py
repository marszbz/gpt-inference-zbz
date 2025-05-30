"""
数据集生成器
用于生成基于WikiText-103、Pile子集和Synthetic随机token的基准测试数据集
"""

import json
import random
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer
import yaml
from tqdm import tqdm
import jsonlines

class DatasetGenerator:
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """初始化数据集生成器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 设置随机种子
        random.seed(self.config['processing']['seed'])
        np.random.seed(self.config['processing']['seed'])
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def load_wikitext103(self) -> List[str]:
        """加载WikiText-103数据"""
        if not self.config['dataset']['sources']['wikitext103']['enabled']:
            return []
        
        self.logger.info("加载WikiText-103数据...")
        dataset = load_dataset(
            self.config['dataset']['sources']['wikitext103']['dataset_name'],
            self.config['dataset']['sources']['wikitext103']['dataset_config'],
            split=self.config['dataset']['sources']['wikitext103']['split']
        )
        
        # 过滤空文本和太短的文本
        texts = [item['text'] for item in dataset if item['text'].strip() and len(item['text']) > 100]
        
        # 随机采样
        num_samples = self.config['dataset']['sources']['wikitext103']['num_samples']
        if len(texts) > num_samples:
            texts = random.sample(texts, num_samples)
        
        self.logger.info(f"加载了 {len(texts)} 条WikiText-103样本")
        return texts
    
    def load_pile_subset(self) -> List[str]:
        """加载Pile子集数据"""
        if not self.config['dataset']['sources']['pile']['enabled']:
            return []
        
        self.logger.info("加载Pile子集数据...")
        try:
            # 注意：实际使用时可能需要调整数据集名称和配置
            # 这里使用一个示例配置
            dataset = load_dataset(
                "EleutherAI/pile",
                split="train",
                streaming=True  # 流式加载大数据集
            )
            
            texts = []
            num_samples = self.config['dataset']['sources']['pile']['num_samples']
            subset_name = self.config['dataset']['sources']['pile']['subset']
            
            for i, item in enumerate(dataset):
                if len(texts) >= num_samples:
                    break
                    
                # 根据subset筛选
                if 'meta' in item and 'pile_set_name' in item['meta']:
                    if item['meta']['pile_set_name'] == subset_name:
                        if item['text'].strip() and len(item['text']) > 100:
                            texts.append(item['text'])
                
                if i % 10000 == 0:
                    self.logger.info(f"已处理 {i} 条Pile数据，收集到 {len(texts)} 条有效样本")
            
        except Exception as e:
            self.logger.warning(f"加载Pile数据时出错: {e}，使用替代数据源")
            # 如果Pile数据不可用，使用其他开源数据集作为替代
            texts = self._load_alternative_dataset()
        
        self.logger.info(f"加载了 {len(texts)} 条Pile子集样本")
        return texts
    
    def _load_alternative_dataset(self) -> List[str]:
        """加载替代数据集（当Pile不可用时）"""
        try:
            # 使用OpenWebText作为替代
            dataset = load_dataset("openwebtext", split="train", streaming=True)
            texts = []
            num_samples = self.config['dataset']['sources']['pile']['num_samples']
            
            for i, item in enumerate(dataset):
                if len(texts) >= num_samples:
                    break
                    
                if item['text'].strip() and len(item['text']) > 100:
                    texts.append(item['text'])
            
            return texts
        except:
            # 最后的备选方案：生成一些示例文本
            return [f"This is sample text number {i} for testing purposes. " * 10 
                   for i in range(self.config['dataset']['sources']['pile']['num_samples'])]
    
    def generate_synthetic_data(self) -> List[str]:
        """生成合成随机token数据"""
        if not self.config['dataset']['sources']['synthetic']['enabled']:
            return []
        
        self.logger.info("生成合成随机token数据...")
        texts = []
        vocab_size = self.config['dataset']['sources']['synthetic']['vocab_size']
        num_samples = self.config['dataset']['sources']['synthetic']['num_samples']
        
        for _ in range(num_samples):
            # 生成随机长度的token序列
            length = random.randint(200, 800)
            tokens = [random.randint(0, vocab_size - 1) for _ in range(length)]
            
            # 将tokens转换为文本
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(text)
        
        self.logger.info(f"生成了 {len(texts)} 条合成数据样本")
        return texts
    
    def create_test_samples(self, source_texts: List[str]) -> List[Dict[str, Any]]:
        """创建测试样本"""
        self.logger.info("创建测试样本...")
        samples = []
        
        prompt_lengths = self.config['test_configs']['prompt_lengths']
        generation_lengths = self.config['test_configs']['generation_lengths']
        samples_per_config = self.config['test_configs']['samples_per_config']
        
        config_id = 0
        for prompt_len in prompt_lengths:
            for gen_len in generation_lengths:
                self.logger.info(f"创建配置 {config_id}: prompt_len={prompt_len}, gen_len={gen_len}")
                
                for i in range(samples_per_config):
                    # 随机选择源文本
                    source_text = random.choice(source_texts)
                    
                    # 创建prompt
                    prompt = self._create_prompt(source_text, prompt_len)
                    
                    sample = {
                        "id": f"sample_{config_id}_{i}",
                        "config_id": config_id,
                        "prompt": prompt,
                        "prompt_length": prompt_len,
                        "generation_length": gen_len,
                        "source_type": self._get_source_type(source_text),
                        "metadata": {
                            "created_at": "2025-05-30",
                            "tokenizer": "gpt2",
                            "prompt_tokens": len(self.tokenizer.encode(prompt))
                        }
                    }
                    
                    samples.append(sample)
                
                config_id += 1
        
        self.logger.info(f"总共创建了 {len(samples)} 个测试样本")
        return samples
    
    def _create_prompt(self, source_text: str, target_length: int) -> str:
        """从源文本创建指定长度的prompt"""
        tokens = self.tokenizer.encode(source_text)
        
        if len(tokens) >= target_length:
            # 如果文本足够长，随机选择一个起始位置
            start_idx = random.randint(0, len(tokens) - target_length)
            selected_tokens = tokens[start_idx:start_idx + target_length]
        else:
            # 如果文本太短，重复文本直到达到目标长度
            repeated_tokens = tokens * ((target_length // len(tokens)) + 1)
            selected_tokens = repeated_tokens[:target_length]
        
        return self.tokenizer.decode(selected_tokens, skip_special_tokens=True)
    
    def _get_source_type(self, text: str) -> str:
        """简单地根据文本特征判断来源类型"""
        # 这是一个简化的实现，实际可以根据更复杂的启发式规则
        if len(text) > 500 and any(word in text.lower() for word in ['wikipedia', 'article', 'section']):
            return "wikitext103"
        elif any(char in text for char in ['@', 'http', 'def ', 'import ']):
            return "pile"
        else:
            return "synthetic"
    
    def save_dataset(self, samples: List[Dict[str, Any]]) -> None:
        """保存数据集"""
        output_path = Path(self.config['output']['dataset_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config['output']['split_by_config']:
            # 按配置分割保存
            self._save_split_dataset(samples, output_path)
        else:
            # 保存为单个文件
            self._save_single_dataset(samples, output_path)
    
    def _save_single_dataset(self, samples: List[Dict[str, Any]], output_path: Path) -> None:
        """保存为单个JSONL文件"""
        self.logger.info(f"保存数据集到 {output_path}")
        
        with jsonlines.open(output_path, mode='w') as writer:
            for sample in samples:
                writer.write(sample)
        
        self.logger.info(f"数据集已保存，共 {len(samples)} 个样本")
    
    def _save_split_dataset(self, samples: List[Dict[str, Any]], base_path: Path) -> None:
        """按配置分割保存数据集"""
        # 按config_id分组
        config_groups = {}
        for sample in samples:
            config_id = sample['config_id']
            if config_id not in config_groups:
                config_groups[config_id] = []
            config_groups[config_id].append(sample)
        
        # 保存每个配置的数据
        for config_id, config_samples in config_groups.items():
            config_path = base_path.parent / f"{base_path.stem}_config_{config_id}.jsonl"
            
            with jsonlines.open(config_path, mode='w') as writer:
                for sample in config_samples:
                    writer.write(sample)
            
            self.logger.info(f"配置 {config_id} 已保存到 {config_path}，共 {len(config_samples)} 个样本")
    
    def generate_full_dataset(self) -> None:
        """生成完整数据集"""
        self.logger.info("开始生成完整数据集...")
        
        # 加载所有数据源
        all_texts = []
        
        wikitext_data = self.load_wikitext103()
        all_texts.extend(wikitext_data)
        
        pile_data = self.load_pile_subset()
        all_texts.extend(pile_data)
        
        synthetic_data = self.generate_synthetic_data()
        all_texts.extend(synthetic_data)
        
        self.logger.info(f"总共加载了 {len(all_texts)} 条源文本")
        
        # 创建测试样本
        samples = self.create_test_samples(all_texts)
        
        # 保存数据集
        self.save_dataset(samples)
        
        self.logger.info("数据集生成完成！")


if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate_full_dataset()
