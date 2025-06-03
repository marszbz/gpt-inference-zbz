#!/usr/bin/env python3
"""
数据集验证脚本
验证生成的数据集文件的质量和结构
"""

import json
import os
from collections import defaultdict

def validate_dataset():
    """验证数据集文件"""
    dataset_dir = 'data/datasets'
    files = [f for f in os.listdir(dataset_dir) if f.startswith('benchmark_dataset_config_') and f.endswith('.jsonl')]
    print(f'数据集文件数量: {len(files)}')
    
    # 验证每个配置文件的样本数量和结构
    total_samples = 0
    configs = {}
    
    for file in sorted(files):
        config_id = int(file.split('_')[-1].split('.')[0])
        file_path = os.path.join(dataset_dir, file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f if line.strip()]
        
        total_samples += len(samples)
        
        # 统计每种source_type的数量
        source_types = defaultdict(int)
        prompt_lengths = set()
        generation_lengths = set()
        
        for sample in samples:
            source_type = sample.get('source_type', 'unknown')
            source_types[source_type] += 1
            prompt_lengths.add(sample.get('prompt_length'))
            generation_lengths.add(sample.get('generation_length'))
        
        configs[config_id] = {
            'sample_count': len(samples),
            'source_types': dict(source_types),
            'prompt_lengths': list(prompt_lengths),
            'generation_lengths': list(generation_lengths)
        }
        
        print(f'Config {config_id}: {len(samples)} 样本, 数据源: {dict(source_types)}')
    
    print(f'\n总样本数: {total_samples}')
    print(f'配置数量: {len(configs)}')
    
    # 验证配置组合
    expected_configs = []
    for prompt_len in [32, 128, 512, 1024]:
        for gen_len in [32, 64]:
            expected_configs.append((prompt_len, gen_len))
    
    print(f'\n期望的配置组合: {len(expected_configs)}')
    print(f'实际的配置组合: {len(configs)}')
    
    # 检查每个配置的详细信息
    print('\n配置详情:')
    for config_id in sorted(configs.keys()):
        config = configs[config_id]
        print(f'Config {config_id}: prompt_length={config["prompt_lengths"]}, generation_length={config["generation_lengths"]}')
    
    # 验证样本结构
    print('\n样本结构验证:')
    sample_file = os.path.join(dataset_dir, 'benchmark_dataset_config_0.jsonl')
    with open(sample_file, 'r', encoding='utf-8') as f:
        first_sample = json.loads(f.readline())
    
    required_fields = ['id', 'config_id', 'prompt', 'prompt_length', 'generation_length', 'source_type', 'metadata']
    missing_fields = [field for field in required_fields if field not in first_sample]
    
    if missing_fields:
        print(f'缺失字段: {missing_fields}')
    else:
        print('所有必需字段都存在')
    
    print(f'示例样本结构: {list(first_sample.keys())}')
    
    # 检查数据源分布
    print('\n数据源分布统计:')
    total_source_types = defaultdict(int)
    for config in configs.values():
        for source_type, count in config['source_types'].items():
            total_source_types[source_type] += count
    
    for source_type, count in total_source_types.items():
        percentage = (count / total_samples) * 100
        print(f'{source_type}: {count} 样本 ({percentage:.1f}%)')
    
    return configs

if __name__ == "__main__":
    validate_dataset()
