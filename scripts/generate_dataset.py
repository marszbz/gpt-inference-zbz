"""
数据集生成脚本
用于生成GPT-1.5B推理性能测试数据集
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation import DatasetGenerator

def setup_logging(log_level: str = "INFO") -> None:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/dataset_generation.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='生成GPT推理性能测试数据集')
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--force', action='store_true',
                       help='强制重新生成数据集（覆盖现有文件）')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 检查输出文件是否存在
    output_path = Path("data/datasets/benchmark_dataset.jsonl")
    if output_path.exists() and not args.force:
        logger.warning(f"数据集文件已存在: {output_path}")
        response = input("是否要覆盖现有文件? (y/N): ")
        if response.lower() != 'y':
            logger.info("取消数据集生成")
            return
    
    try:
        # 创建数据集生成器
        logger.info("初始化数据集生成器...")
        generator = DatasetGenerator(args.config)
        
        # 生成数据集
        logger.info("开始生成数据集...")
        generator.generate_full_dataset()
        
        logger.info("数据集生成完成！")
        
    except Exception as e:
        logger.error(f"数据集生成失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
