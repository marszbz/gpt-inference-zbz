"""
完整示例运行脚本
演示如何使用GPT-1.5B分布式推理性能测试系统
"""

import sys
import os
import subprocess
import time
import logging
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/demo.log')
        ]
    )

def run_command(command, description):
    """运行命令并处理错误"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== {description} ===")
    logger.info(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        
        if result.stdout:
            logger.info(f"输出: {result.stdout}")
        
        logger.info(f"{description} 完成")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} 失败")
        logger.error(f"错误代码: {e.returncode}")
        logger.error(f"错误输出: {e.stderr}")
        return False

def check_requirements():
    """检查运行环境"""
    logger = logging.getLogger(__name__)
    
    logger.info("检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        logger.error("需要Python 3.8或更高版本")
        return False
    
    # 检查CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU模式（性能会显著降低）")
        else:
            logger.info(f"CUDA可用，GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        logger.error("PyTorch未安装")
        return False
    
    # 检查必要的目录
    directories = ['data', 'logs', 'results', 'config']
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.error(f"目录不存在: {dir_name}")
            return False
    
    logger.info("环境检查通过")
    return True

def install_dependencies():
    """安装依赖包"""
    logger = logging.getLogger(__name__)
    
    logger.info("安装Python依赖包...")
    
    # 检查requirements.txt是否存在
    if not Path("requirements.txt").exists():
        logger.error("requirements.txt文件不存在")
        return False
    
    # 安装依赖
    command = f"{sys.executable} -m pip install -r requirements.txt"
    return run_command(command, "安装依赖包")

def generate_dataset():
    """生成测试数据集"""
    logger = logging.getLogger(__name__)
    
    # 检查数据集是否已存在
    dataset_path = Path("data/datasets/benchmark_dataset.jsonl")
    if dataset_path.exists():
        logger.info("数据集已存在，跳过生成步骤")
        return True
    
    # 生成数据集
    command = f"{sys.executable} scripts/generate_dataset.py --log-level INFO"
    return run_command(command, "生成测试数据集")

def run_inference_test(mode="single"):
    """运行推理测试"""
    logger = logging.getLogger(__name__)
    
    if mode == "single":
        # 单GPU测试
        command = f"{sys.executable} scripts/run_inference_test.py --max-samples 100 --log-level INFO"
        return run_command(command, "运行单GPU推理测试")
    
    elif mode == "distributed":
        # 分布式测试
        gpu_count = 2  # 可以根据实际情况调整
        command = (f"{sys.executable} scripts/run_inference_test.py "
                  f"--distributed --world-size {gpu_count} "
                  f"--max-samples 100 --log-level INFO")
        return run_command(command, f"运行{gpu_count}GPU分布式推理测试")
    
    else:
        logger.error(f"未知的测试模式: {mode}")
        return False

def analyze_results():
    """分析测试结果"""
    logger = logging.getLogger(__name__)
    
    # 首先打印摘要
    command = f"{sys.executable} scripts/analyze_results.py summary"
    if not run_command(command, "打印结果摘要"):
        return False
    
    # 生成详细分析报告
    command = f"{sys.executable} scripts/analyze_results.py analyze --open-report"
    return run_command(command, "生成分析报告")

def cleanup():
    """清理临时文件"""
    logger = logging.getLogger(__name__)
    
    logger.info("清理临时文件...")
    
    # 清理PyTorch缓存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    logger.info("清理完成")

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== GPT-1.5B 分布式推理性能测试 - 完整示例 ===")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 1. 检查环境
        if not check_requirements():
            logger.error("环境检查失败，退出")
            return
        
        # 2. 安装依赖
        logger.info("跳过依赖安装步骤（假设已安装）")
        # if not install_dependencies():
        #     logger.error("依赖安装失败，退出")
        #     return
        
        # 3. 生成数据集
        if not generate_dataset():
            logger.error("数据集生成失败，退出")
            return
        
        # 4. 运行推理测试
        logger.info("开始推理性能测试...")
        
        # 首先运行单GPU测试
        if not run_inference_test("single"):
            logger.error("单GPU测试失败，但继续执行...")
        
        # 如果有多个GPU，运行分布式测试
        try:
            import torch
            if torch.cuda.device_count() > 1:
                logger.info("检测到多个GPU，运行分布式测试...")
                if not run_inference_test("distributed"):
                    logger.error("分布式测试失败，但继续执行...")
            else:
                logger.info("只有一个GPU，跳过分布式测试")
        except:
            logger.warning("无法检测GPU数量，跳过分布式测试")
        
        # 5. 分析结果
        if not analyze_results():
            logger.error("结果分析失败，但测试已完成")
        
        # 6. 清理
        cleanup()
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"=== 完整测试流程完成，总耗时: {total_time:.2f} 秒 ===")
        
        # 显示结果位置
        logger.info("测试结果位置:")
        logger.info(f"- 原始数据: data/datasets/")
        logger.info(f"- 推理结果: results/")
        logger.info(f"- 分析报告: results/analysis_*/")
        logger.info(f"- 日志文件: logs/")
        
    except KeyboardInterrupt:
        logger.info("用户中断了测试流程")
    except Exception as e:
        logger.error(f"测试流程中出现错误: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
