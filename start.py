"""
快速启动脚本
提供便捷的命令行界面来运行分布式推理
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_demo():
    """运行演示"""
    print("运行系统演示...")
    subprocess.run([sys.executable, "scripts/demo_simple.py"])

def run_single_gpu():
    """运行单GPU推理"""
    print("运行单GPU推理测试...")
    subprocess.run([
        sys.executable, "scripts/run_distributed_inference.py",
        "--world_size", "1",
        "--strategy", "pure_data_parallel",
        "--batch_size", "1",
        "--num_iterations", "10"
    ])

def run_multi_gpu(strategy="tensor_data_hybrid", world_size=4):
    """运行多GPU推理"""
    print(f"运行{world_size}GPU推理测试，策略: {strategy}")
    subprocess.run([
        sys.executable, "scripts/run_distributed_inference.py", 
        "--world_size", str(world_size),
        "--strategy", strategy,
        "--batch_size", "1",
        "--num_iterations", "50"
    ])

def run_benchmark():
    """运行完整基准测试"""
    print("运行完整策略基准测试...")
    subprocess.run([
        sys.executable, "scripts/strategy_benchmark.py",
        "--num_iterations", "30",
        "--batch_size", "1"
    ])

def generate_data():
    """生成测试数据"""
    print("生成测试数据集...")
    subprocess.run([sys.executable, "scripts/generate_dataset.py"])

def show_strategies():
    """显示可用策略"""
    print("可用的并行策略:")
    print("1. pure_data_parallel - 纯数据并行 (适合小模型)")
    print("2. tensor_data_hybrid - 张量+数据并行 (推荐)")
    print("3. pipeline_data_hybrid - 流水线+数据并行")
    print("4. full_model_parallel - 完全模型并行 (适合大模型)")

def check_environment():
    """检查环境"""
    import torch
    
    print("环境检查:")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name}, {memory_gb:.1f}GB")
    else:
        print("  警告: 没有可用的GPU")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPT分布式推理系统启动器")
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 演示命令
    subparsers.add_parser('demo', help='运行系统演示')
    
    # 数据生成命令
    subparsers.add_parser('generate-data', help='生成测试数据')
    
    # 单GPU推理命令
    subparsers.add_parser('single-gpu', help='运行单GPU推理')
    
    # 多GPU推理命令
    multi_gpu_parser = subparsers.add_parser('multi-gpu', help='运行多GPU推理')
    multi_gpu_parser.add_argument('--strategy', default='tensor_data_hybrid',
                                 choices=['pure_data_parallel', 'tensor_data_hybrid', 
                                         'pipeline_data_hybrid', 'full_model_parallel'],
                                 help='并行策略')
    multi_gpu_parser.add_argument('--gpus', type=int, default=4, help='GPU数量')
    
    # 基准测试命令
    subparsers.add_parser('benchmark', help='运行策略基准测试')
    
    # 显示策略命令
    subparsers.add_parser('strategies', help='显示可用策略')
    
    # 环境检查命令
    subparsers.add_parser('check-env', help='检查环境')
    
    args = parser.parse_args()
    
    # 改变到项目目录
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    if args.command == 'demo':
        run_demo()
    elif args.command == 'generate-data':
        generate_data()
    elif args.command == 'single-gpu':
        run_single_gpu()
    elif args.command == 'multi-gpu':
        run_multi_gpu(args.strategy, args.gpus)
    elif args.command == 'benchmark':
        run_benchmark()
    elif args.command == 'strategies':
        show_strategies()
    elif args.command == 'check-env':
        check_environment()
    else:
        # 默认显示帮助
        parser.print_help()
        print("\\n快速开始:")
        print("1. 检查环境: python start.py check-env")
        print("2. 运行演示: python start.py demo")
        print("3. 生成数据: python start.py generate-data")
        print("4. 单GPU测试: python start.py single-gpu")
        print("5. 多GPU测试: python start.py multi-gpu --strategy tensor_data_hybrid")
        print("6. 完整基准: python start.py benchmark")

if __name__ == "__main__":
    main()
