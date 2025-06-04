#!/usr/bin/env python3
"""
分布式推理启动器
使用DeepSpeed launcher正确设置环境变量和启动分布式推理
"""

import os
import sys
import subprocess
import argparse
import tempfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="启动分布式GPT推理测试")
    
    # 基本配置
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml",
                        help="模型配置文件路径")
    parser.add_argument("--inference_config", type=str, default="config/inference_config.yaml", 
                        help="推理配置文件路径")
    parser.add_argument("--data_config", type=str, default="config/data_config.yaml",
                        help="数据配置文件路径")
    
    # 推理参数
    parser.add_argument("--data_path", type=str, default="data/datasets/benchmark_dataset_config_1.jsonl",
                        help="测试数据路径")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="测试迭代次数")
    
    # 分布式参数
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="GPU数量")
    parser.add_argument("--strategy", type=str, default="tensor_parallel",
                        choices=["data_parallel", "tensor_parallel", "pipeline_parallel", "hybrid"],
                        help="并行策略")
    
    # DeepSpeed参数
    parser.add_argument("--deepspeed_config", type=str, default="config/deepspeed_config.json",
                        help="DeepSpeed配置文件路径")
    
    args = parser.parse_args()
    
    # 确保脚本路径正确
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    inference_script = script_dir / "distributed_inference_worker.py"
    
    # 构建DeepSpeed启动命令
    cmd = [
        "deepspeed",
        "--num_gpus", str(args.num_gpus),
        "--num_nodes", "1",
        str(inference_script),
        "--deepspeed_config", args.deepspeed_config,
        "--model_config", args.model_config,
        "--inference_config", args.inference_config,
        "--data_config", args.data_config,
        "--data_path", args.data_path,
        "--batch_size", str(args.batch_size),
        "--num_iterations", str(args.num_iterations),
        "--strategy", args.strategy
    ]
    
    print("启动分布式推理...")
    print(f"命令: {' '.join(cmd)}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["PYTHONPATH"] = str(project_root)
    
    # 运行命令
    try:
        result = subprocess.run(cmd, env=env, cwd=str(project_root), check=True)
        print("分布式推理完成！")
    except subprocess.CalledProcessError as e:
        print(f"分布式推理失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("错误: 找不到deepspeed命令。请确保已安装DeepSpeed:")
        print("pip install deepspeed")
        sys.exit(1)

if __name__ == "__main__":
    main()
