#!/usr/bin/env python3
"""
分布式推理工作进程
被DeepSpeed启动器调用的工作脚本
"""

import os
import sys
import argparse
import logging
import torch
import deepspeed
import json
import time
from pathlib import Path
import traceback

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_manager import ModelManager
from src.inference.inference_engine import InferenceEngine
from src.utils.performance_monitor import PerformanceMonitor
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_logging():
    """设置日志"""
    rank = int(os.environ.get('LOCAL_RANK', 0))
    logger = logging.getLogger(f"DistributedWorker-Rank{rank}")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 文件日志
        file_handler = logging.FileHandler(log_dir / f"distributed_rank_{rank}.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 只有rank 0输出到控制台
        if rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    return logger

def load_test_data(data_path: str, max_samples: int = 100):
    """加载测试数据"""
    data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                data.append(json.loads(line.strip()))
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        # 返回默认测试数据
        return [
            {"prompt": "The future of artificial intelligence is", "max_length": 100},
            {"prompt": "In a world where technology advances rapidly", "max_length": 100},
            {"prompt": "Climate change poses significant challenges", "max_length": 100}
        ]

def run_distributed_inference(args):
    """运行分布式推理"""
    logger = setup_logging()
    
    # 获取分布式信息
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    logger.info(f"启动工作进程 - Local Rank: {local_rank}, World Rank: {world_rank}, World Size: {world_size}")
    
    # 设置CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"使用GPU: {device}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")
    
    try:
        # 1. 加载模型和tokenizer
        model_name = "gpt2-xl"  # 默认模型
        logger.info(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=None  # 让DeepSpeed管理设备
        )
        
        # 2. 初始化DeepSpeed
        if args.strategy == "tensor_parallel":
            # 张量并行配置
            ds_config = {
                "train_batch_size": args.batch_size * world_size,
                "train_micro_batch_size_per_gpu": args.batch_size,
                "fp16": {"enabled": torch.cuda.is_available()},
                "zero_optimization": {"stage": 0},  # 不使用ZeRO
                "tensor_parallel": {"tp_size": world_size}
            }
        elif args.strategy == "data_parallel":
            # 数据并行配置
            ds_config = {
                "train_batch_size": args.batch_size * world_size,
                "train_micro_batch_size_per_gpu": args.batch_size,
                "fp16": {"enabled": torch.cuda.is_available()},
                "zero_optimization": {"stage": 1}  # ZeRO stage 1
            }
        elif args.strategy == "pipeline_parallel":
            # 流水线并行配置
            ds_config = {
                "train_batch_size": args.batch_size * world_size,
                "train_micro_batch_size_per_gpu": args.batch_size,
                "fp16": {"enabled": torch.cuda.is_available()},
                "zero_optimization": {"stage": 0},
                "pipeline": {"stages": world_size}
            }
        else:  # hybrid
            # 混合并行配置
            ds_config = {
                "train_batch_size": args.batch_size * world_size,
                "train_micro_batch_size_per_gpu": args.batch_size,
                "fp16": {"enabled": torch.cuda.is_available()},
                "zero_optimization": {"stage": 2},
                "tensor_parallel": {"tp_size": min(2, world_size)}
            }
        
        logger.info(f"DeepSpeed配置: {ds_config}")
        
        # 初始化DeepSpeed引擎
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        
        logger.info("DeepSpeed初始化完成")
        
        # 3. 加载测试数据
        test_data = load_test_data(args.data_path, args.num_iterations)
        logger.info(f"加载了 {len(test_data)} 条测试数据")
        
        # 4. 性能监控器
        performance_monitor = PerformanceMonitor()
        
        # 5. 运行推理测试
        results = []
        total_tokens = 0
        total_time = 0
        
        logger.info("开始推理测试...")
        
        for i, sample in enumerate(test_data):
            if world_rank == 0:
                print(f"\\r进度: {i+1}/{len(test_data)}", end="", flush=True)
            
            try:
                # 准备输入
                prompt = sample.get("prompt", "The future of AI is")
                max_length = sample.get("max_length", 100)
                
                # 编码输入
                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # 开始计时
                start_time = time.time()
                performance_monitor.start_monitoring()
                
                # 生成
                with torch.no_grad():
                    # 使用DeepSpeed模型引擎
                    model_engine.eval()
                    
                    # 简单的贪心生成
                    generated_ids = input_ids.clone()
                    for _ in range(max_length - input_ids.shape[1]):
                        outputs = model_engine(generated_ids, attention_mask=attention_mask)
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                        
                        # 更新attention mask
                        attention_mask = torch.cat([
                            attention_mask, 
                            torch.ones((attention_mask.shape[0], 1), device=device)
                        ], dim=-1)
                        
                        # 检查是否生成了结束符
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                
                # 停止计时
                end_time = time.time()
                performance_monitor.stop_monitoring()
                
                # 计算指标
                inference_time = end_time - start_time
                generated_tokens = generated_ids.shape[1] - input_ids.shape[1]
                tokens_per_second = generated_tokens / inference_time if inference_time > 0 else 0
                
                # 记录结果
                result = {
                    "prompt": prompt,
                    "generated_tokens": generated_tokens,
                    "inference_time": inference_time,
                    "tokens_per_second": tokens_per_second,
                    "gpu_memory_allocated": torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0,
                    "gpu_memory_reserved": torch.cuda.memory_reserved(device) / 1024**2 if torch.cuda.is_available() else 0,
                }
                
                results.append(result)
                total_tokens += generated_tokens
                total_time += inference_time
                
                # 解码生成的文本（仅在rank 0时打印）
                if world_rank == 0 and i < 3:  # 只打印前3个示例
                    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    logger.info(f"\\n示例 {i+1}:")
                    logger.info(f"  提示: {prompt}")
                    logger.info(f"  生成: {generated_text[len(prompt):]}")
                    logger.info(f"  生成令牌数: {generated_tokens}")
                    logger.info(f"  吞吐量: {tokens_per_second:.2f} tokens/s")
                
            except Exception as e:
                logger.error(f"处理样本 {i} 时出错: {str(e)}")
                continue
        
        if world_rank == 0:
            print()  # 换行
        
        # 6. 计算和保存结果
        if results:
            avg_throughput = total_tokens / total_time if total_time > 0 else 0
            avg_latency = total_time / len(results)
            avg_memory = sum(r["gpu_memory_allocated"] for r in results) / len(results)
            
            summary = {
                "strategy": args.strategy,
                "world_size": world_size,
                "local_rank": local_rank,
                "world_rank": world_rank,
                "total_samples": len(results),
                "total_tokens": total_tokens,
                "total_time": total_time,
                "avg_throughput": avg_throughput,
                "avg_latency": avg_latency,
                "avg_memory_mb": avg_memory,
                "results": results
            }
            
            # 保存详细结果
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_file = results_dir / f"distributed_{args.strategy}_rank{world_rank}_{timestamp}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"结果保存到: {result_file}")
            
            # 打印性能摘要
            if world_rank == 0:
                print("\\n" + "="*80)
                print(f"分布式推理性能测试结果 - 策略: {args.strategy}")
                print("="*80)
                print(f"GPU数量: {world_size}")
                print(f"总样本数: {len(results)}")
                print(f"总令牌数: {total_tokens}")
                print(f"总时间: {total_time:.2f}s")
                print(f"平均吞吐量: {avg_throughput:.2f} tokens/s")
                print(f"平均延迟: {avg_latency:.3f}s")
                print(f"平均显存使用: {avg_memory:.0f} MB")
                print("="*80)
        
        else:
            logger.warning("没有成功处理任何样本")
        
    except Exception as e:
        logger.error(f"分布式推理失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser(description="分布式推理工作进程")
    
    # 配置文件参数
    parser.add_argument("--deepspeed_config", type=str, help="DeepSpeed配置文件")
    parser.add_argument("--model_config", type=str, help="模型配置文件")
    parser.add_argument("--inference_config", type=str, help="推理配置文件")
    parser.add_argument("--data_config", type=str, help="数据配置文件")
    
    # 推理参数
    parser.add_argument("--data_path", type=str, required=True, help="测试数据路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_iterations", type=int, default=10, help="测试迭代次数")
    parser.add_argument("--strategy", type=str, default="tensor_parallel", 
                        choices=["data_parallel", "tensor_parallel", "pipeline_parallel", "hybrid"],
                        help="并行策略")
    
    # DeepSpeed会自动添加的参数
    parser.add_argument("--local_rank", type=int, default=0, help="本地rank")
    
    args = parser.parse_args()
    
    # 运行分布式推理
    run_distributed_inference(args)

if __name__ == "__main__":
    main()
