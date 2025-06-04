#!/usr/bin/env python3
"""
修复版分布式推理工作进程
正确处理环境变量和DeepSpeed初始化
"""

import os
import sys
import argparse
import logging
import torch
import json
import time
from pathlib import Path
import traceback
import datetime
from typing import Dict, List, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_manager import ModelManager
from src.models.parallel_strategy import ParallelStrategyManager
from src.utils.performance_monitor import PerformanceMonitor

def setup_logging():
    """设置日志"""
    rank = int(os.environ.get('LOCAL_RANK', 0))
    logger = logging.getLogger(f"FixedWorker-Rank{rank}")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 文件日志
        file_handler = logging.FileHandler(log_dir / f"fixed_worker_rank_{rank}.log")
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

def load_test_data(data_path: str, max_samples: int = 100) -> List[Dict]:
    """加载测试数据"""
    data = []
    try:
        if Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    data.append(json.loads(line.strip()))
        else:
            # 如果文件不存在，使用默认数据
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
        return data
    except Exception as e:
        print(f"加载数据失败: {e}，使用默认测试数据")
        # 返回默认测试数据
        return [
            {"prompt": "The future of artificial intelligence is", "max_length": 100},
            {"prompt": "In a world where technology advances rapidly", "max_length": 100},
            {"prompt": "Climate change poses significant challenges", "max_length": 100},
            {"prompt": "The benefits of renewable energy include", "max_length": 100},
            {"prompt": "Machine learning algorithms can help us", "max_length": 100},
            {"prompt": "The importance of data privacy is", "max_length": 100},
            {"prompt": "Quantum computing has the potential to", "max_length": 100},
            {"prompt": "The role of automation in manufacturing", "max_length": 100},
            {"prompt": "Sustainable development goals focus on", "max_length": 100},
            {"prompt": "The evolution of communication technology", "max_length": 100},
            {"prompt": "Space exploration continues to reveal", "max_length": 100},
            {"prompt": "The impact of social media on society", "max_length": 100}
        ]

def run_distributed_inference_worker(args):
    """运行分布式推理工作进程"""
    logger = setup_logging()
    
    # 获取分布式信息
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    logger.info(f"启动工作进程 - Local Rank: {local_rank}, World Rank: {world_rank}, World Size: {world_size}")
      # 设置CUDA设备
    if not args.no_cuda and torch.cuda.is_available():
        # 在分布式设置中，CUDA_VISIBLE_DEVICES可能限制了可见GPU数量
        # 因此需要映射local_rank到实际可用的GPU索引
        available_gpus = torch.cuda.device_count()
        actual_device_id = local_rank % available_gpus
        torch.cuda.set_device(actual_device_id)
        device = f"cuda:{actual_device_id}"
        logger.info(f"使用GPU设备: {device}, 可用GPU数量: {available_gpus}, Local Rank: {local_rank}")
    else:
        device = "cpu"
        logger.info("使用CPU设备")
    
    try:
        # 初始化性能监控器
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
          # 加载测试数据
        logger.info(f"加载测试数据: {args.data_path}")
        test_data = load_test_data(args.data_path, getattr(args, 'num_samples', 100))
        
        # 为当前rank分配数据
        chunk_size = len(test_data) // world_size
        start_idx = world_rank * chunk_size
        end_idx = start_idx + chunk_size if world_rank < world_size - 1 else len(test_data)
        local_data = test_data[start_idx:end_idx]
        
        logger.info(f"Rank {world_rank} 处理数据: {len(local_data)} 个样本 (索引 {start_idx}-{end_idx-1})")
        
        # 初始化模型管理器
        logger.info(f"初始化模型管理器，策略: {args.strategy}")
        model_manager = ModelManager(args.model_config)
        
        # 配置并行策略
        strategy_manager = ParallelStrategyManager(args.model_config)
        strategy_config = strategy_manager.get_strategy_config(args.strategy)
        
        # 根据策略初始化模型
        model, tokenizer = model_manager.initialize_model_with_strategy(
            strategy=args.strategy,
            local_rank=local_rank,
            world_size=world_size,
            **strategy_config
        )
        
        logger.info(f"模型初始化完成，策略: {args.strategy}")
        
        # 运行推理测试
        results = []
        total_start_time = time.time()
        
        for iteration in range(args.num_iterations):
            logger.info(f"开始第 {iteration + 1}/{args.num_iterations} 次迭代")
            iteration_start = time.time()
            
            # 批处理推理
            batch_results = []
            for i in range(0, len(local_data), args.batch_size):
                batch_data = local_data[i:i + args.batch_size]
                
                # 准备输入
                prompts = [item['prompt'] for item in batch_data]
                max_length = batch_data[0].get('max_length', 100)
                
                # Tokenize
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # 推理
                batch_start = time.time()
                
                with torch.no_grad():
                    if hasattr(model, 'generate'):
                        # 直接模型生成
                        outputs = model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=max_length,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    elif hasattr(model, 'module') and hasattr(model.module, 'generate'):
                        # DataParallel包装的模型
                        outputs = model.module.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=max_length,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    else:
                        # 手动生成
                        outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
                        # 简单的贪心解码
                        next_token_logits = outputs.logits[:, -1, :]
                        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        outputs = torch.cat([inputs.input_ids, next_tokens], dim=1)
                
                batch_end = time.time()
                batch_time = batch_end - batch_start
                
                # 解码结果
                generated_texts = []
                for j, output in enumerate(outputs):
                    # 只取新生成的部分
                    input_length = inputs.input_ids[j].shape[0]
                    generated_tokens = output[input_length:]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    generated_texts.append(generated_text)
                
                # 记录结果
                for j, (prompt, generated_text) in enumerate(zip(prompts, generated_texts)):
                    result = {
                        'rank': world_rank,
                        'iteration': iteration,
                        'batch_idx': i // args.batch_size,
                        'sample_idx': i + j,
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'input_tokens': inputs.input_ids[j].shape[0],
                        'output_tokens': len(generated_tokens) if 'generated_tokens' in locals() else 0,
                        'batch_time': batch_time,
                        'tokens_per_second': (inputs.input_ids[j].shape[0] + len(generated_tokens)) / batch_time if 'generated_tokens' in locals() else 0
                    }
                    batch_results.append(result)
                
                logger.info(f"批次 {i//args.batch_size + 1} 完成，{len(batch_data)} 个样本，耗时 {batch_time:.2f}s")
            
            iteration_end = time.time()
            iteration_time = iteration_end - iteration_start
            
            results.extend(batch_results)
            logger.info(f"迭代 {iteration + 1} 完成，耗时 {iteration_time:.2f}s")
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
          # 计算性能指标
        total_samples = len(results)
        total_tokens = sum(r.get('input_tokens', 0) + r.get('output_tokens', 0) for r in results)
        throughput = total_tokens / total_time if total_time > 0 else 0
        avg_latency = total_time / total_samples if total_samples > 0 else 0
          # 获取内存使用情况
        memory_stats = {}
        if torch.cuda.is_available():
            # 使用actual_device_id而不是local_rank来获取内存统计
            available_gpus = torch.cuda.device_count()
            actual_device_id = local_rank % available_gpus
            memory_stats = {
                'max_memory_allocated': torch.cuda.max_memory_allocated(actual_device_id) / 1024**2,  # MB
                'max_memory_reserved': torch.cuda.max_memory_reserved(actual_device_id) / 1024**2,  # MB
                'current_memory_allocated': torch.cuda.memory_allocated(actual_device_id) / 1024**2,  # MB
            }
        
        # 停止监控
        monitor.stop_monitoring()
        system_metrics = monitor.get_statistics()
        
        # 准备结果
        final_result = {
            'rank': world_rank,
            'local_rank': local_rank,
            'world_size': world_size,
            'strategy': args.strategy,
            'device': device,
            'total_samples': total_samples,
            'total_time': total_time,
            'metrics': {
                'throughput_tokens_per_sec': throughput,
                'average_latency_sec': avg_latency,
                'total_tokens': total_tokens,
                'total_samples': total_samples,
                'total_time': total_time
            },            'memory_stats': memory_stats,
            'system_metrics': system_metrics,
            'detailed_results': results,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # 保存结果 - 使用绝对路径
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / f"distributed_{args.strategy}_rank_{world_rank}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Rank {world_rank} 结果保存到: {result_file}")
        
        # 打印性能摘要
        logger.info(f"=== Rank {world_rank} 性能摘要 ===")
        logger.info(f"策略: {args.strategy}")
        logger.info(f"设备: {device}")
        logger.info(f"样本数: {total_samples}")
        logger.info(f"总时间: {total_time:.2f}s")
        logger.info(f"吞吐量: {throughput:.2f} tokens/sec")
        logger.info(f"平均延迟: {avg_latency:.2f}s")
        if memory_stats:
            logger.info(f"内存使用: {memory_stats['max_memory_allocated']:.2f}MB")
          # 清理资源
        model_manager.cleanup()
        
        logger.info(f"Rank {world_rank} 推理完成")
        return final_result
        
    except Exception as e:
        logger.error(f"Rank {world_rank} 推理过程出错: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        
        # 保存错误信息
        error_result = {
            'rank': world_rank,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        error_file = results_dir / f"error_{args.strategy}_rank_{world_rank}.json"
        
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=2, ensure_ascii=False)
        
        raise
        
    finally:
        # 确保清理资源
        if 'model_manager' in locals():
            try:
                model_manager.cleanup()
            except:
                pass
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="修复版分布式推理工作进程")
    
    # 模型参数
    parser.add_argument("--model_config", type=str, 
                        default="config/model_config.yaml",
                        help="模型配置文件路径")
    parser.add_argument("--strategy", type=str,
                        choices=['pure_data_parallel', 'tensor_data_hybrid', 
                               'pipeline_data_hybrid', 'full_model_parallel'],
                        default='pure_data_parallel',
                        help="并行策略")    
    # 推理参数
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--data_path", type=str,
                        default="data/test_prompts.jsonl",
                        help="测试数据路径")
    parser.add_argument("--num_iterations", type=int, default=3,
                        help="测试迭代次数")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="测试样本数量")
    
    # 其他参数
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 运行推理
    run_distributed_inference_worker(args)

if __name__ == "__main__":
    main()
