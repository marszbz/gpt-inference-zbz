"""
结果分析脚本
用于分析和可视化推理性能测试结果
"""

import sys
import argparse
import logging
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import PerformanceEvaluator, MetricsComparator

def setup_logging(log_level: str = "INFO") -> None:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/analysis.log')
        ]
    )

def find_latest_result_file(results_dir: str = "results") -> Path:
    """查找最新的结果文件"""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"结果目录不存在: {results_dir}")
    
    json_files = list(results_path.glob("benchmark_results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"在 {results_dir} 中未找到结果文件")
    
    # 返回最新的文件
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    return latest_file

def analyze_single_result(args):
    """分析单个结果文件"""
    logger = logging.getLogger(__name__)
    
    # 确定结果文件路径
    if args.result_file:
        result_file = Path(args.result_file)
    else:
        result_file = find_latest_result_file(args.results_dir)
    
    if not result_file.exists():
        logger.error(f"结果文件不存在: {result_file}")
        return
    
    logger.info(f"分析结果文件: {result_file}")
    
    # 创建评估器
    evaluator = PerformanceEvaluator(str(result_file))
    evaluator.load_results()
    
    # 生成分析报告
    output_dir = args.output_dir or f"results/analysis_{result_file.stem}"
    report_path = evaluator.generate_performance_report(output_dir)
    
    logger.info(f"分析报告已生成: {report_path}")
    
    # 如果指定了自动打开，则打开报告
    if args.open_report:
        import webbrowser
        webbrowser.open(f"file://{Path(report_path).absolute()}")

def compare_multiple_results(args):
    """比较多个结果文件"""
    logger = logging.getLogger(__name__)
    
    if len(args.compare_files) < 2:
        logger.error("至少需要提供两个结果文件进行比较")
        return
    
    logger.info(f"比较 {len(args.compare_files)} 个结果文件")
    
    # 创建比较器
    comparator = MetricsComparator()
    comparison_results = comparator.compare_results(args.compare_files)
    
    # 保存比较结果
    output_dir = Path(args.output_dir or "results/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_dir / "comparison_results.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"比较结果已保存: {comparison_file}")

def print_result_summary(args):
    """打印结果摘要"""
    logger = logging.getLogger(__name__)
    
    # 确定结果文件路径
    if args.result_file:
        result_file = Path(args.result_file)
    else:
        result_file = find_latest_result_file(args.results_dir)
    
    logger.info(f"读取结果文件: {result_file}")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n=== GPT-1.5B 分布式推理性能测试结果摘要 ===")
    
    # 检查数据格式，支持两种格式：
    # 1. 新的分布式格式：包含 rank, strategy, device 等顶级字段
    # 2. 旧的基准测试格式：包含 metadata 和 results 字段
    
    if 'rank' in data and 'strategy' in data:
        # 新的分布式测试格式
        print(f"Rank: {data.get('rank', 'N/A')}")
        print(f"分布式策略: {data.get('strategy', 'N/A')}")
        print(f"设备: {data.get('device', 'N/A')}")
        print(f"总样本数: {data.get('total_samples', 'N/A')}")
        print(f"总用时: {data.get('total_time', 0):.2f} 秒")
        
        if 'metrics' in data:
            metrics = data['metrics']
            print(f"\n=== 性能指标 ===")
            print(f"吞吐量: {metrics.get('throughput_tokens_per_sec', 0):.2f} tokens/s")
            print(f"平均延迟: {metrics.get('average_latency_sec', 0):.2f} 秒")
            print(f"总token数: {metrics.get('total_tokens', 0)}")
        
        if 'memory_stats' in data:
            memory = data['memory_stats']
            print(f"\n=== 内存使用 ===")
            print(f"最大分配内存: {memory.get('max_memory_allocated', 0):.2f} MB")
            print(f"最大保留内存: {memory.get('max_memory_reserved', 0):.2f} MB")
            print(f"当前分配内存: {memory.get('current_memory_allocated', 0):.2f} MB")
        
        if 'system_metrics' in data:
            system = data['system_metrics']
            if 'gpu_usage' in system:
                gpu_util = system['gpu_usage']
                print(f"\n=== 系统指标 ===")
                print(f"平均GPU利用率: {gpu_util.get('mean', 0):.1f}%")
                print(f"最大GPU利用率: {gpu_util.get('max', 0):.1f}%")
                print(f"GPU利用率标准差: {gpu_util.get('std', 0):.2f}%")
                
            if 'cpu_usage' in system:
                cpu_util = system['cpu_usage']
                print(f"平均CPU利用率: {cpu_util.get('mean', 0):.1f}%")
                
            if 'gpu_memory' in system:
                gpu_mem = system['gpu_memory']
                print(f"平均GPU内存使用: {gpu_mem.get('mean', 0):.1f}%")
    
    elif 'metadata' in data and 'results' in data:
        # 旧的基准测试格式
        print(f"测试时间: {data['metadata']['timestamp']}")
        print(f"模型: {data['metadata']['model_info'].get('model_name', 'N/A')}")
        print(f"总参数量: {data['metadata']['model_info'].get('num_parameters', 'N/A')}")
        print(f"设备: {data['metadata']['model_info'].get('device', 'N/A')}")
        print(f"分布式: {data['metadata']['model_info'].get('is_distributed', False)}")
        
        print("\n=== 配置性能对比 ===")
        print(f"{'配置ID':<8} {'Prompt长度':<12} {'生成长度':<10} {'吞吐量':<15} {'延迟':<12} {'GPU利用率':<12}")
        print("-" * 80)
        
        for config_id, config_data in data['results'].items():
            config = config_data['config']
            stats = config_data['statistics']
            
            throughput = stats['throughput']['mean']
            latency = stats['latency']['total_time']['mean']
            gpu_util = stats['gpu_utilization']['mean']
            
            print(f"{config_id:<8} {config['prompt_length']:<12} {config['generation_length']:<10} "
                  f"{throughput:<15.2f} {latency:<12.2f} {gpu_util:<12.1f}")
        
        # 计算总体统计
        all_throughputs = [data['results'][cid]['statistics']['throughput']['mean'] 
                          for cid in data['results']]
        all_latencies = [data['results'][cid]['statistics']['latency']['total_time']['mean'] 
                        for cid in data['results']]
        all_gpu_utils = [data['results'][cid]['statistics']['gpu_utilization']['mean'] 
                        for cid in data['results']]
        
        print("\n=== 总体统计 ===")
        print(f"平均吞吐量: {sum(all_throughputs)/len(all_throughputs):.2f} tokens/s")
        print(f"最大吞吐量: {max(all_throughputs):.2f} tokens/s")
        print(f"平均延迟: {sum(all_latencies)/len(all_latencies):.2f} ms")
        print(f"最小延迟: {min(all_latencies):.2f} ms")
        print(f"平均GPU利用率: {sum(all_gpu_utils)/len(all_gpu_utils):.1f}%")
    
    else:
        print("⚠️ 无法识别的结果文件格式")
        print("支持的格式:")
        print("1. 分布式测试结果（包含 rank, strategy 字段）")
        print("2. 基准测试结果（包含 metadata, results 字段）")

def main():
    parser = argparse.ArgumentParser(description='分析GPT推理性能测试结果')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 分析单个结果
    analyze_parser = subparsers.add_parser('analyze', help='分析单个结果文件')
    analyze_parser.add_argument('--result-file', type=str,
                               help='指定结果文件路径（默认使用最新文件）')
    analyze_parser.add_argument('--results-dir', type=str, default='results',
                               help='结果文件目录')
    analyze_parser.add_argument('--output-dir', type=str,
                               help='输出目录路径')
    analyze_parser.add_argument('--open-report', action='store_true',
                               help='生成报告后自动打开')
    
    # 比较多个结果
    compare_parser = subparsers.add_parser('compare', help='比较多个结果文件')
    compare_parser.add_argument('compare_files', nargs='+',
                               help='要比较的结果文件路径列表')
    compare_parser.add_argument('--output-dir', type=str,
                               help='输出目录路径')
    
    # 打印摘要
    summary_parser = subparsers.add_parser('summary', help='打印结果摘要')
    summary_parser.add_argument('--result-file', type=str,
                               help='指定结果文件路径（默认使用最新文件）')
    summary_parser.add_argument('--results-dir', type=str, default='results',
                               help='结果文件目录')
    
    # 全局参数
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 如果没有指定命令，默认为分析
    if not args.command:
        args.command = 'analyze'
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == 'analyze':
            analyze_single_result(args)
        elif args.command == 'compare':
            compare_multiple_results(args)
        elif args.command == 'summary':
            print_result_summary(args)
        else:
            logger.error(f"未知命令: {args.command}")
            
    except Exception as e:
        logger.error(f"分析过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
