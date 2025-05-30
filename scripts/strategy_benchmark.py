"""
并行策略性能比较脚本
自动测试不同并行策略的性能表现
"""

import os
import sys
import subprocess
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import yaml
from typing import Dict, List, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.parallel_strategy import ParallelStrategyManager

class StrategyBenchmark:
    """并行策略基准测试"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """初始化基准测试"""
        self.config_path = config_path
        self.strategy_manager = ParallelStrategyManager(config_path)
        self.results = {}
        self.results_dir = Path("results/strategy_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_strategy_test(self, strategy_name: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个策略测试"""
        print(f"\\n测试策略: {strategy_name}")
        print("-" * 50)
        
        # 构建命令
        cmd = [
            sys.executable, 
            "scripts/run_distributed_inference.py",
            "--strategy", strategy_name,
            "--world_size", str(test_config.get('world_size', 4)),
            "--batch_size", str(test_config.get('batch_size', 1)),
            "--num_iterations", str(test_config.get('num_iterations', 50)),
            "--data_path", test_config.get('data_path', 'data/processed/test_dataset.jsonl')
        ]
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行测试
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=test_config.get('timeout', 1800)  # 30分钟超时
            )
            
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"策略 {strategy_name} 测试成功")
                
                # 解析结果
                metrics = self._parse_results(strategy_name)
                metrics['total_time'] = end_time - start_time
                metrics['success'] = True
                metrics['error_message'] = None
                
                return metrics
            else:
                print(f"策略 {strategy_name} 测试失败:")
                print(f"stderr: {result.stderr}")
                
                return {
                    'success': False,
                    'error_message': result.stderr,
                    'total_time': end_time - start_time
                }
        
        except subprocess.TimeoutExpired:
            print(f"策略 {strategy_name} 测试超时")
            return {
                'success': False,
                'error_message': 'Timeout',
                'total_time': test_config.get('timeout', 1800)
            }
        except Exception as e:
            print(f"策略 {strategy_name} 测试出错: {str(e)}")
            return {
                'success': False,
                'error_message': str(e),
                'total_time': time.time() - start_time
            }
    
    def _parse_results(self, strategy_name: str) -> Dict[str, Any]:
        """解析测试结果"""
        # 查找最新的结果文件
        pattern = f"inference_results_{strategy_name}_*.json"
        result_files = list(self.results_dir.parent.glob(pattern))
        
        if not result_files:
            # 如果没有找到，搜索所有结果文件
            result_files = list(self.results_dir.parent.glob("inference_results_*.json"))
            result_files = [f for f in result_files if strategy_name in f.name]
        
        if result_files:
            # 使用最新的文件
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('metrics', {})
        
        return {}
    
    def run_full_benchmark(self, test_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """运行完整基准测试"""
        print("开始并行策略性能比较测试")
        print("=" * 80)
        
        # 获取所有可用策略
        available_strategies = self.strategy_manager.get_available_strategies()
        
        print(f"将测试以下策略:")
        for name, desc in available_strategies.items():
            print(f"  - {name}: {desc}")
        
        print(f"\\n测试配置: {test_config}")
        
        # 逐个测试策略
        results = {}
        for strategy_name in available_strategies.keys():
            try:
                result = self.run_strategy_test(strategy_name, test_config)
                results[strategy_name] = result
                
                # 打印简要结果
                if result['success']:
                    throughput = result.get('avg_throughput', 0)
                    latency = result.get('avg_latency', 0)
                    memory = result.get('avg_memory_usage', {}).get('gpu_memory_allocated_mb', 0)
                    print(f"  ✓ 吞吐量: {throughput:.2f} tokens/s, 延迟: {latency:.3f}s, 显存: {memory:.0f}MB")
                else:
                    print(f"  ✗ 失败: {result.get('error_message', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print(f"\\n用户中断测试")
                break
            except Exception as e:
                print(f"策略 {strategy_name} 测试异常: {str(e)}")
                results[strategy_name] = {
                    'success': False,
                    'error_message': str(e),
                    'total_time': 0
                }
        
        # 保存综合结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        benchmark_file = self.results_dir / f"strategy_benchmark_{timestamp}.json"
        
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_config': test_config,
                'results': results,
                'timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\\n基准测试完成，结果保存到: {benchmark_file}")
        return results
    
    def generate_comparison_report(self, results: Dict[str, Dict[str, Any]]) -> None:
        """生成比较报告"""
        print("\\n生成性能比较报告...")
        
        # 准备数据
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if not successful_results:
            print("没有成功的测试结果，无法生成报告")
            return
        
        # 创建DataFrame
        data = []
        for strategy, metrics in successful_results.items():
            data.append({
                '策略': strategy,
                '吞吐量(tokens/s)': metrics.get('avg_throughput', 0),
                '平均延迟(s)': metrics.get('avg_latency', 0), 
                '首令牌延迟(s)': metrics.get('avg_first_token_latency', 0),
                'GPU利用率(%)': metrics.get('avg_gpu_utilization', 0),
                '显存使用(MB)': metrics.get('avg_memory_usage', {}).get('gpu_memory_allocated_mb', 0),
                '总耗时(s)': metrics.get('total_time', 0)
            })
        
        df = pd.DataFrame(data)
        
        # 保存表格
        table_file = self.results_dir / f"strategy_comparison_table_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(table_file, index=False, encoding='utf-8-sig')
        
        # 打印表格
        print("\\n性能比较表:")
        print(df.to_string(index=False, float_format='%.3f'))
        
        # 生成可视化图表
        self._generate_visualizations(df)
        
        # 生成推荐
        self._generate_recommendations(df)
    
    def _generate_visualizations(self, df: pd.DataFrame) -> None:
        """生成可视化图表"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('并行策略性能比较', fontsize=16, fontweight='bold')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 吞吐量比较
        axes[0, 0].bar(df['策略'], df['吞吐量(tokens/s)'], color='skyblue')
        axes[0, 0].set_title('吞吐量比较')
        axes[0, 0].set_ylabel('tokens/s')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 延迟比较
        axes[0, 1].bar(df['策略'], df['平均延迟(s)'], color='lightcoral')
        axes[0, 1].set_title('平均延迟比较')
        axes[0, 1].set_ylabel('秒')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. GPU利用率比较
        axes[0, 2].bar(df['策略'], df['GPU利用率(%)'], color='lightgreen')
        axes[0, 2].set_title('GPU利用率比较')
        axes[0, 2].set_ylabel('%')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 显存使用比较
        axes[1, 0].bar(df['策略'], df['显存使用(MB)'], color='orange')
        axes[1, 0].set_title('显存使用比较')
        axes[1, 0].set_ylabel('MB')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. 首令牌延迟比较
        axes[1, 1].bar(df['策略'], df['首令牌延迟(s)'], color='mediumpurple')
        axes[1, 1].set_title('首令牌延迟比较')
        axes[1, 1].set_ylabel('秒')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. 综合评分（吞吐量/延迟比）
        df['综合评分'] = df['吞吐量(tokens/s)'] / (df['平均延迟(s)'] + 0.001)  # 避免除零
        axes[1, 2].bar(df['策略'], df['综合评分'], color='gold')
        axes[1, 2].set_title('综合评分 (吞吐量/延迟)')
        axes[1, 2].set_ylabel('分数')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.results_dir / f"strategy_comparison_charts_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化图表保存到: {chart_file}")
    
    def _generate_recommendations(self, df: pd.DataFrame) -> None:
        """生成策略推荐"""
        print("\\n策略推荐:")
        print("-" * 50)
        
        # 最高吞吐量
        best_throughput = df.loc[df['吞吐量(tokens/s)'].idxmax()]
        print(f"最高吞吐量: {best_throughput['策略']} ({best_throughput['吞吐量(tokens/s)']:.2f} tokens/s)")
        
        # 最低延迟
        best_latency = df.loc[df['平均延迟(s)'].idxmin()]
        print(f"最低延迟: {best_latency['策略']} ({best_latency['平均延迟(s)']:.3f}s)")
        
        # 最高GPU利用率
        best_gpu = df.loc[df['GPU利用率(%)'].idxmax()]
        print(f"最高GPU利用率: {best_gpu['策略']} ({best_gpu['GPU利用率(%)']:.1f}%)")
        
        # 最低显存使用
        best_memory = df.loc[df['显存使用(MB)'].idxmin()]
        print(f"最低显存使用: {best_memory['策略']} ({best_memory['显存使用(MB)']:.0f}MB)")
        
        # 综合推荐
        df['综合评分'] = (
            df['吞吐量(tokens/s)'] / df['吞吐量(tokens/s)'].max() * 0.4 +
            (1 - df['平均延迟(s)'] / df['平均延迟(s)'].max()) * 0.3 +
            df['GPU利用率(%)'] / df['GPU利用率(%)'].max() * 0.2 +
            (1 - df['显存使用(MB)'] / df['显存使用(MB)'].max()) * 0.1
        )
        
        best_overall = df.loc[df['综合评分'].idxmax()]
        print(f"\\n综合推荐: {best_overall['策略']} (综合评分: {best_overall['综合评分']:.3f})")
        
        # RTX 3080特定建议
        print("\\nRTX 3080优化建议:")
        print("- 对于高吞吐量需求，推荐使用tensor_data_hybrid策略")
        print("- 对于低延迟需求，推荐使用pure_data_parallel策略") 
        print("- 对于显存受限场景，推荐使用full_model_parallel策略")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="并行策略性能比较")
    
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="测试批次大小")
    parser.add_argument("--num_iterations", type=int, default=50,
                        help="测试迭代次数")
    parser.add_argument("--world_size", type=int, default=4,
                        help="GPU数量")
    parser.add_argument("--data_path", type=str, default="data/processed/test_dataset.jsonl",
                        help="测试数据路径")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="单个策略测试超时时间（秒）")
    parser.add_argument("--strategies", nargs='+', 
                        default=["pure_data_parallel", "tensor_data_hybrid", 
                                "pipeline_data_hybrid", "full_model_parallel"],
                        help="要测试的策略列表")
    
    args = parser.parse_args()
    
    # 创建基准测试器
    benchmark = StrategyBenchmark(args.config)
    
    # 测试配置
    test_config = {
        'batch_size': args.batch_size,
        'num_iterations': args.num_iterations,
        'world_size': args.world_size,
        'data_path': args.data_path,
        'timeout': args.timeout
    }
    
    # 运行基准测试
    results = benchmark.run_full_benchmark(test_config)
    
    # 生成比较报告
    benchmark.generate_comparison_report(results)

if __name__ == "__main__":
    main()
