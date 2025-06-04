#!/usr/bin/env python3
"""
最终性能分析报告生成器
汇总所有分布式推理测试结果并生成综合分析报告
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinalPerformanceReporter:
    """最终性能报告生成器"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.all_results = {}
        self.performance_df = None
        
    def load_all_results(self):
        """加载所有测试结果"""
        print("=== 加载所有测试结果 ===")
        
        # 定义所有测试配置
        test_configs = [
            # 1GPU测试
            ("pure_data_parallel", 1, "distributed_pure_data_parallel_rank_0.json"),
            
            # 2GPU测试
            ("pure_data_parallel", 2, "merged_pure_data_parallel_gpu2_*.json"),
            ("tensor_data_hybrid", 2, "merged_tensor_data_hybrid_gpu2_*.json"),
            ("pipeline_data_hybrid", 2, "merged_pipeline_data_hybrid_gpu2_*.json"),
            ("full_model_parallel", 2, "merged_full_model_parallel_gpu2_*.json"),
            
            # 4GPU测试
            ("pure_data_parallel", 4, "merged_pure_data_parallel_gpu4_*.json"),
            ("tensor_data_hybrid", 4, "merged_tensor_data_hybrid_gpu4_*.json"),
            ("pipeline_data_hybrid", 4, "merged_pipeline_data_hybrid_gpu4_*.json"),
            ("full_model_parallel", 4, "merged_full_model_parallel_gpu4_*.json"),
        ]
        
        for strategy, num_gpus, pattern in test_configs:
            print(f"查找 {strategy} ({num_gpus}GPU): {pattern}")
            
            if "*" in pattern:
                # 查找匹配的文件
                matching_files = list(self.results_dir.glob(pattern))
                if matching_files:
                    # 取最新的文件
                    result_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                else:
                    print(f"  未找到匹配文件")
                    continue
            else:
                result_file = self.results_dir / pattern
                if not result_file.exists():
                    print(f"  文件不存在: {result_file}")
                    continue
            
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                key = f"{strategy}_{num_gpus}gpu"
                self.all_results[key] = data
                print(f"  ✓ 已加载: {result_file.name}")
                
            except Exception as e:
                print(f"  ✗ 加载失败: {e}")
        
        print(f"\n总共加载了 {len(self.all_results)} 个测试结果")
        
    def extract_performance_metrics(self):
        """提取性能指标"""
        print("\n=== 提取性能指标 ===")
        
        performance_data = []
        
        for config_key, data in self.all_results.items():
            strategy, gpu_config = config_key.rsplit('_', 1)
            num_gpus = int(gpu_config.replace('gpu', ''))
            
            # 提取主要指标
            if 'aggregate_metrics' in data:
                metrics = data['aggregate_metrics']
                
                performance_data.append({
                    'strategy': strategy,
                    'num_gpus': num_gpus,
                    'throughput_tokens_per_sec': metrics.get('total_throughput_tokens_per_sec', 0),
                    'average_latency_sec': metrics.get('average_latency_sec', 0),
                    'speedup': metrics.get('speedup', 1),
                    'parallel_efficiency': metrics.get('parallel_efficiency', 1),
                    'total_samples': metrics.get('total_samples', 0),
                    'total_time': metrics.get('total_time', 0),
                    'memory_usage_mb': metrics.get('average_memory_allocated_mb', 0),
                    'gpu_utilization': metrics.get('average_gpu_utilization', 0)
                })
            elif 'metrics' in data:
                # 单GPU结果格式
                metrics = data['metrics']
                
                performance_data.append({
                    'strategy': strategy,
                    'num_gpus': num_gpus,
                    'throughput_tokens_per_sec': metrics.get('throughput_tokens_per_sec', 0),
                    'average_latency_sec': metrics.get('average_latency_sec', 0),
                    'speedup': 1.0,  # 基准
                    'parallel_efficiency': 1.0,  # 基准
                    'total_samples': metrics.get('total_samples', 0),
                    'total_time': metrics.get('total_time', 0),
                    'memory_usage_mb': data.get('memory_stats', {}).get('max_memory_allocated', 0),
                    'gpu_utilization': data.get('system_metrics', {}).get('gpu_utilization', {}).get('average', 0) * 100
                })
        
        self.performance_df = pd.DataFrame(performance_data)
        print(f"提取了 {len(performance_data)} 条性能记录")
        
        # 显示数据概览
        if not self.performance_df.empty:
            print("\n性能数据概览:")
            print(self.performance_df.to_string(index=False))
    
    def calculate_baseline_speedup(self):
        """计算相对于1GPU pure_data_parallel的加速比"""
        if self.performance_df is None or self.performance_df.empty:
            return
        
        # 找到基准性能(1GPU pure_data_parallel)
        baseline = self.performance_df[
            (self.performance_df['strategy'] == 'pure_data_parallel') & 
            (self.performance_df['num_gpus'] == 1)
        ]
        
        if baseline.empty:
            print("警告: 未找到1GPU基准测试结果")
            return
        
        baseline_throughput = baseline.iloc[0]['throughput_tokens_per_sec']
        
        # 计算所有配置相对于基准的加速比
        self.performance_df['speedup_vs_baseline'] = (
            self.performance_df['throughput_tokens_per_sec'] / baseline_throughput
        )
        
        # 计算理论并行效率
        self.performance_df['theoretical_efficiency'] = (
            self.performance_df['speedup_vs_baseline'] / self.performance_df['num_gpus'] * 100
        )
        
        print(f"\n基准吞吐量 (1GPU pure_data_parallel): {baseline_throughput:.2f} tokens/sec")
    
    def generate_comparison_charts(self):
        """生成对比图表"""
        if self.performance_df is None or self.performance_df.empty:
            print("无数据可绘图")
            return
        
        print("\n=== 生成性能对比图表 ===")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GPT-1.5B 分布式推理性能全面对比', fontsize=16, fontweight='bold')
        
        # 1. 吞吐量对比
        ax1 = axes[0, 0]
        strategies = self.performance_df['strategy'].unique()
        gpu_counts = sorted(self.performance_df['num_gpus'].unique())
        
        bar_width = 0.15
        x = np.arange(len(gpu_counts))
        
        for i, strategy in enumerate(strategies):
            strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
            throughputs = []
            for gpu_count in gpu_counts:
                gpu_data = strategy_data[strategy_data['num_gpus'] == gpu_count]
                if not gpu_data.empty:
                    throughputs.append(gpu_data.iloc[0]['throughput_tokens_per_sec'])
                else:
                    throughputs.append(0)
            
            ax1.bar(x + i * bar_width, throughputs, bar_width, 
                   label=strategy.replace('_', ' ').title(), alpha=0.8)
        
        ax1.set_xlabel('GPU数量')
        ax1.set_ylabel('吞吐量 (tokens/sec)')
        ax1.set_title('各策略吞吐量对比')
        ax1.set_xticks(x + bar_width * (len(strategies) - 1) / 2)
        ax1.set_xticklabels([f'{gpu}GPU' for gpu in gpu_counts])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 延迟对比
        ax2 = axes[0, 1]
        for i, strategy in enumerate(strategies):
            strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
            latencies = []
            for gpu_count in gpu_counts:
                gpu_data = strategy_data[strategy_data['num_gpus'] == gpu_count]
                if not gpu_data.empty:
                    latencies.append(gpu_data.iloc[0]['average_latency_sec'])
                else:
                    latencies.append(0)
            
            ax2.bar(x + i * bar_width, latencies, bar_width, 
                   label=strategy.replace('_', ' ').title(), alpha=0.8)
        
        ax2.set_xlabel('GPU数量')
        ax2.set_ylabel('平均延迟 (秒)')
        ax2.set_title('各策略延迟对比')
        ax2.set_xticks(x + bar_width * (len(strategies) - 1) / 2)
        ax2.set_xticklabels([f'{gpu}GPU' for gpu in gpu_counts])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 加速比对比
        ax3 = axes[0, 2]
        if 'speedup_vs_baseline' in self.performance_df.columns:
            for i, strategy in enumerate(strategies):
                strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
                speedups = []
                for gpu_count in gpu_counts:
                    gpu_data = strategy_data[strategy_data['num_gpus'] == gpu_count]
                    if not gpu_data.empty:
                        speedups.append(gpu_data.iloc[0]['speedup_vs_baseline'])
                    else:
                        speedups.append(0)
                
                ax3.bar(x + i * bar_width, speedups, bar_width, 
                       label=strategy.replace('_', ' ').title(), alpha=0.8)
        
        # 添加理论最大加速比线
        ax3.plot(x + bar_width * (len(strategies) - 1) / 2, gpu_counts, 
                'r--', linewidth=2, label='理论最大加速比')
        
        ax3.set_xlabel('GPU数量')
        ax3.set_ylabel('加速比')
        ax3.set_title('各策略加速比对比')
        ax3.set_xticks(x + bar_width * (len(strategies) - 1) / 2)
        ax3.set_xticklabels([f'{gpu}GPU' for gpu in gpu_counts])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 并行效率对比
        ax4 = axes[1, 0]
        if 'theoretical_efficiency' in self.performance_df.columns:
            for i, strategy in enumerate(strategies):
                strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
                efficiencies = []
                for gpu_count in gpu_counts:
                    gpu_data = strategy_data[strategy_data['num_gpus'] == gpu_count]
                    if not gpu_data.empty:
                        efficiencies.append(gpu_data.iloc[0]['theoretical_efficiency'])
                    else:
                        efficiencies.append(0)
                
                ax4.bar(x + i * bar_width, efficiencies, bar_width, 
                       label=strategy.replace('_', ' ').title(), alpha=0.8)
        
        ax4.axhline(y=100, color='r', linestyle='--', linewidth=2, label='理想效率(100%)')
        ax4.set_xlabel('GPU数量')
        ax4.set_ylabel('并行效率 (%)')
        ax4.set_title('各策略并行效率对比')
        ax4.set_xticks(x + bar_width * (len(strategies) - 1) / 2)
        ax4.set_xticklabels([f'{gpu}GPU' for gpu in gpu_counts])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 内存使用对比
        ax5 = axes[1, 1]
        for i, strategy in enumerate(strategies):
            strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
            memory_usages = []
            for gpu_count in gpu_counts:
                gpu_data = strategy_data[strategy_data['num_gpus'] == gpu_count]
                if not gpu_data.empty:
                    memory_usages.append(gpu_data.iloc[0]['memory_usage_mb'])
                else:
                    memory_usages.append(0)
            
            ax5.bar(x + i * bar_width, memory_usages, bar_width, 
                   label=strategy.replace('_', ' ').title(), alpha=0.8)
        
        ax5.set_xlabel('GPU数量')
        ax5.set_ylabel('内存使用 (MB)')
        ax5.set_title('各策略内存使用对比')
        ax5.set_xticks(x + bar_width * (len(strategies) - 1) / 2)
        ax5.set_xticklabels([f'{gpu}GPU' for gpu in gpu_counts])
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. GPU利用率对比
        ax6 = axes[1, 2]
        for i, strategy in enumerate(strategies):
            strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
            utilizations = []
            for gpu_count in gpu_counts:
                gpu_data = strategy_data[strategy_data['num_gpus'] == gpu_count]
                if not gpu_data.empty:
                    utilizations.append(gpu_data.iloc[0]['gpu_utilization'])
                else:
                    utilizations.append(0)
            
            ax6.bar(x + i * bar_width, utilizations, bar_width, 
                   label=strategy.replace('_', ' ').title(), alpha=0.8)
        
        ax6.set_xlabel('GPU数量')
        ax6.set_ylabel('GPU利用率 (%)')
        ax6.set_title('各策略GPU利用率对比')
        ax6.set_xticks(x + bar_width * (len(strategies) - 1) / 2)
        ax6.set_xticklabels([f'{gpu}GPU' for gpu in gpu_counts])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.results_dir / f"final_performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"性能对比图表已保存: {chart_file}")
    
    def generate_detailed_report(self):
        """生成详细分析报告"""
        if self.performance_df is None or self.performance_df.empty:
            print("无数据生成报告")
            return
        
        print("\n=== 生成详细分析报告 ===")
        
        report_lines = []
        report_lines.append("# GPT-1.5B 分布式推理性能全面测试报告")
        report_lines.append(f"## 测试时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        report_lines.append("")
        
        # 测试概览
        report_lines.append("## 1. 测试概览")
        report_lines.append(f"- **测试策略**: {len(self.performance_df['strategy'].unique())} 种并行策略")
        report_lines.append(f"- **GPU配置**: {list(sorted(self.performance_df['num_gpus'].unique()))} GPU")
        report_lines.append(f"- **总测试配置**: {len(self.performance_df)} 个")
        report_lines.append("")
        
        # 性能摘要表格
        report_lines.append("## 2. 性能摘要")
        report_lines.append("| 策略 | GPU数 | 吞吐量(tokens/sec) | 延迟(sec) | 加速比 | 并行效率(%) | 内存(MB) | GPU利用率(%) |")
        report_lines.append("|------|-------|-------------------|----------|---------|-------------|-----------|-------------|")
        
        for _, row in self.performance_df.iterrows():
            speedup = row.get('speedup_vs_baseline', row.get('speedup', 1))
            efficiency = row.get('theoretical_efficiency', row.get('parallel_efficiency', 100))
            
            report_lines.append(f"| {row['strategy'].replace('_', ' ').title()} | "
                              f"{row['num_gpus']} | "
                              f"{row['throughput_tokens_per_sec']:.2f} | "
                              f"{row['average_latency_sec']:.3f} | "
                              f"{speedup:.2f}x | "
                              f"{efficiency:.1f} | "
                              f"{row['memory_usage_mb']:.1f} | "
                              f"{row['gpu_utilization']:.1f} |")
        
        report_lines.append("")
        
        # 关键发现
        report_lines.append("## 3. 关键发现")
        
        # 找出最佳性能配置
        best_throughput = self.performance_df.loc[self.performance_df['throughput_tokens_per_sec'].idxmax()]
        best_efficiency = self.performance_df.loc[self.performance_df.get('theoretical_efficiency', pd.Series([0])).idxmax()]
        
        report_lines.append("### 3.1 最佳性能配置")
        report_lines.append(f"- **最高吞吐量**: {best_throughput['strategy'].replace('_', ' ').title()} "
                          f"({best_throughput['num_gpus']}GPU) - {best_throughput['throughput_tokens_per_sec']:.2f} tokens/sec")
        
        if 'theoretical_efficiency' in self.performance_df.columns and not self.performance_df['theoretical_efficiency'].isna().all():
            report_lines.append(f"- **最高效率**: {best_efficiency['strategy'].replace('_', ' ').title()} "
                              f"({best_efficiency['num_gpus']}GPU) - {best_efficiency['theoretical_efficiency']:.1f}%")
        
        # 扩展性分析
        report_lines.append("")
        report_lines.append("### 3.2 扩展性分析")
        
        for strategy in self.performance_df['strategy'].unique():
            strategy_data = self.performance_df[self.performance_df['strategy'] == strategy].sort_values('num_gpus')
            if len(strategy_data) > 1:
                max_speedup = strategy_data['speedup_vs_baseline'].max() if 'speedup_vs_baseline' in strategy_data.columns else strategy_data['speedup'].max()
                max_gpus = strategy_data['num_gpus'].max()
                report_lines.append(f"- **{strategy.replace('_', ' ').title()}**: 最大{max_speedup:.2f}x加速比 ({max_gpus}GPU)")
        
        # 资源利用分析
        report_lines.append("")
        report_lines.append("### 3.3 资源利用分析")
        avg_gpu_util = self.performance_df['gpu_utilization'].mean()
        avg_memory = self.performance_df['memory_usage_mb'].mean()
        
        report_lines.append(f"- **平均GPU利用率**: {avg_gpu_util:.1f}%")
        report_lines.append(f"- **平均内存使用**: {avg_memory:.1f}MB")
        
        # 策略对比
        report_lines.append("")
        report_lines.append("### 3.4 策略对比结论")
        
        # 按策略分组分析
        strategy_summary = self.performance_df.groupby('strategy').agg({
            'throughput_tokens_per_sec': 'mean',
            'average_latency_sec': 'mean',
            'gpu_utilization': 'mean'
        }).round(2)
        
        for strategy, metrics in strategy_summary.iterrows():
            report_lines.append(f"- **{strategy.replace('_', ' ').title()}**: "
                              f"平均吞吐量 {metrics['throughput_tokens_per_sec']:.2f} tokens/sec, "
                              f"平均延迟 {metrics['average_latency_sec']:.3f}s, "
                              f"平均GPU利用率 {metrics['gpu_utilization']:.1f}%")
        
        # 建议
        report_lines.append("")
        report_lines.append("## 4. 使用建议")
        report_lines.append("- **高吞吐量场景**: 推荐使用4GPU配置的任意并行策略")
        report_lines.append("- **资源受限场景**: 推荐使用2GPU Pure Data Parallel策略")
        report_lines.append("- **低延迟场景**: 推荐使用4GPU配置以获得最低延迟")
        report_lines.append("- **内存优化场景**: 根据测试结果选择内存使用最少的策略")
        
        # 技术规格
        report_lines.append("")
        report_lines.append("## 5. 测试环境")
        report_lines.append("- **模型**: GPT-1.5B")
        report_lines.append("- **测试数据**: 20个多样化prompt")
        report_lines.append("- **批次大小**: 4")
        report_lines.append("- **迭代次数**: 3次")
        report_lines.append("- **测试策略**: Pure Data Parallel, Tensor+Data Hybrid, Pipeline+Data Hybrid, Full Model Parallel")
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_file = self.results_dir / f"final_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"详细报告已保存: {report_file}")
        
        # 在控制台显示摘要
        print("\n" + "="*60)
        print("性能测试摘要")
        print("="*60)
        for line in report_lines[4:20]:  # 显示概览部分
            print(line)
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("开始生成最终性能分析报告...")
        
        # 1. 加载所有结果
        self.load_all_results()
        
        if not self.all_results:
            print("错误: 未找到任何测试结果文件")
            return
        
        # 2. 提取性能指标
        self.extract_performance_metrics()
        
        # 3. 计算基准加速比
        self.calculate_baseline_speedup()
        
        # 4. 生成对比图表
        self.generate_comparison_charts()
        
        # 5. 生成详细报告
        self.generate_detailed_report()
        
        print("\n✅ 最终性能分析报告生成完成!")

def main():
    """主函数"""
    reporter = FinalPerformanceReporter()
    reporter.run_complete_analysis()

if __name__ == "__main__":
    main()
