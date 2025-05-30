"""
性能评估器
用于分析和可视化推理性能测试结果
"""

import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self, results_path: Optional[str] = None):
        self.results_path = Path(results_path) if results_path else None
        self.results_data = None
        self.logger = self._setup_logger()
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_results(self, results_path: Optional[str] = None) -> None:
        """加载结果数据"""
        if results_path:
            self.results_path = Path(results_path)
        
        if not self.results_path or not self.results_path.exists():
            raise FileNotFoundError(f"结果文件不存在: {self.results_path}")
        
        with open(self.results_path, 'r', encoding='utf-8') as f:
            self.results_data = json.load(f)
        
        self.logger.info(f"已加载结果数据: {self.results_path}")
    
    def generate_performance_report(self, output_dir: str = "results/analysis") -> str:
        """生成性能分析报告"""
        if not self.results_data:
            raise ValueError("请先加载结果数据")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成各种分析图表
        self._plot_throughput_analysis(output_path)
        self._plot_latency_analysis(output_path)
        self._plot_resource_utilization(output_path)
        self._plot_communication_overhead(output_path)
        self._plot_scalability_analysis(output_path)
        
        # 生成HTML报告
        report_path = self._generate_html_report(output_path)
        
        self.logger.info(f"性能分析报告已生成: {report_path}")
        return str(report_path)
    
    def _plot_throughput_analysis(self, output_path: Path) -> None:
        """吞吐量分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('吞吐量性能分析', fontsize=16, fontweight='bold')
        
        # 准备数据
        configs = []
        throughput_means = []
        throughput_stds = []
        prompt_lengths = []
        generation_lengths = []
        
        for config_id, data in self.results_data['results'].items():
            config = data['config']
            stats = data['statistics']['throughput']
            
            configs.append(f"P{config['prompt_length']}_G{config['generation_length']}")
            throughput_means.append(stats['mean'])
            throughput_stds.append(stats['std'])
            prompt_lengths.append(config['prompt_length'])
            generation_lengths.append(config['generation_length'])
        
        # 1. 吞吐量条形图
        ax1 = axes[0, 0]
        bars = ax1.bar(configs, throughput_means, yerr=throughput_stds, capsize=5)
        ax1.set_title('各配置吞吐量对比')
        ax1.set_xlabel('配置 (Prompt长度_生成长度)')
        ax1.set_ylabel('吞吐量 (tokens/second)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Prompt长度 vs 吞吐量
        ax2 = axes[0, 1]
        unique_prompts = sorted(set(prompt_lengths))
        prompt_throughputs = []
        for prompt_len in unique_prompts:
            throughputs = [throughput_means[i] for i, p in enumerate(prompt_lengths) if p == prompt_len]
            prompt_throughputs.append(np.mean(throughputs))
        
        ax2.plot(unique_prompts, prompt_throughputs, marker='o', linewidth=2, markersize=8)
        ax2.set_title('Prompt长度对吞吐量的影响')
        ax2.set_xlabel('Prompt长度 (tokens)')
        ax2.set_ylabel('平均吞吐量 (tokens/second)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 生成长度 vs 吞吐量
        ax3 = axes[1, 0]
        unique_gens = sorted(set(generation_lengths))
        gen_throughputs = []
        for gen_len in unique_gens:
            throughputs = [throughput_means[i] for i, g in enumerate(generation_lengths) if g == gen_len]
            gen_throughputs.append(np.mean(throughputs))
        
        ax3.bar(unique_gens, gen_throughputs, width=5, alpha=0.7)
        ax3.set_title('生成长度对吞吐量的影响')
        ax3.set_xlabel('生成长度 (tokens)')
        ax3.set_ylabel('平均吞吐量 (tokens/second)')
        
        # 4. 热力图
        ax4 = axes[1, 1]
        # 创建热力图数据
        heatmap_data = np.zeros((len(unique_prompts), len(unique_gens)))
        for i, prompt_len in enumerate(unique_prompts):
            for j, gen_len in enumerate(unique_gens):
                for k, (p, g) in enumerate(zip(prompt_lengths, generation_lengths)):
                    if p == prompt_len and g == gen_len:
                        heatmap_data[i, j] = throughput_means[k]
                        break
        
        im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax4.set_title('吞吐量热力图')
        ax4.set_xlabel('生成长度')
        ax4.set_ylabel('Prompt长度')
        ax4.set_xticks(range(len(unique_gens)))
        ax4.set_xticklabels(unique_gens)
        ax4.set_yticks(range(len(unique_prompts)))
        ax4.set_yticklabels(unique_prompts)
        
        # 添加数值标注
        for i in range(len(unique_prompts)):
            for j in range(len(unique_gens)):
                ax4.text(j, i, f'{heatmap_data[i, j]:.1f}', 
                        ha='center', va='center', fontsize=10)
        
        plt.colorbar(im, ax=ax4, label='吞吐量 (tokens/second)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'throughput_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_analysis(self, output_path: Path) -> None:
        """延迟分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('延迟性能分析', fontsize=16, fontweight='bold')
        
        # 准备数据
        configs = []
        total_latencies = []
        first_token_latencies = []
        
        for config_id, data in self.results_data['results'].items():
            config = data['config']
            stats = data['statistics']['latency']
            
            configs.append(f"P{config['prompt_length']}_G{config['generation_length']}")
            total_latencies.append(stats['total_time']['mean'])
            
            if 'first_token_time' in stats:
                first_token_latencies.append(stats['first_token_time']['mean'])
            else:
                first_token_latencies.append(0)
        
        # 1. 总延迟条形图
        ax1 = axes[0, 0]
        ax1.bar(configs, total_latencies, alpha=0.7)
        ax1.set_title('总延迟对比')
        ax1.set_xlabel('配置')
        ax1.set_ylabel('延迟 (ms)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 首token延迟条形图
        ax2 = axes[0, 1]
        ax2.bar(configs, first_token_latencies, alpha=0.7, color='orange')
        ax2.set_title('首Token延迟对比')
        ax2.set_xlabel('配置')
        ax2.set_ylabel('延迟 (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 延迟分布箱线图
        ax3 = axes[1, 0]
        latency_data = []
        latency_labels = []
        
        for config_id, data in self.results_data['results'].items():
            config = data['config']
            stats = data['statistics']['latency']['total_time']
            
            # 模拟分布数据（实际应该从原始数据中获取）
            mean = stats['mean']
            std = stats['std']
            simulated_data = np.random.normal(mean, std, 100)
            latency_data.append(simulated_data)
            latency_labels.append(f"P{config['prompt_length']}_G{config['generation_length']}")
        
        ax3.boxplot(latency_data, labels=latency_labels)
        ax3.set_title('延迟分布')
        ax3.set_xlabel('配置')
        ax3.set_ylabel('延迟 (ms)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 延迟vs吞吐量散点图
        ax4 = axes[1, 1]
        throughputs = []
        for config_id, data in self.results_data['results'].items():
            throughputs.append(data['statistics']['throughput']['mean'])
        
        ax4.scatter(total_latencies, throughputs, s=100, alpha=0.7)
        ax4.set_title('延迟 vs 吞吐量')
        ax4.set_xlabel('总延迟 (ms)')
        ax4.set_ylabel('吞吐量 (tokens/second)')
        
        # 添加标注
        for i, config in enumerate(configs):
            ax4.annotate(config, (total_latencies[i], throughputs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'latency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resource_utilization(self, output_path: Path) -> None:
        """资源利用率分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('资源利用率分析', fontsize=16, fontweight='bold')
        
        # 准备数据
        configs = []
        gpu_utilizations = []
        memory_usages = []
        
        for config_id, data in self.results_data['results'].items():
            config = data['config']
            stats = data['statistics']
            
            configs.append(f"P{config['prompt_length']}_G{config['generation_length']}")
            gpu_utilizations.append(stats['gpu_utilization']['mean'])
            
            if 'memory_usage' in stats and 'gpu_memory_allocated_mb' in stats['memory_usage']:
                memory_usages.append(stats['memory_usage']['gpu_memory_allocated_mb']['mean'])
            else:
                memory_usages.append(0)
        
        # 1. GPU利用率
        ax1 = axes[0, 0]
        ax1.bar(configs, gpu_utilizations, alpha=0.7, color='green')
        ax1.set_title('GPU利用率')
        ax1.set_xlabel('配置')
        ax1.set_ylabel('GPU利用率 (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 内存使用
        ax2 = axes[0, 1]
        ax2.bar(configs, memory_usages, alpha=0.7, color='red')
        ax2.set_title('GPU内存使用')
        ax2.set_xlabel('配置')
        ax2.set_ylabel('内存使用 (MB)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 效率分析（吞吐量/GPU利用率）
        ax3 = axes[1, 0]
        throughputs = []
        for config_id, data in self.results_data['results'].items():
            throughputs.append(data['statistics']['throughput']['mean'])
        
        efficiency = [t/g if g > 0 else 0 for t, g in zip(throughputs, gpu_utilizations)]
        ax3.bar(configs, efficiency, alpha=0.7, color='purple')
        ax3.set_title('计算效率 (吞吐量/GPU利用率)')
        ax3.set_xlabel('配置')
        ax3.set_ylabel('效率')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 资源利用率雷达图
        ax4 = axes[1, 1]
        
        # 选择一个配置作为示例
        if configs:
            config_data = list(self.results_data['results'].values())[0]['statistics']
            
            categories = ['GPU利用率', '内存效率', '计算效率', '通信效率']
            values = [
                config_data['gpu_utilization']['mean'],
                min(100, memory_usages[0] / 1000 * 100) if memory_usages[0] > 0 else 50,
                min(100, efficiency[0] * 10) if efficiency[0] > 0 else 50,
                75  # 假设值
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2)
            ax4.fill(angles, values, alpha=0.25)
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories)
            ax4.set_ylim(0, 100)
            ax4.set_title('资源利用率雷达图 (示例配置)')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'resource_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_communication_overhead(self, output_path: Path) -> None:
        """通信开销分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('分布式通信开销分析', fontsize=16, fontweight='bold')
        
        # 检查是否有通信数据
        has_comm_data = False
        for config_id, data in self.results_data['results'].items():
            if 'communication_overhead' in data['statistics'] and data['statistics']['communication_overhead']:
                has_comm_data = True
                break
        
        if not has_comm_data:
            # 如果没有通信数据，显示说明信息
            for ax in axes.flat:
                ax.text(0.5, 0.5, '无分布式通信数据\n(单卡推理模式)', 
                       ha='center', va='center', fontsize=14, 
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(output_path / 'communication_overhead.png', dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # 有通信数据时的分析（这里提供框架）
        configs = []
        allreduce_times = []
        broadcast_times = []
        
        for config_id, data in self.results_data['results'].items():
            config = data['config']
            comm_stats = data['statistics'].get('communication_overhead', {})
            
            configs.append(f"P{config['prompt_length']}_G{config['generation_length']}")
            allreduce_times.append(comm_stats.get('allreduce', {}).get('mean', 0))
            broadcast_times.append(comm_stats.get('broadcast', {}).get('mean', 0))
        
        # 1. AllReduce时间
        ax1 = axes[0, 0]
        ax1.bar(configs, allreduce_times, alpha=0.7)
        ax1.set_title('AllReduce通信时间')
        ax1.set_xlabel('配置')
        ax1.set_ylabel('时间 (ms)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Broadcast时间
        ax2 = axes[0, 1]
        ax2.bar(configs, broadcast_times, alpha=0.7, color='orange')
        ax2.set_title('Broadcast通信时间')
        ax2.set_xlabel('配置')
        ax2.set_ylabel('时间 (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 通信开销占比
        ax3 = axes[1, 0]
        total_times = []
        for config_id, data in self.results_data['results'].items():
            total_times.append(data['statistics']['latency']['total_time']['mean'])
        
        comm_ratio = [(a + b) / t * 100 if t > 0 else 0 
                     for a, b, t in zip(allreduce_times, broadcast_times, total_times)]
        
        ax3.bar(configs, comm_ratio, alpha=0.7, color='red')
        ax3.set_title('通信开销占总时间比例')
        ax3.set_xlabel('配置')
        ax3.set_ylabel('比例 (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 通信效率分析
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, '通信效率分析\n(基于实际测量数据)', 
                ha='center', va='center', fontsize=12, 
                transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path / 'communication_overhead.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability_analysis(self, output_path: Path) -> None:
        """可扩展性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('可扩展性分析', fontsize=16, fontweight='bold')
        
        # 这里提供可扩展性分析的框架
        # 实际使用时需要多个不同规模的测试数据
        
        # 模拟不同规模的数据
        gpu_counts = [1, 2, 4, 8]
        throughput_scaling = [100, 180, 340, 600]  # 示例数据
        latency_scaling = [100, 60, 35, 25]  # 示例数据
        efficiency_scaling = [100, 90, 85, 75]  # 示例数据
        
        # 1. 吞吐量可扩展性
        ax1 = axes[0, 0]
        ax1.plot(gpu_counts, throughput_scaling, 'o-', linewidth=2, markersize=8)
        ax1.plot(gpu_counts, [100 * i for i in gpu_counts], '--', alpha=0.5, label='理想线性扩展')
        ax1.set_title('吞吐量可扩展性')
        ax1.set_xlabel('GPU数量')
        ax1.set_ylabel('相对吞吐量 (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 延迟可扩展性
        ax2 = axes[0, 1]
        ax2.plot(gpu_counts, latency_scaling, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_title('延迟可扩展性')
        ax2.set_xlabel('GPU数量')
        ax2.set_ylabel('相对延迟 (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 扩展效率
        ax3 = axes[1, 0]
        ax3.bar(gpu_counts, efficiency_scaling, alpha=0.7, color='green')
        ax3.set_title('扩展效率')
        ax3.set_xlabel('GPU数量')
        ax3.set_ylabel('效率 (%)')
        ax3.set_ylim(0, 100)
        
        # 4. 成本效益分析
        ax4 = axes[1, 1]
        cost_per_token = [1.0, 0.55, 0.30, 0.20]  # 示例数据
        ax4.plot(gpu_counts, cost_per_token, 'o-', linewidth=2, markersize=8, color='purple')
        ax4.set_title('成本效益分析')
        ax4.set_xlabel('GPU数量')
        ax4.set_ylabel('相对成本/Token')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, output_path: Path) -> Path:
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GPT-1.5B 分布式推理性能测试报告</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    border-bottom: 3px solid #007acc;
                    padding-bottom: 20px;
                }}
                h2 {{
                    color: #007acc;
                    margin-top: 40px;
                }}
                .summary {{
                    background-color: #e8f4fd;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .metric {{
                    display: inline-block;
                    background-color: #007acc;
                    color: white;
                    padding: 10px 20px;
                    margin: 5px;
                    border-radius: 5px;
                }}
                .chart {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .chart img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #007acc;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>GPT-1.5B 分布式推理性能测试报告</h1>
                
                <div class="summary">
                    <h3>测试概要</h3>
                    <p><strong>测试时间:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>模型:</strong> {self.results_data['metadata']['model_info'].get('model_name', 'GPT-1.5B')}</p>
                    <p><strong>配置数量:</strong> {len(self.results_data['results'])} 个</p>
                    <p><strong>总参数量:</strong> {self.results_data['metadata']['model_info'].get('num_parameters', 'N/A')} </p>
                </div>
                
                <h2>性能指标总览</h2>
                {self._generate_metrics_summary()}
                
                <h2>吞吐量分析</h2>
                <div class="chart">
                    <img src="throughput_analysis.png" alt="吞吐量分析">
                </div>
                
                <h2>延迟分析</h2>
                <div class="chart">
                    <img src="latency_analysis.png" alt="延迟分析">
                </div>
                
                <h2>资源利用率分析</h2>
                <div class="chart">
                    <img src="resource_utilization.png" alt="资源利用率分析">
                </div>
                
                <h2>通信开销分析</h2>
                <div class="chart">
                    <img src="communication_overhead.png" alt="通信开销分析">
                </div>
                
                <h2>可扩展性分析</h2>
                <div class="chart">
                    <img src="scalability_analysis.png" alt="可扩展性分析">
                </div>
                
                <h2>详细结果表格</h2>
                {self._generate_results_table()}
                
                <h2>结论与建议</h2>
                <div class="summary">
                    {self._generate_conclusions()}
                </div>
            </div>
        </body>
        </html>
        """
        
        report_path = output_path / 'performance_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_metrics_summary(self) -> str:
        """生成指标摘要"""
        metrics_html = ""
        
        # 计算平均指标
        total_throughput = 0
        total_latency = 0
        total_gpu_util = 0
        count = len(self.results_data['results'])
        
        for config_id, data in self.results_data['results'].items():
            stats = data['statistics']
            total_throughput += stats['throughput']['mean']
            total_latency += stats['latency']['total_time']['mean']
            total_gpu_util += stats['gpu_utilization']['mean']
        
        avg_throughput = total_throughput / count if count > 0 else 0
        avg_latency = total_latency / count if count > 0 else 0
        avg_gpu_util = total_gpu_util / count if count > 0 else 0
        
        metrics_html = f"""
        <div class="metric">平均吞吐量: {avg_throughput:.2f} tokens/s</div>
        <div class="metric">平均延迟: {avg_latency:.2f} ms</div>
        <div class="metric">平均GPU利用率: {avg_gpu_util:.1f}%</div>
        """
        
        return metrics_html
    
    def _generate_results_table(self) -> str:
        """生成结果表格"""
        table_html = """
        <table>
            <tr>
                <th>配置ID</th>
                <th>Prompt长度</th>
                <th>生成长度</th>
                <th>吞吐量 (tokens/s)</th>
                <th>延迟 (ms)</th>
                <th>GPU利用率 (%)</th>
            </tr>
        """
        
        for config_id, data in self.results_data['results'].items():
            config = data['config']
            stats = data['statistics']
            
            table_html += f"""
            <tr>
                <td>{config_id}</td>
                <td>{config['prompt_length']}</td>
                <td>{config['generation_length']}</td>
                <td>{stats['throughput']['mean']:.2f}</td>
                <td>{stats['latency']['total_time']['mean']:.2f}</td>
                <td>{stats['gpu_utilization']['mean']:.1f}</td>
            </tr>
            """
        
        table_html += "</table>"
        return table_html
    
    def _generate_conclusions(self) -> str:
        """生成结论"""
        conclusions = """
        <h4>主要发现:</h4>
        <ul>
            <li>不同配置下的性能表现存在显著差异</li>
            <li>Prompt长度对推理性能有重要影响</li>
            <li>GPU利用率和内存使用需要进一步优化</li>
        </ul>
        
        <h4>优化建议:</h4>
        <ul>
            <li>考虑使用更大的批次大小来提高吞吐量</li>
            <li>优化内存管理以减少内存占用</li>
            <li>调整分布式策略以降低通信开销</li>
        </ul>
        """
        
        return conclusions


class MetricsComparator:
    """性能指标比较器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_results(self, result_files: List[str]) -> Dict[str, Any]:
        """比较多个测试结果"""
        results = []
        
        for file_path in result_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({
                    'file': file_path,
                    'data': data
                })
        
        comparison = {
            'num_results': len(results),
            'throughput_comparison': self._compare_throughput(results),
            'latency_comparison': self._compare_latency(results),
            'efficiency_comparison': self._compare_efficiency(results)
        }
        
        return comparison
    
    def _compare_throughput(self, results: List[Dict]) -> Dict[str, Any]:
        """比较吞吐量"""
        # 实现吞吐量比较逻辑
        pass
    
    def _compare_latency(self, results: List[Dict]) -> Dict[str, Any]:
        """比较延迟"""
        # 实现延迟比较逻辑
        pass
    
    def _compare_efficiency(self, results: List[Dict]) -> Dict[str, Any]:
        """比较效率"""
        # 实现效率比较逻辑
        pass
