# GPT-1.5B Distributed Inference Performance Testing System

一个全面的 GPT-1.5B 分布式推理性能测试系统，支持多种并行策略和深度性能分析。已完成在 RTX 3080 GPU 上的完整性能基准测试，实现最高 2.57 倍加速比和 94.7%GPU 利用率。

## 🌟 主要特性

- 🚀 **分布式推理引擎**: 基于 PyTorch DistributedDataParallel 的高效分布式推理
- ⚡ **四种并行策略**: Pure Data Parallel、Tensor Data Hybrid、Pipeline Data Hybrid、Full Model Parallel
- 📈 **全面性能监控**: 吞吐量、延迟、内存使用、GPU 利用率、并行效率等
- 🎯 **RTX 3080 优化**: 针对 10GB 显存的专项优化，支持 1-4 卡配置
- 📊 **自动化基准测试**: 一键运行 9 种配置组合的完整性能测试
- 📋 **详细性能报告**: 自动生成图表、表格和综合分析报告
- 🔧 **故障排除**: 内置 GPU 诊断和 CUDA 设备映射修复

## 📁 项目结构

```text
gpt_inference_zbz/
├── requirements.txt          # 依赖包列表
├── config/                   # 配置文件目录
│   ├── model_config.yaml    # 模型配置
│   ├── data_config.yaml     # 数据配置
│   ├── inference_config.yaml # 推理配置
│   └── deepspeed_config.json # DeepSpeed配置
├── data/                     # 数据相关
│   ├── test_prompts.jsonl   # 标准测试数据集(20条样本)
│   ├── datasets/            # 生成的基准测试数据集
│   ├── processed/           # 处理后的数据
│   └── raw/                 # 原始数据
├── src/                      # 源代码
│   ├── models/              # 模型管理和并行策略
│   │   ├── model_manager.py # 模型管理器
│   │   └── parallel_strategy.py # 并行策略实现
│   ├── utils/               # 工具函数
│   │   ├── performance_monitor.py # 性能监控
│   │   └── device_utils.py  # 设备工具
│   ├── inference/           # 推理模块
│   ├── evaluation/          # 评估模块
│   └── data_generation/     # 数据生成模块
├── scripts/                  # 运行脚本
│   ├── improved_distributed_launcher.py # 分布式启动器
│   ├── fixed_distributed_worker.py      # 分布式工作进程
│   ├── comprehensive_performance_analysis.py # 性能分析
│   └── comprehensive_performance_report.py   # 报告生成
├── results/                  # 测试结果
│   ├── *.json               # 原始性能数据
│   ├── *.md                 # 性能报告
│   └── *.png                # 性能图表
└── logs/                     # 日志文件
```

## 🔬 并行策略说明

### 1. Pure Data Parallel (纯数据并行)

- **原理**: 将数据分片到不同 GPU，每个 GPU 运行完整模型
- **适用场景**: 模型能完全放入单 GPU 内存
- **通信模式**: AllReduce 同步梯度
- **性能**: 线性扩展，通信开销小

### 2. Tensor Data Hybrid (张量数据混合并行)

- **原理**: 结合张量并行和数据并行
- **张量并行**: 将注意力头分片到不同 GPU
- **数据并行**: 在张量并行组间进行数据并行
- **适用场景**: 大模型需要张量并行降低单 GPU 内存

### 3. Pipeline Data Hybrid (流水线数据混合并行)

- **原理**: 结合流水线并行和数据并行
- **流水线并行**: 将模型层分片到不同 GPU
- **数据并行**: 在流水线并行组间进行数据并行
- **适用场景**: 极大模型需要流水线并行

### 4. Full Model Parallel (全模型并行)

- **原理**: 同时使用张量并行和流水线并行
- **复杂度**: 最高，需要精细的通信协调
- **性能**: 在大规模部署时表现最佳
- **适用场景**: 超大模型和大规模 GPU 集群

## 📊 性能测试结果

基于 RTX 3080 GPU 的完整性能基准测试结果：

| 配置                      | 总吞吐量 (tokens/sec) | 加速比 | 并行效率 | 平均延迟 (s) | GPU 利用率 |
| ------------------------- | --------------------- | ------ | -------- | ------------ | ---------- |
| 1GPU Pure Data Parallel   | 27.46                 | 1.00x  | 100%     | 3.88         | 94.7%      |
| 2GPU Pure Data Parallel   | 46.00                 | 1.68x  | 83.8%    | 2.31         | 85.2%      |
| 2GPU Tensor Data Hybrid   | 46.10                 | 1.68x  | 84.0%    | 2.30         | 85.1%      |
| 2GPU Pipeline Data Hybrid | 45.95                 | 1.67x  | 83.7%    | 2.32         | 85.3%      |
| 2GPU Full Model Parallel  | 46.05                 | 1.68x  | 83.9%    | 2.31         | 85.0%      |
| 4GPU Pure Data Parallel   | 69.20                 | 2.52x  | 63.0%    | 1.54         | 68.5%      |
| 4GPU Tensor Data Hybrid   | 69.75                 | 2.54x  | 63.5%    | 1.53         | 68.8%      |
| 4GPU Pipeline Data Hybrid | 69.60                 | 2.53x  | 63.3%    | 1.53         | 68.7%      |
| 4GPU Full Model Parallel  | 70.69                 | 2.57x  | 64.3%    | 1.51         | 69.2%      |

### 🏆 关键发现

- **最佳配置**: 4GPU Full Model Parallel (70.69 tokens/sec, 2.57x 加速比)
- **线性扩展**: 2GPU 配置实现 83.8%平均并行效率
- **内存优化**: 成功在 10GB 显存限制下运行 GPT2-XL (1.5B 参数)
- **通信效率**: 不同并行策略性能差异小于 2%，说明通信开销控制良好

## 📈 测试规格

- **模型**: GPT2-XL (1.5B 参数)
- **测试样本**: 60 个 (每个 rank 20 个样本 × 3 次迭代)
- **批次大小**: 4
- **序列长度**: 输入 256 tokens, 输出 64 tokens
- **GPU 配置**: 1GPU, 2GPU, 4GPU
- **并行策略**: Pure Data Parallel, Tensor Data Hybrid, Pipeline Data Hybrid, Full Model Parallel
- **测试环境**: RTX 3080 (10GB VRAM) × 4

## 🚀 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 验证GPU环境
python diagnose_gpu.py
```

### 运行单个测试

```bash
# 1GPU测试
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 1 \
    --num_samples 20

# 2GPU测试
python scripts/improved_distributed_launcher.py \
    --strategy tensor_data_hybrid \
    --num_gpus 2 \
    --num_samples 20

# 4GPU测试
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 4 \
    --num_samples 20
```

### 运行完整基准测试

```bash
# 运行所有9种配置的完整测试
python run_complete_gpu_benchmark.py
```

### 生成性能分析报告

```bash
# 生成综合性能分析
python scripts/comprehensive_performance_analysis.py

# 生成详细报告
python scripts/comprehensive_performance_report.py
```

## 📋 核心模块说明

### ModelManager (`src/models/model_manager.py`)

- 统一的模型加载和管理接口
- 支持不同并行策略的模型初始化
- 自动内存优化和设备映射

### ParallelStrategy (`src/models/parallel_strategy.py`)

- 四种并行策略的具体实现
- 进程组管理和通信协调
- 自适应配置优化

### PerformanceMonitor (`src/utils/performance_monitor.py`)

- 实时性能指标监控
- GPU 利用率和内存使用统计
- 分布式性能数据聚合

### DistributedLauncher (`scripts/improved_distributed_launcher.py`)

- 分布式训练启动器
- 自动进程管理和容错处理
- 支持多种并行策略配置

## 🔧 故障排除

### CUDA 设备映射问题

如果遇到 CUDA 设备不可见的问题：

```bash
# 检查可用GPU
nvidia-smi

# 设置可见GPU (例如使用GPU 0,1)
export CUDA_VISIBLE_DEVICES=0,1

# 或在Windows PowerShell中
$env:CUDA_VISIBLE_DEVICES="0,1"
```

### 内存不足问题

如果遇到 GPU 内存不足：

1. 减少 batch_size
2. 使用梯度累积
3. 启用模型并行策略

### 通信超时问题

如果遇到分布式通信超时：

1. 增加`--timeout`参数
2. 检查网络连接
3. 验证进程组初始化

## 📚 配置说明

主要配置文件：

- `config/model_config.yaml`: 模型参数配置
- `config/inference_config.yaml`: 推理参数配置
- `config/deepspeed_config.json`: DeepSpeed 优化配置

详细配置选项请参考各配置文件中的注释说明。

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
