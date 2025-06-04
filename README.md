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

## 🚀 详细使用指南

### 步骤 1: 环境准备

#### 1.1 系统要求

- **操作系统**: Windows 10/11, Linux Ubuntu 18.04+
- **Python**: 3.8 - 3.11 (推荐 3.9)
- **GPU**: NVIDIA RTX 系列 (RTX 3080/4080/4090 等)，显存 ≥ 8GB
- **CUDA**: 11.8+ 或 12.0+
- **磁盘空间**: ≥ 10GB (用于模型、数据和结果)

#### 1.2 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd gpt_inference_zbz

# 创建虚拟环境 (推荐)
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# 安装依赖包
pip install -r requirements.txt

# 验证PyTorch CUDA支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

#### 1.3 GPU 环境检查

```bash
# 检查GPU状态和内存
nvidia-smi

# 运行GPU诊断工具
python diagnose_gpu.py

# 预期输出示例:
# ✓ 检测到 4 个 GPU 设备
# ✓ GPU 0: NVIDIA GeForce RTX 3080 (10.0GB)
# ✓ PyTorch CUDA 支持: 是
# ✓ 所有 GPU 可正常访问
```

### 步骤 2: 准备测试数据

#### 2.1 使用预置测试数据 (推荐新手)

```bash
# 验证预置测试数据
python validate_dataset.py

# 查看测试数据样本
head -n 3 data/test_prompts.jsonl
```

#### 2.2 生成自定义测试数据

```bash
# 生成不同配置的测试数据集
python scripts/generate_dataset.py --num_samples 50 --output data/custom_test.jsonl

# 生成基准测试用的数据集
python scripts/generate_dataset.py --benchmark --num_configs 6
```

### 步骤 3: 单策略性能测试

#### 3.1 1GPU 测试 (入门测试)

```bash
# 基础单GPU测试
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 1 \
    --num_samples 20 \
    --batch_size 4 \
    --verbose

# 预期输出:
# 正在初始化 1GPU Pure Data Parallel 推理...
# 加载模型: gpt2-xl (1.5B 参数)
# 开始推理测试...
# 性能结果: 27.46 tokens/sec, 延迟: 3.88s, GPU利用率: 94.7%
```

#### 3.2 2GPU 分布式测试

```bash
# 纯数据并行
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 2 \
    --num_samples 20

# 张量数据混合并行
python scripts/improved_distributed_launcher.py \
    --strategy tensor_data_hybrid \
    --num_gpus 2 \
    --num_samples 20

# 流水线数据混合并行
python scripts/improved_distributed_launcher.py \
    --strategy pipeline_data_hybrid \
    --num_gpus 2 \
    --num_samples 20

# 全模型并行
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 2 \
    --num_samples 20
```

#### 3.3 4GPU 高性能测试

```bash
# 4GPU 全模型并行 (最佳性能配置)
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 4 \
    --num_samples 20 \
    --batch_size 4

# 期待结果: ~70 tokens/sec, 2.57x 加速比
```

#### 3.4 高级参数配置

```bash
# 自定义配置测试
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 4 \
    --num_samples 50 \
    --batch_size 8 \
    --max_length 512 \
    --temperature 0.8 \
    --top_p 0.9 \
    --repetition_penalty 1.1 \
    --timeout 1800 \
    --output_dir results/custom_test
```

### 步骤 4: 完整基准测试

#### 4.1 运行完整性能基准

```bash
# 运行所有 9 种配置组合的完整测试
# (1×1GPU + 4×2GPU + 4×4GPU)
python run_complete_gpu_benchmark.py

# 测试过程 (约30-60分钟):
# [1/9] 1GPU Pure Data Parallel...     ✓ 完成
# [2/9] 2GPU Pure Data Parallel...     ✓ 完成
# [3/9] 2GPU Tensor Data Hybrid...     ✓ 完成
# [4/9] 2GPU Pipeline Data Hybrid...   ✓ 完成
# [5/9] 2GPU Full Model Parallel...    ✓ 完成
# [6/9] 4GPU Pure Data Parallel...     ✓ 完成
# [7/9] 4GPU Tensor Data Hybrid...     ✓ 完成
# [8/9] 4GPU Pipeline Data Hybrid...   ✓ 完成
# [9/9] 4GPU Full Model Parallel...    ✓ 完成
```

#### 4.2 监控测试进度

```bash
# 在另一个终端监控GPU使用情况
watch -n 1 nvidia-smi

# 查看实时日志
tail -f logs/distributed_inference.log
```

### 步骤 5: 结果分析

#### 5.1 生成性能分析报告

```bash
# 生成综合性能分析 (图表 + 数据)
python scripts/comprehensive_performance_analysis.py

# 输出文件:
# - analysis_reports/throughput_comparison.png      # 吞吐量对比图
# - analysis_reports/latency_comparison.png         # 延迟对比图
# - analysis_reports/parallel_efficiency.png        # 并行效率图
# - analysis_reports/avg_throughput_per_gpu.png     # 单GPU性能图
# - analysis_reports/performance_summary.csv        # 性能数据表
# - analysis_reports/performance_report.txt         # 详细文本报告
```

#### 5.2 生成详细性能报告

```bash
# 生成 Markdown 格式的综合报告
python scripts/comprehensive_performance_report.py

# 输出: results/comprehensive_performance_report_YYYYMMDD_HHMMSS.md
```

#### 5.3 查看和分析结果

```bash
# 查看性能摘要
cat analysis_reports/performance_report.txt

# 查看详细数据
head -n 20 analysis_reports/performance_summary.csv

# 在浏览器中查看图表
start analysis_reports/throughput_comparison.png        # Windows
open analysis_reports/throughput_comparison.png         # macOS
xdg-open analysis_reports/throughput_comparison.png     # Linux
```

### 步骤 6: 高级使用

#### 6.1 自定义配置文件

编辑配置文件以适应您的需求：

```bash
# 编辑模型配置
notepad config/model_config.yaml        # Windows
vim config/model_config.yaml            # Linux/Mac

# 关键配置项:
# model_name: "gpt2-xl"                  # 模型名称
# max_position_embeddings: 1024         # 最大序列长度
# torch_dtype: "float16"                # 数据类型
# device_map: "auto"                    # 设备映射策略
```

#### 6.2 批量测试脚本

```bash
# 创建批量测试脚本
cat > batch_test.sh << 'EOF'
#!/bin/bash
strategies=("pure_data_parallel" "tensor_data_hybrid" "pipeline_data_hybrid" "full_model_parallel")
gpus=(1 2 4)

for strategy in "${strategies[@]}"; do
    for gpu in "${gpus[@]}"; do
        echo "Testing $strategy with $gpu GPU(s)..."
        python scripts/improved_distributed_launcher.py \
            --strategy $strategy \
            --num_gpus $gpu \
            --num_samples 10 \
            --output_dir results/batch_$strategy_$gpu
    done
done
EOF

chmod +x batch_test.sh
./batch_test.sh
```

#### 6.3 性能调优

```bash
# GPU 内存优化测试
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 4 \
    --num_samples 20 \
    --enable_memory_efficient_attention \
    --gradient_checkpointing \
    --mixed_precision

# 大批次大小测试 (如果显存充足)
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 4 \
    --num_samples 20 \
    --batch_size 16 \
    --max_length 1024
```

### 步骤 7: 常见使用场景

#### 7.1 性能对比测试

```bash
# 对比不同并行策略在2GPU配置下的性能
for strategy in pure_data_parallel tensor_data_hybrid pipeline_data_hybrid full_model_parallel; do
    echo "Testing $strategy..."
    python scripts/improved_distributed_launcher.py \
        --strategy $strategy \
        --num_gpus 2 \
        --num_samples 20 \
        --output_dir results/comparison_$strategy
done

# 分析对比结果
python scripts/comprehensive_performance_analysis.py
```

#### 7.2 扩展性测试

```bash
# 测试相同策略在不同GPU数量下的扩展性
strategy="full_model_parallel"
for gpus in 1 2 4; do
    echo "Testing $strategy with $gpus GPU(s)..."
    python scripts/improved_distributed_launcher.py \
        --strategy $strategy \
        --num_gpus $gpus \
        --num_samples 20 \
        --output_dir results/scaling_$strategy_$gpus
done
```

#### 7.3 生产环境性能评估

```bash
# 模拟生产负载测试
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 4 \
    --num_samples 100 \
    --batch_size 8 \
    --max_length 512 \
    --temperature 0.7 \
    --repetition_penalty 1.05 \
    --output_dir results/production_test \
    --save_detailed_metrics
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

## 🔧 详细故障排除指南

### 常见问题解决

#### 问题 1: CUDA 设备不可见

**症状**:

```
RuntimeError: No CUDA devices available
CUDA_VISIBLE_DEVICES shows no devices
```

**解决方案**:

```bash
# 1. 检查GPU驱动和CUDA安装
nvidia-smi
nvcc --version

# 2. 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"

# 3. 设置可见GPU环境变量
# Windows PowerShell:
$env:CUDA_VISIBLE_DEVICES="0,1,2,3"
# Windows CMD:
set CUDA_VISIBLE_DEVICES=0,1,2,3
# Linux/Mac:
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 4. 重新运行GPU诊断
python diagnose_gpu.py
```

#### 问题 2: GPU 内存不足 (CUDA OOM)

**症状**:

```
RuntimeError: CUDA out of memory. Tried to allocate X.XXGiB
```

**解决方案**:

```bash
# 方案1: 减少批次大小
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 2 \
    --batch_size 2 \
    --num_samples 20

# 方案2: 启用梯度检查点
python scripts/improved_distributed_launcher.py \
    --strategy tensor_data_hybrid \
    --num_gpus 2 \
    --gradient_checkpointing \
    --num_samples 20

# 方案3: 使用混合精度
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 4 \
    --mixed_precision \
    --num_samples 20

# 方案4: 启用内存高效注意力
python scripts/improved_distributed_launcher.py \
    --strategy pipeline_data_hybrid \
    --num_gpus 2 \
    --enable_memory_efficient_attention \
    --num_samples 20
```

#### 问题 3: 分布式通信超时

**症状**:

```
RuntimeError: ProcessGroup timeout
torch.distributed initialization failed
```

**解决方案**:

```bash
# 1. 增加超时时间
python scripts/improved_distributed_launcher.py \
    --strategy tensor_data_hybrid \
    --num_gpus 2 \
    --timeout 3600 \
    --num_samples 20

# 2. 检查网络连接和防火墙
# Windows防火墙设置允许Python.exe
# 或暂时关闭防火墙进行测试

# 3. 使用单机测试模式
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 1 \
    --num_samples 10

# 4. 重置PyTorch分布式后端
export NCCL_DEBUG=INFO  # Linux
$env:NCCL_DEBUG="INFO"  # Windows PowerShell
```

#### 问题 4: 模型加载失败

**症状**:

```
OSError: Can't load config/model from 'gpt2-xl'
ConnectionError: Couldn't reach huggingface.co
```

**解决方案**:

```bash
# 1. 手动下载模型 (首次运行)
python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
print('模型下载完成')
"

# 2. 设置HuggingFace缓存目录
# Windows:
$env:HF_HOME="D:\cache\huggingface"
# Linux:
export HF_HOME="/path/to/huggingface/cache"

# 3. 使用离线模式
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 1 \
    --offline_mode \
    --num_samples 10
```

#### 问题 5: 进程启动失败

**症状**:

```
torch.multiprocessing spawn failed
ProcessGroup initialization error
```

**解决方案**:

```bash
# 1. 清理现有进程
# Windows:
taskkill /f /im python.exe
# Linux:
pkill -f python

# 2. 检查端口占用
netstat -ano | findstr :29500  # Windows
lsof -i :29500                 # Linux

# 3. 使用不同端口
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 2 \
    --master_port 29501 \
    --num_samples 20

# 4. 重启系统以清理资源
```

#### 问题 6: 权限错误

**症状**:

```
PermissionError: Access denied
OSError: Cannot write to results directory
```

**解决方案**:

```bash
# 1. 检查目录权限
# Windows:
icacls results
mkdir results 2>nul

# Linux:
ls -la results/
chmod 755 results/

# 2. 以管理员身份运行 (Windows)
# 右键点击PowerShell -> "以管理员身份运行"

# 3. 更改输出目录
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 1 \
    --output_dir C:\temp\gpu_results \
    --num_samples 10
```

### 调试工具和方法

#### 启用详细日志

```bash
# 启用详细输出
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 2 \
    --verbose \
    --debug \
    --num_samples 10

# 查看详细日志
tail -f logs/distributed_inference.log  # Linux
Get-Content logs\distributed_inference.log -Wait  # Windows PowerShell
```

#### GPU 内存监控

```bash
# 实时监控GPU状态
# Linux:
watch -n 1 nvidia-smi

# Windows PowerShell:
while ($true) { clear; nvidia-smi; Start-Sleep 1 }

# 监控特定GPU
nvidia-smi -i 0 -l 1  # 监控GPU 0

# 查看进程详情
nvidia-smi pmon -i 0 -d 1  # 进程监控
```

#### 性能分析工具

```bash
# 使用PyTorch Profiler
python scripts/improved_distributed_launcher.py \
    --strategy tensor_data_hybrid \
    --num_gpus 2 \
    --enable_profiling \
    --num_samples 5

# 分析Profile结果
python -c "
import torch
profile = torch.load('results/pytorch_profile.pt')
print(profile.key_averages().table(sort_by='cuda_time_total'))
"
```

#### 环境检查脚本

```bash
# 创建完整环境检查脚本
cat > check_environment.py << 'EOF'
import torch
import transformers
import sys
import os

print("=== 系统环境检查 ===")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"Transformers版本: {transformers.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
print("=== 检查完成 ===")
EOF

python check_environment.py
```

### 性能优化建议

#### 针对 RTX 3080 的优化

```bash
# RTX 3080 (10GB) 推荐配置
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 4 \
    --batch_size 4 \
    --max_length 256 \
    --mixed_precision \
    --enable_memory_efficient_attention \
    --num_samples 20
```

#### 针对 RTX 4090 的优化

```bash
# RTX 4090 (24GB) 高性能配置
python scripts/improved_distributed_launcher.py \
    --strategy tensor_data_hybrid \
    --num_gpus 2 \
    --batch_size 8 \
    --max_length 512 \
    --num_samples 50
```

#### 针对多 GPU 服务器的优化

```bash
# 8GPU 大规模部署配置
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 8 \
    --batch_size 16 \
    --max_length 1024 \
    --gradient_checkpointing \
    --num_samples 100
```

## ❓ 常见问题 FAQ

### Q1: 为什么我的 GPU 利用率很低？

**A**: GPU 利用率低可能由以下原因造成：

1. **批次大小太小**: 增加 `--batch_size` 参数
2. **序列长度太短**: 增加 `--max_length` 参数
3. **CPU 瓶颈**: 检查 CPU 使用率，增加数据预处理线程
4. **内存带宽限制**: 使用混合精度 `--mixed_precision`

```bash
# 优化GPU利用率的配置示例
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 2 \
    --batch_size 8 \
    --max_length 512 \
    --mixed_precision \
    --num_samples 20
```

### Q2: 不同并行策略的选择建议是什么？

**A**: 选择策略依据：

- **1-2 GPU**: 推荐 `pure_data_parallel`，简单高效
- **2-4 GPU**: 推荐 `tensor_data_hybrid`，平衡性能和复杂度
- **4+ GPU**: 推荐 `full_model_parallel`，最大化利用多 GPU
- **显存不足**: 推荐 `pipeline_data_hybrid`，最省显存

### Q3: 如何解释性能指标？

**A**: 关键指标说明：

- **总吞吐量** (tokens/sec): 系统整体处理速度，越高越好
- **加速比**: 相对于 1GPU 的性能提升倍数，理想值为 GPU 数量
- **并行效率**: 加速比/GPU 数量，反映资源利用效率，>80%为优秀
- **平均延迟**: 单个请求的处理时间，越低越好
- **GPU 利用率**: GPU 计算资源使用百分比，>90%为优秀

### Q4: 测试结果不稳定怎么办？

**A**: 提高测试稳定性的方法：

```bash
# 1. 增加测试样本数量
python scripts/improved_distributed_launcher.py \
    --strategy tensor_data_hybrid \
    --num_gpus 2 \
    --num_samples 50 \
    --num_iterations 5

# 2. 启用预热 (warm-up)
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 2 \
    --warmup_steps 10 \
    --num_samples 30

# 3. 固定随机种子
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 4 \
    --seed 42 \
    --num_samples 20
```

### Q5: 如何在生产环境中部署？

**A**: 生产部署建议：

1. **选择最佳配置**: 根据基准测试结果选择最优策略
2. **资源监控**: 实施 GPU、内存、网络监控
3. **负载均衡**: 配置请求分发和队列管理
4. **容错处理**: 实现自动重启和故障恢复
5. **性能调优**: 根据实际负载调整批次大小和并发数

### Q6: 支持哪些模型？

**A**: 当前支持的模型：

- **GPT-2 系列**: gpt2, gpt2-medium, gpt2-large, gpt2-xl
- **扩展支持**: 系统设计支持其他 Transformer 模型
- **自定义模型**: 可通过修改配置文件支持自定义模型

修改 `config/model_config.yaml` 以支持其他模型：

```yaml
model_name: "your-custom-model"
model_type: "gpt2" # 或其他支持的类型
torch_dtype: "float16"
device_map: "auto"
```

### Q7: 如何添加新的并行策略？

**A**: 添加自定义并行策略：

1. 在 `src/models/parallel_strategy.py` 中添加新的策略类
2. 实现 `setup_parallel_groups()` 和 `get_model_parallel_config()` 方法
3. 在启动器中注册新策略
4. 进行充分测试验证

### Q8: 测试数据的格式要求是什么？

**A**: 测试数据格式 (JSON Lines):

```json
{"prompt": "请写一篇关于人工智能的文章", "max_length": 100}
{"prompt": "解释一下机器学习的基本概念", "max_length": 150}
{"prompt": "描述深度学习的发展历程", "max_length": 200}
```

必需字段:

- `prompt`: 输入文本
- `max_length`: 生成长度 (可选，默认 64)

## 📋 参数配置详解

### 启动器参数说明

| 参数                                  | 类型  | 默认值             | 说明             |
| ------------------------------------- | ----- | ------------------ | ---------------- |
| `--strategy`                          | str   | pure_data_parallel | 并行策略         |
| `--num_gpus`                          | int   | 1                  | GPU 数量         |
| `--num_samples`                       | int   | 20                 | 测试样本数       |
| `--batch_size`                        | int   | 4                  | 批次大小         |
| `--max_length`                        | int   | 64                 | 最大生成长度     |
| `--temperature`                       | float | 1.0                | 采样温度         |
| `--top_p`                             | float | 1.0                | nucleus 采样阈值 |
| `--repetition_penalty`                | float | 1.0                | 重复惩罚         |
| `--mixed_precision`                   | bool  | False              | 混合精度训练     |
| `--gradient_checkpointing`            | bool  | False              | 梯度检查点       |
| `--enable_memory_efficient_attention` | bool  | False              | 内存高效注意力   |
| `--timeout`                           | int   | 1800               | 超时时间(秒)     |
| `--master_port`                       | int   | 29500              | 主端口           |
| `--output_dir`                        | str   | results            | 输出目录         |
| `--verbose`                           | bool  | False              | 详细输出         |
| `--debug`                             | bool  | False              | 调试模式         |

### 并行策略配置

#### Pure Data Parallel

```bash
python scripts/improved_distributed_launcher.py \
    --strategy pure_data_parallel \
    --num_gpus 4 \
    --batch_size 4 \
    --num_samples 20
```

适用场景: 模型可完全放入单 GPU，需要提高吞吐量

#### Tensor Data Hybrid

```bash
python scripts/improved_distributed_launcher.py \
    --strategy tensor_data_hybrid \
    --num_gpus 4 \
    --tensor_parallel_size 2 \
    --data_parallel_size 2 \
    --num_samples 20
```

适用场景: 模型较大，需要张量并行减少单 GPU 内存

#### Pipeline Data Hybrid

```bash
python scripts/improved_distributed_launcher.py \
    --strategy pipeline_data_hybrid \
    --num_gpus 4 \
    --pipeline_parallel_size 2 \
    --data_parallel_size 2 \
    --num_samples 20
```

适用场景: 模型极大，需要流水线并行

#### Full Model Parallel

```bash
python scripts/improved_distributed_launcher.py \
    --strategy full_model_parallel \
    --num_gpus 4 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 2 \
    --num_samples 20
```

适用场景: 超大模型，需要所有并行技术

## 📚 配置说明

主要配置文件：

- `config/model_config.yaml`: 模型参数配置
- `config/inference_config.yaml`: 推理参数配置
- `config/deepspeed_config.json`: DeepSpeed 优化配置

详细配置选项请参考各配置文件中的注释说明。

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
