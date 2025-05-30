# GPT-1.5B Distributed Inference Performance Testing System

一个全面的 GPT-1.5B 分布式推理性能测试系统，支持多种并行策略和性能评估指标。本系统专为 RTX 3080 GPU 优化，支持 4 卡分布式推理。

## 🌟 主要特性

- 📊 **多数据源测试集**: 支持 WikiText-103、Pile 子集和合成随机 tokens
- 🚀 **分布式推理引擎**: 基于 DeepSpeed 的高效分布式推理
- ⚡ **多种并行策略**: 数据并行、张量并行、流水线并行等
- 📈 **全面性能监控**: 吞吐量、延迟、内存使用、通信开销等
- 🎯 **RTX 3080 优化**: 针对 10GB 显存的专项优化
- 📋 **自动化基准测试**: 一键比较不同并行策略性能

## 项目结构

```
gpt_inference_zbz/
├── requirements.txt          # 依赖包列表
├── config/                   # 配置文件目录
│   ├── model_config.yaml    # 模型配置
│   ├── data_config.yaml     # 数据配置
│   └── inference_config.yaml # 推理配置
├── data/                     # 数据相关
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后的数据
│   └── datasets/            # 生成的测试数据集
├── src/                      # 源代码
│   ├── data_generation/     # 数据生成模块
│   ├── models/              # 模型相关
│   ├── inference/           # 推理模块
│   ├── evaluation/          # 评估模块
│   └── utils/               # 工具函数
├── scripts/                  # 运行脚本
├── results/                  # 测试结果
└── logs/                     # 日志文件
```

## 数据集规格

- **语料**: WikiText-103 + Pile 子集 + Synthetic 随机 token
- **Prompt 长度**: 32, 128, 512, 1024 tokens
- **生成长度**: 32 或 64 tokens
- **样本数**: 每种配置 500 条 (总计 16,000 条样本)
- **格式**: JSONL

## 评估指标

1. **吞吐量** (Throughput): tokens/second
2. **延迟** (Latency): 首 token 延迟和总延迟
3. **计算效率** (Compute Efficiency): GPU 利用率、内存使用率
4. **通信开销** (Communication Overhead): 分布式通信时间

## 快速开始

1. 安装依赖:

```bash
pip install -r requirements.txt
```

2. 生成测试数据集:

```bash
python scripts/generate_dataset.py
```

3. 运行推理测试:

```bash
python scripts/run_inference_test.py
```

4. 分析结果:

```bash
python scripts/analyze_results.py
```

## 配置说明

详细配置请参考 `config/` 目录下的 YAML 文件。
