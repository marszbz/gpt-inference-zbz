#!/usr/bin/env python3
"""
GPU环境诊断脚本
帮助诊断CUDA和GPU相关问题
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def check_system_info():
    """检查系统信息"""
    print("=== 系统信息 ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"Python版本: {sys.version}")
    print()

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("=== 检查NVIDIA驱动 ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ NVIDIA驱动已安装")
            print("NVIDIA-SMI输出:")
            print(result.stdout)
        else:
            print("✗ nvidia-smi命令执行失败")
            print(f"错误输出: {result.stderr}")
    except FileNotFoundError:
        print("✗ nvidia-smi命令未找到")
        print("  可能原因: NVIDIA驱动未安装或未添加到PATH")
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi命令超时")
    except Exception as e:
        print(f"✗ 检查NVIDIA驱动时出错: {e}")
    print()

def check_cuda_installation():
    """检查CUDA安装"""
    print("=== 检查CUDA安装 ===")
    
    # 检查CUDA_PATH环境变量
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"✓ CUDA_PATH环境变量: {cuda_path}")
    else:
        print("✗ CUDA_PATH环境变量未设置")
    
    # 检查PATH中的CUDA
    path_dirs = os.environ.get('PATH', '').split(';')
    cuda_in_path = any('cuda' in dir.lower() for dir in path_dirs)
    if cuda_in_path:
        print("✓ PATH中包含CUDA目录")
    else:
        print("✗ PATH中未找到CUDA目录")
    
    # 尝试运行nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ CUDA编译器(nvcc)可用")
            print(result.stdout.strip())
        else:
            print("✗ nvcc命令执行失败")
    except FileNotFoundError:
        print("✗ nvcc命令未找到")
        print("  可能原因: CUDA Toolkit未安装或未添加到PATH")
    except Exception as e:
        print(f"✗ 检查nvcc时出错: {e}")
    print()

def check_pytorch_installation():
    """检查PyTorch安装"""
    print("=== 检查PyTorch安装 ===")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"  CUDA支持: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  CUDA版本: 不可用")
            print("  原因分析:")
            
            # 检查PyTorch是否为CPU版本
            if 'cpu' in torch.__file__:
                print("    - 安装的是CPU版本的PyTorch")
            
            # 检查CUDA版本兼容性
            try:
                cuda_version = torch.version.cuda
                if cuda_version is None:
                    print("    - PyTorch编译时未包含CUDA支持")
                else:
                    print(f"    - PyTorch CUDA版本: {cuda_version}")
            except:
                print("    - 无法获取PyTorch CUDA版本信息")
    
    except ImportError as e:
        print(f"✗ PyTorch未安装: {e}")
    except Exception as e:
        print(f"✗ 检查PyTorch时出错: {e}")
    print()

def check_cuda_libraries():
    """检查CUDA库文件"""
    print("=== 检查CUDA库文件 ===")
    
    # 常见的CUDA库路径
    possible_cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\CUDA",
        os.environ.get('CUDA_PATH', '')
    ]
    
    cuda_found = False
    for cuda_path in possible_cuda_paths:
        if cuda_path and Path(cuda_path).exists():
            print(f"✓ 找到CUDA安装目录: {cuda_path}")
            
            # 检查版本目录
            versions = []
            cuda_path_obj = Path(cuda_path)
            if cuda_path_obj.exists():
                for item in cuda_path_obj.iterdir():
                    if item.is_dir() and item.name.startswith('v'):
                        versions.append(item.name)
            
            if versions:
                print(f"  可用版本: {', '.join(sorted(versions))}")
                
                # 检查最新版本的bin目录
                latest_version = sorted(versions)[-1]
                bin_path = cuda_path_obj / latest_version / "bin"
                if bin_path.exists():
                    print(f"  ✓ {latest_version}/bin 目录存在")
                    # 检查关键文件
                    key_files = ['nvcc.exe', 'cudart64_*.dll']
                    for pattern in key_files:
                        files = list(bin_path.glob(pattern))
                        if files:
                            print(f"    ✓ 找到 {pattern}: {[f.name for f in files]}")
                        else:
                            print(f"    ✗ 未找到 {pattern}")
            
            cuda_found = True
            break
    
    if not cuda_found:
        print("✗ 未找到CUDA安装目录")
    print()

def provide_solutions():
    """提供解决方案"""
    print("=== 解决方案建议 ===")
    
    print("如果CUDA不可用，请按以下步骤操作：")
    print()
    
    print("1. 检查硬件兼容性:")
    print("   - 确保您有NVIDIA GPU（不支持AMD GPU）")
    print("   - GPU必须支持CUDA（计算能力3.5及以上）")
    print()
    
    print("2. 安装NVIDIA驱动:")
    print("   - 下载最新的NVIDIA驱动: https://www.nvidia.com/drivers/")
    print("   - 安装后重启电脑")
    print()
    
    print("3. 安装CUDA Toolkit:")
    print("   - 下载CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("   - 建议安装CUDA 11.8或12.1版本")
    print("   - 安装时选择自定义安装，确保包含所有组件")
    print()
    
    print("4. 安装支持CUDA的PyTorch:")
    print("   - 卸载当前PyTorch: pip uninstall torch torchvision torchaudio")
    print("   - 访问: https://pytorch.org/get-started/locally/")
    print("   - 选择适合您CUDA版本的安装命令")
    print("   - 例如: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    
    print("5. 环境变量配置:")
    print("   - 添加CUDA_PATH环境变量")
    print("   - 添加CUDA\\bin和CUDA\\libnvvp到PATH")
    print()
    
    print("6. 验证安装:")
    print("   - 重新运行此诊断脚本")
    print("   - 或运行: python -c \"import torch; print(torch.cuda.is_available())\"")

def main():
    print("GPU环境诊断工具")
    print("="*50)
    
    check_system_info()
    check_nvidia_driver()
    check_cuda_installation()
    check_cuda_libraries()
    check_pytorch_installation()
    provide_solutions()

if __name__ == "__main__":
    main()
