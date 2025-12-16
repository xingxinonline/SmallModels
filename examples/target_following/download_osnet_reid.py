"""
下载预训练的 OSNet ReID 模型
Download Pre-trained OSNet Re-ID Model

OSNet 是目前最常用的轻量级 ReID 模型:
- OSNet-x0.25: 最轻量 (~0.98M 参数)
- 在 Market1501 + DukeMTMC + MSMT17 上训练
- 512D embedding
"""

import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def download_with_progress(url: str, save_path: Path, desc: str = ""):
    """带进度条的下载"""
    print(f"\n[下载] {desc}")
    print(f"  URL: {url}")
    print(f"  保存: {save_path}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = '=' * filled + '-' * (bar_len - filled)
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            sys.stdout.write(f'\r  [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, save_path, progress_hook)
        print("\n  ✓ 下载完成!")
        return True
    except Exception as e:
        print(f"\n  ✗ 下载失败: {e}")
        return False


def download_osnet():
    """下载 OSNet-x0.25 预训练模型"""
    
    # 先下载 PyTorch 模型，然后转换为 ONNX
    pth_path = MODELS_DIR / "osnet_x0_25_msmt17.pth"
    onnx_path = MODELS_DIR / "osnet_x0_25.onnx"
    
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        print(f"[跳过] OSNet ONNX 已存在 ({size_mb:.1f} MB): {onnx_path}")
        return True
    
    # 从 torchreid 官方下载预训练权重
    # https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
    urls = [
        # OSNet-x0.25 trained on MSMT17
        "https://drive.google.com/uc?export=download&id=1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer",
    ]
    
    # Google Drive 链接需要特殊处理，改用 Hugging Face
    hf_urls = [
        # 尝试 Hugging Face 上的模型
        "https://huggingface.co/spaces/hysts/insightface-SCRFD/resolve/main/models/osnet_x0_25_msmt17.pt",
    ]
    
    print("\n[注意] OSNet 预训练模型需要从 torchreid 获取")
    print("  方法 1: 使用 torchreid 库自动下载")
    print("    pip install torchreid")
    print("    python -c \"from torchreid import models; models.build_model('osnet_x0_25', pretrained=True)\"")
    print("")
    print("  方法 2: 手动下载")
    print("    1. 访问: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO")
    print("    2. 下载: osnet_x0_25_msmt17.pth")
    print(f"    3. 放到: {pth_path}")
    print("")
    
    return False


def convert_osnet_to_onnx():
    """将 OSNet PyTorch 模型转换为 ONNX"""
    import torch
    import torch.nn as nn
    
    pth_path = MODELS_DIR / "osnet_x0_25_msmt17.pth"
    onnx_path = MODELS_DIR / "osnet_x0_25.onnx"
    
    if not pth_path.exists():
        print(f"[错误] 找不到 PyTorch 模型: {pth_path}")
        return False
    
    # OSNet 架构 (简化版，用于加载权重)
    # 完整版需要 torchreid 库
    print("[提示] 转换需要 torchreid 库")
    print("  pip install torchreid")
    
    return False


def create_color_histogram_reid():
    """
    创建基于颜色直方图的简单 ReID
    不需要深度学习，但效果有限
    """
    print("\n[备选方案] 使用颜色直方图 ReID")
    print("  优点: 不需要预训练模型")
    print("  缺点: 对光照敏感，区分度有限")
    

if __name__ == "__main__":
    print("=" * 60)
    print("    OSNet ReID 模型下载")
    print("=" * 60)
    
    download_osnet()
