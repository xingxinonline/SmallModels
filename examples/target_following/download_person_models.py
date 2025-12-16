"""
下载人体检测和识别模型
Download Person Detection and ReID Models

模型:
1. YOLOv5-Nano (ONNX) - 人体检测 (~3.9MB)
2. MobileNetV2-ReID (ONNX) - 人体识别 (~13MB)
"""

import os
import sys
import urllib.request
from pathlib import Path

# 模型目录
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


def download_yolov5n():
    """下载 YOLOv5-Nano ONNX 模型"""
    save_path = MODELS_DIR / "yolov5n.onnx"
    
    if save_path.exists():
        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"[跳过] YOLOv5-Nano 已存在 ({size_mb:.1f} MB): {save_path}")
        return True
    
    # 使用 Ultralytics 官方导出的 ONNX
    # 可选: https://github.com/ultralytics/yolov5/releases
    urls = [
        # 官方 YOLOv5n ONNX
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
    ]
    
    for url in urls:
        if download_with_progress(url, save_path, "YOLOv5-Nano (人体检测)"):
            return True
    
    # 如果下载失败，提供手动下载说明
    print("\n[提示] 自动下载失败，请手动下载:")
    print("  1. 访问: https://github.com/ultralytics/yolov5/releases/tag/v7.0")
    print("  2. 下载: yolov5n.onnx")
    print(f"  3. 放到: {save_path}")
    return False


def download_reid_model():
    """下载 MobileNetV2-ReID ONNX 模型"""
    save_path = MODELS_DIR / "mobilenetv2_reid.onnx"
    
    if save_path.exists():
        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"[跳过] MobileNetV2-ReID 已存在 ({size_mb:.1f} MB): {save_path}")
        return True
    
    # ReID 模型来源选项:
    # 1. torchreid 导出
    # 2. fast-reid 导出
    # 3. 预训练 ONNX
    
    urls = [
        # ONNX Model Zoo (如果有)
        # 或者使用 Hugging Face
        # 这里提供一个常见的 ReID 模型源
    ]
    
    # 由于 ReID 模型不像 YOLO 那样有官方 ONNX 发布，
    # 我们需要从 PyTorch 模型转换
    
    print("\n[注意] MobileNetV2-ReID 需要手动导出:")
    print("  选项 1: 使用 torchreid 库导出")
    print("    pip install torchreid")
    print("    python -c \"")
    print("      from torchreid import models")
    print("      model = models.build_model(name='mobilenetv2_x1_0', num_classes=1000)")
    print("      # 导出 ONNX...")
    print("    \"")
    print("")
    print("  选项 2: 使用提供的转换脚本")
    print("    python convert_reid_to_onnx.py")
    print("")
    print(f"  保存到: {save_path}")
    
    # 尝试使用 HuggingFace 或其他源
    # 这里我们创建一个简单的导出脚本
    return False


def create_reid_export_script():
    """创建 ReID 模型导出脚本"""
    script_path = MODELS_DIR.parent / "convert_reid_to_onnx.py"
    
    if script_path.exists():
        print(f"[跳过] 导出脚本已存在: {script_path}")
        return
    
    script_content = '''"""
将 MobileNetV2-ReID 模型导出为 ONNX 格式
Export MobileNetV2-ReID model to ONNX format

依赖: pip install torch torchvision
可选: pip install torchreid (如果需要完整 ReID 功能)
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from pathlib import Path
import os


class MobileNetV2ReID(nn.Module):
    """
    MobileNetV2 用于 ReID 任务
    输出: 256D embedding
    """
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        
        # 加载预训练 MobileNetV2 (ImageNet)
        base = mobilenet_v2(weights="IMAGENET1K_V1")
        
        # 移除分类头
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ReID embedding 层
        in_features = base.last_channel  # 1280
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x):
        # x: [N, 3, 256, 128]
        x = self.features(x)  # [N, 1280, H, W]
        x = self.pool(x)       # [N, 1280, 1, 1]
        x = x.view(x.size(0), -1)  # [N, 1280]
        x = self.embedding(x)  # [N, 256]
        return x


def export_onnx():
    """导出 ONNX 模型"""
    model = MobileNetV2ReID(embedding_dim=256)
    model.eval()
    
    # 输入尺寸: [1, 3, 256, 128] (height=256, width=128)
    dummy_input = torch.randn(1, 3, 256, 128)
    
    # 输出路径
    save_path = Path(__file__).parent / "models" / "mobilenetv2_reid.onnx"
    save_path.parent.mkdir(exist_ok=True)
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        opset_version=11,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"}
        }
    )
    
    size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"[✓] 模型已导出: {save_path} ({size_mb:.1f} MB)")
    
    # 验证
    try:
        import onnxruntime as ort
        import numpy as np
        
        sess = ort.InferenceSession(str(save_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        
        test_input = np.random.randn(1, 3, 256, 128).astype(np.float32)
        output = sess.run(None, {input_name: test_input})[0]
        
        print(f"[✓] 验证通过: 输入 {test_input.shape} → 输出 {output.shape}")
        
    except Exception as e:
        print(f"[!] 验证跳过: {e}")


if __name__ == "__main__":
    export_onnx()
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"[✓] 导出脚本已创建: {script_path}")


def main():
    print("=" * 60)
    print("    人体检测和识别模型下载")
    print("=" * 60)
    print(f"\n模型目录: {MODELS_DIR}")
    
    # 下载 YOLOv5-Nano
    yolo_ok = download_yolov5n()
    
    # 创建 ReID 导出脚本
    create_reid_export_script()
    
    # 尝试下载或导出 ReID 模型
    reid_ok = download_reid_model()
    
    print("\n" + "=" * 60)
    print("    下载总结")
    print("=" * 60)
    print(f"  YOLOv5-Nano:     {'✓ 已就绪' if yolo_ok else '✗ 需手动下载'}")
    print(f"  MobileNetV2-ReID: {'✓ 已就绪' if reid_ok else '✗ 需导出'}")
    
    if not reid_ok:
        print("\n[下一步] 运行以下命令导出 ReID 模型:")
        print("  uv run python convert_reid_to_onnx.py")


if __name__ == "__main__":
    main()
