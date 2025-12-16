"""
MobileFaceNet 模型下载脚本
Download MobileFaceNet Model

模型来源: InsightFace
大小: ~4MB (对比 ArcFace ~166MB)
"""

import os
import urllib.request
import sys

# 模型目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# MobileFaceNet 模型 URL (从 ONNX Model Zoo)
# 备选: InsightFace buffalo_s 包含更小的识别模型
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"

# 使用 InsightFace 的 buffalo_s (更小)
BUFFALO_S_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"


def download_with_progress(url: str, filepath: str) -> bool:
    """带进度条的下载"""
    print(f"[INFO] 正在下载: {url}")
    print(f"[INFO] 保存到: {filepath}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r[PROGRESS] {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, filepath, show_progress)
        print("\n[SUCCESS] 下载完成!")
        return True
    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        return False


def download_mobilefacenet():
    """下载 MobileFaceNet 模型"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    output_path = os.path.join(MODELS_DIR, "mobilefacenet.onnx")
    
    if os.path.exists(output_path):
        print(f"[INFO] MobileFaceNet 模型已存在: {output_path}")
        return True
    
    print("=" * 60)
    print("  下载 MobileFaceNet 模型 (轻量级人脸识别)")
    print("=" * 60)
    print()
    
    # 尝试从 buffalo_s.zip 提取
    zip_path = os.path.join(MODELS_DIR, "buffalo_s.zip")
    
    if not os.path.exists(zip_path):
        print("[INFO] 下载 buffalo_s.zip (包含轻量级识别模型)...")
        if not download_with_progress(BUFFALO_S_URL, zip_path):
            return False
    
    # 解压提取 w600k_mbf.onnx
    print("[INFO] 解压模型...")
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # 查找 MobileFaceNet 模型
            for name in zf.namelist():
                if 'w600k_mbf' in name.lower() or 'mobilefacenet' in name.lower():
                    print(f"[INFO] 找到: {name}")
                    # 提取
                    with zf.open(name) as src:
                        with open(output_path, 'wb') as dst:
                            dst.write(src.read())
                    print(f"[SUCCESS] 已保存: {output_path}")
                    return True
            
            # 列出所有文件
            print("[INFO] ZIP 包含以下文件:")
            for name in zf.namelist():
                print(f"  - {name}")
            
            # 如果没找到 mbf，使用 det_500m 作为备选（这是检测模型，不是识别模型）
            # 实际上 buffalo_s 里可能没有 MobileFaceNet，需要用其他方式
            print("[WARNING] 未找到 MobileFaceNet，将使用备选方案...")
            
    except Exception as e:
        print(f"[ERROR] 解压失败: {e}")
    
    # 备选: 从 ONNX Model Zoo 下载 (较大)
    print("[INFO] 尝试备选下载源...")
    alt_url = "https://huggingface.co/rocca/mobilefacenet-onnx/resolve/main/mobilefacenet.onnx"
    return download_with_progress(alt_url, output_path)


def main():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║           MobileFaceNet 模型下载工具                    ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    success = download_mobilefacenet()
    
    if success:
        print()
        print("[SUCCESS] MobileFaceNet 下载完成!")
        print("[INFO] 模型位置:", os.path.join(MODELS_DIR, "mobilefacenet.onnx"))
        print()
        print("现在可以在 main_v2.py 中使用 'F' 键切换到 MobileFaceNet")
    else:
        print()
        print("[ERROR] 下载失败，请手动下载模型")
        print("[INFO] 可以从以下地址下载:")
        print("  - https://github.com/deepinsight/insightface")
        print("  - https://huggingface.co/rocca/mobilefacenet-onnx")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
