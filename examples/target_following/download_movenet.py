"""
下载 MoveNet Lightning TFLite 模型
Download MoveNet Lightning TFLite Model

MoveNet 是 Google 开发的超快姿态估计模型：
- Lightning 版本: 192x192 输入，~2.5MB
- Thunder 版本: 256x256 输入，~5MB
"""

import os
import sys
import urllib.request

# 模型目录
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download_movenet():
    """下载 MoveNet Lightning 模型"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # TFHub 提供的 TFLite 模型 URL
    # Lightning int8 版本 (最小最快)
    models = {
        "movenet_lightning_int8": {
            "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
            "filename": "movenet_lightning_int8.tflite",
            "size": "2.5 MB",
            "input_size": 192
        },
        "movenet_lightning_f16": {
            "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
            "filename": "movenet_lightning_f16.tflite",
            "size": "4.8 MB",
            "input_size": 192
        }
    }
    
    # 默认下载 int8 版本 (最快)
    model_key = "movenet_lightning_int8"
    model_info = models[model_key]
    
    output_path = os.path.join(MODELS_DIR, model_info["filename"])
    
    if os.path.exists(output_path):
        size = os.path.getsize(output_path) / 1024 / 1024
        print(f"[INFO] 模型已存在: {output_path} ({size:.1f} MB)")
        return output_path
    
    print(f"[INFO] 下载 MoveNet Lightning ({model_info['size']})...")
    print(f"[INFO] URL: {model_info['url']}")
    print(f"[INFO] 保存到: {output_path}")
    
    try:
        # 下载模型
        urllib.request.urlretrieve(model_info["url"], output_path)
        
        size = os.path.getsize(output_path) / 1024 / 1024
        print(f"[SUCCESS] 下载完成: {size:.1f} MB")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        
        # 备用下载链接 (Kaggle/直接链接)
        backup_url = "https://storage.googleapis.com/movenet/models/movenet_singlepose_lightning_int8_4.tflite"
        print(f"[INFO] 尝试备用链接: {backup_url}")
        
        try:
            urllib.request.urlretrieve(backup_url, output_path)
            size = os.path.getsize(output_path) / 1024 / 1024
            print(f"[SUCCESS] 下载完成: {size:.1f} MB")
            return output_path
        except Exception as e2:
            print(f"[ERROR] 备用下载也失败: {e2}")
            return None


def main():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║           MoveNet Lightning 模型下载工具                ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print("MoveNet 特点:")
    print("  - 输入尺寸: 192x192 (比 YOLOv8 的 640x640 小很多)")
    print("  - 模型大小: ~2.5 MB (比 YOLOv8n-pose 的 12.9 MB 小)")
    print("  - 速度: 50+ FPS (比 YOLOv8 快 5-10 倍)")
    print("  - 关键点: 17个 COCO 标准关键点")
    print()
    
    result = download_movenet()
    
    if result:
        print()
        print("[SUCCESS] MoveNet 下载完成!")
        print(f"[INFO] 模型位置: {result}")
    else:
        print()
        print("[FAILED] 下载失败，请手动下载:")
        print("  1. 访问 https://tfhub.dev/google/movenet/singlepose/lightning/4")
        print("  2. 下载 TFLite 模型")
        print("  3. 保存到 models/movenet_lightning_int8.tflite")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
