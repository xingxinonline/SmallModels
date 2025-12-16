"""
模型下载脚本
Model Download Script

下载目标跟随系统所需的所有模型
"""

import os
import urllib.request
import zipfile
import shutil
from config import MODELS_DIR


# 模型信息
MODELS = {
    "scrfd": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        "filename": "det_500m.onnx",
        "output_name": "scrfd_500m_bnkps.onnx",
        "description": "SCRFD-500M 人脸检测模型",
        "size_mb": 14.3,
        "is_zip": True
    },
    "arcface": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        "filename": "w600k_r50.onnx",
        "output_name": "w600k_r50.onnx",
        "description": "ArcFace 人脸识别模型 (从 buffalo_sc.zip)",
        "size_mb": 14.3,
        "is_zip": True,
        "share_download": "scrfd"  # 与 SCRFD 共享下载
    },
    "yolov8n-pose": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
        "output_name": "yolov8n-pose.onnx",
        "description": "YOLOv8n-Pose 人体检测模型",
        "size_mb": 6.5,
        "is_zip": False,
        "requires_export": True  # 需要导出为 ONNX
    }
}


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


def extract_from_zip(zip_path: str, filename: str, output_path: str) -> bool:
    """从 ZIP 中提取文件"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith(filename):
                    print(f"[INFO] 从 ZIP 中提取: {name}")
                    with zf.open(name) as src, open(output_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"[SUCCESS] 已保存到: {output_path}")
                    return True
        print(f"[ERROR] ZIP 中未找到 {filename}")
        return False
    except Exception as e:
        print(f"[ERROR] 解压失败: {e}")
        return False


def export_yolov8_to_onnx() -> bool:
    """将 YOLOv8 PT 模型导出为 ONNX"""
    try:
        from ultralytics import YOLO
        
        pt_path = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
        onnx_path = os.path.join(MODELS_DIR, "yolov8n-pose.onnx")
        
        if not os.path.exists(pt_path):
            print("[ERROR] PT 模型不存在")
            return False
        
        print("[INFO] 正在导出 YOLOv8 为 ONNX...")
        model = YOLO(pt_path)
        model.export(format="onnx", imgsz=640, simplify=True)
        
        # 移动导出的文件
        exported = pt_path.replace(".pt", ".onnx")
        if os.path.exists(exported):
            shutil.move(exported, onnx_path)
            print(f"[SUCCESS] ONNX 模型已导出: {onnx_path}")
            
            # 清理 PT 文件
            os.remove(pt_path)
            return True
        
        return False
    except ImportError:
        print("[ERROR] 请先安装 ultralytics: uv add ultralytics")
        return False
    except Exception as e:
        print(f"[ERROR] 导出失败: {e}")
        return False


def download_all():
    """下载所有模型"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("目标跟随系统模型下载工具")
    print("=" * 60)
    print()
    
    # 下载 buffalo_sc.zip (包含 SCRFD 和 ArcFace)
    zip_path = os.path.join(MODELS_DIR, "buffalo_sc.zip")
    scrfd_path = os.path.join(MODELS_DIR, "scrfd_500m_bnkps.onnx")
    arcface_path = os.path.join(MODELS_DIR, "w600k_r50.onnx")
    
    need_download_zip = not os.path.exists(scrfd_path) or not os.path.exists(arcface_path)
    
    if need_download_zip:
        print("[1/3] 下载 SCRFD + ArcFace 模型...")
        if download_with_progress(MODELS["scrfd"]["url"], zip_path):
            # 提取 SCRFD
            if not os.path.exists(scrfd_path):
                extract_from_zip(zip_path, "det_500m.onnx", scrfd_path)
            
            # 提取 ArcFace
            if not os.path.exists(arcface_path):
                extract_from_zip(zip_path, "w600k_r50.onnx", arcface_path)
            
            # 清理 ZIP
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print("[INFO] 已清理临时文件")
    else:
        print("[1/3] SCRFD + ArcFace 模型已存在 ✓")
    
    print()
    
    # 下载 YOLOv8n-Pose
    yolo_path = os.path.join(MODELS_DIR, "yolov8n-pose.onnx")
    pt_path = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
    
    if not os.path.exists(yolo_path):
        print("[2/3] 下载 YOLOv8n-Pose 模型...")
        if download_with_progress(MODELS["yolov8n-pose"]["url"], pt_path):
            print("[3/3] 导出 ONNX 格式...")
            export_yolov8_to_onnx()
    else:
        print("[2/3] YOLOv8n-Pose 模型已存在 ✓")
        print("[3/3] 跳过 ONNX 导出 ✓")
    
    print()
    print("=" * 60)
    print("模型检查:")
    print("-" * 60)
    
    models_status = [
        ("SCRFD (人脸检测)", scrfd_path),
        ("ArcFace (人脸识别)", arcface_path),
        ("YOLOv8n-Pose (人体检测)", yolo_path),
    ]
    
    all_ready = True
    for name, path in models_status:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {name}: 未找到")
            all_ready = False
    
    print("=" * 60)
    
    if all_ready:
        print("[SUCCESS] 所有模型已就绪!")
    else:
        print("[WARNING] 部分模型未下载成功，请检查网络后重试")
    
    return all_ready


if __name__ == "__main__":
    download_all()
