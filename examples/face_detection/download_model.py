"""
模型下载脚本
Model Download Script

下载 SCRFD 预训练模型 (来自 InsightFace)
支持两种方式:
1. 从 buffalo_sc.zip 解压获取 (推荐)
2. 从备用镜像下载
"""

import os
import urllib.request
import zipfile
import shutil
from config import MODELS_DIR

# SCRFD 模型信息
MODELS = {
    "buffalo_sc": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        "filename": "det_500m.onnx",  # ZIP 内的检测模型
        "output_name": "scrfd_500m_bnkps.onnx",
        "size_mb": 14.3,
        "description": "SCRFD-500M (from buffalo_sc.zip), 640x640 input, with keypoints"
    },
    "scrfd_person": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_person_2.5g.onnx",
        "filename": "scrfd_person_2.5g.onnx",
        "output_name": "scrfd_person_2.5g.onnx",
        "size_mb": 3.54,
        "description": "SCRFD-2.5G for person detection"
    }
}

DEFAULT_MODEL = "buffalo_sc"


def download_with_progress(url: str, filepath: str):
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


def extract_model_from_zip(zip_path: str, model_name: str, output_path: str) -> bool:
    """从 ZIP 文件中提取模型"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # 查找目标文件
            for name in zf.namelist():
                if name.endswith(model_name):
                    print(f"[INFO] 从 ZIP 中提取: {name}")
                    # 提取到临时文件
                    with zf.open(name) as src, open(output_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"[SUCCESS] 模型已保存到: {output_path}")
                    return True
            print(f"[ERROR] ZIP 中未找到 {model_name}")
            return False
    except Exception as e:
        print(f"[ERROR] 解压失败: {e}")
        return False


def download_model(model_name: str = DEFAULT_MODEL) -> bool:
    """
    下载指定模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        是否成功
    """
    if model_name not in MODELS:
        print(f"[ERROR] 未知模型: {model_name}")
        print(f"[INFO] 可用模型: {list(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_name]
    output_path = os.path.join(MODELS_DIR, model_info["output_name"])
    
    # 检查是否已存在
    if os.path.exists(output_path):
        print(f"[INFO] 模型已存在: {output_path}")
        return True
    
    # 确保目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 下载
    print(f"[INFO] 模型: {model_info['description']}")
    print(f"[INFO] 预计大小: {model_info['size_mb']} MB")
    
    url = model_info["url"]
    
    # 如果是 ZIP 文件
    if url.endswith('.zip'):
        zip_path = os.path.join(MODELS_DIR, os.path.basename(url))
        if not download_with_progress(url, zip_path):
            return False
        
        # 从 ZIP 中提取模型
        success = extract_model_from_zip(zip_path, model_info["filename"], output_path)
        
        # 清理 ZIP 文件
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"[INFO] 已清理临时文件: {zip_path}")
        
        return success
    else:
        # 直接下载 ONNX 文件
        return download_with_progress(url, output_path)


def list_models():
    """列出可用模型"""
    print("\n可用的 SCRFD 模型:")
    print("-" * 60)
    for name, info in MODELS.items():
        status = "✓ 已下载" if os.path.exists(os.path.join(MODELS_DIR, info["output_name"])) else "✗ 未下载"
        print(f"  {name}")
        print(f"    描述: {info['description']}")
        print(f"    大小: {info['size_mb']} MB")
        print(f"    状态: {status}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_models()
        else:
            download_model(sys.argv[1])
    else:
        # 默认下载轻量级模型
        print("=" * 60)
        print("SCRFD 人脸检测模型下载工具")
        print("=" * 60)
        list_models()
        
        print("-" * 60)
        print(f"正在下载默认模型: {DEFAULT_MODEL}")
        download_model(DEFAULT_MODEL)
