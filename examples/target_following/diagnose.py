"""
诊断脚本 - 检查系统资源使用情况
Diagnostic Script - Check System Resource Usage
"""

import psutil
import sys
import time
import gc

def print_memory_info():
    """打印内存信息"""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    # 系统内存
    system_mem = psutil.virtual_memory()
    
    print("=" * 60)
    print("  内存使用情况")
    print("=" * 60)
    print(f"  进程内存:")
    print(f"    RSS (常驻内存): {mem_info.rss / 1024 / 1024:.1f} MB")
    print(f"    VMS (虚拟内存): {mem_info.vms / 1024 / 1024:.1f} MB")
    print()
    print(f"  系统内存:")
    print(f"    总计: {system_mem.total / 1024 / 1024 / 1024:.1f} GB")
    print(f"    已用: {system_mem.used / 1024 / 1024 / 1024:.1f} GB ({system_mem.percent}%)")
    print(f"    可用: {system_mem.available / 1024 / 1024 / 1024:.1f} GB")
    print("=" * 60)
    print()
    
    return mem_info.rss / 1024 / 1024

def test_model_loading():
    """测试模型加载内存"""
    print("\n[测试] 模型加载内存占用\n")
    
    base_mem = print_memory_info()
    
    # 1. 测试 MediaPipe
    print("[1/4] 加载 MediaPipe Hands...")
    from detectors.gesture_detector import GestureDetector
    from config import GestureConfig
    gesture = GestureDetector(GestureConfig())
    gesture.load()
    mem_after_gesture = print_memory_info()
    print(f"  MediaPipe 增加: {mem_after_gesture - base_mem:.1f} MB\n")
    
    # 2. 测试 SCRFD
    print("[2/4] 加载 SCRFD...")
    from detectors.face_detector import FaceDetector
    from config import FaceDetectorConfig
    face_det = FaceDetector(FaceDetectorConfig())
    face_det.load()
    mem_after_face_det = print_memory_info()
    print(f"  SCRFD 增加: {mem_after_face_det - mem_after_gesture:.1f} MB\n")
    
    # 3. 测试 MobileFaceNet
    print("[3/4] 加载 MobileFaceNet...")
    from detectors.mobilefacenet_recognizer import MobileFaceNetRecognizer
    from config import MobileFaceNetConfig
    face_rec = MobileFaceNetRecognizer(MobileFaceNetConfig())
    face_rec.load()
    mem_after_face_rec = print_memory_info()
    print(f"  MobileFaceNet 增加: {mem_after_face_rec - mem_after_face_det:.1f} MB\n")
    
    # 4. 测试 YOLOv8-Pose
    print("[4/4] 加载 YOLOv8-Pose...")
    from detectors.person_detector import PersonDetector
    from config import PersonDetectorConfig
    person = PersonDetector(PersonDetectorConfig())
    person.load()
    mem_after_person = print_memory_info()
    print(f"  YOLOv8-Pose 增加: {mem_after_person - mem_after_face_rec:.1f} MB\n")
    
    print("=" * 60)
    print(f"  总计模型内存: {mem_after_person - base_mem:.1f} MB")
    print("=" * 60)
    
    return gesture, face_det, face_rec, person

def test_inference_loop(gesture, face_det, face_rec, person, frames=100):
    """测试推理循环的内存"""
    import cv2
    import numpy as np
    
    print(f"\n[测试] 推理循环 ({frames} 帧)\n")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    mem_before = print_memory_info()
    
    start_time = time.time()
    for i in range(frames):
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 手势检测 (每3帧)
        if i % 3 == 0:
            gesture.detect(frame)
        
        # 人脸检测 (每6帧)
        if i % 6 == 0:
            faces = face_det.detect(frame)
            # 人脸识别
            for face in faces[:1]:  # 只处理第一个
                face_rec.extract_feature(frame, bbox=face.bbox, keypoints=face.keypoints)
        
        # 人体检测 (每6帧)
        if i % 6 == 0:
            person.detect(frame)
        
        # 每20帧打印一次
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"  帧 {i+1}/{frames}: {fps:.1f} FPS, 内存: {mem:.1f} MB")
    
    cap.release()
    
    elapsed = time.time() - start_time
    mem_after = print_memory_info()
    
    print(f"\n  推理 {frames} 帧耗时: {elapsed:.1f} 秒")
    print(f"  平均帧率: {frames / elapsed:.1f} FPS")
    print(f"  内存变化: {mem_after - mem_before:+.1f} MB")
    
    # 强制垃圾回收
    gc.collect()
    mem_after_gc = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"  GC后内存: {mem_after_gc:.1f} MB (释放: {mem_after - mem_after_gc:.1f} MB)")

def check_potential_issues():
    """检查潜在问题"""
    print("\n[检查] 潜在问题\n")
    
    issues = []
    
    # 1. 检查系统内存
    mem = psutil.virtual_memory()
    if mem.available < 2 * 1024 * 1024 * 1024:  # < 2GB
        issues.append(f"⚠️ 可用内存不足: {mem.available / 1024 / 1024 / 1024:.1f} GB")
    
    # 2. 检查 CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        issues.append(f"⚠️ CPU 使用率过高: {cpu_percent}%")
    
    # 3. 检查模型文件
    import os
    from config import MODELS_DIR
    models = [
        ("scrfd_500m_bnkps.onnx", "人脸检测"),
        ("mobilefacenet.onnx", "人脸识别"),
        ("yolov8n-pose.onnx", "人体检测"),
    ]
    for model, name in models:
        path = os.path.join(MODELS_DIR, model)
        if not os.path.exists(path):
            issues.append(f"❌ 缺少模型: {name} ({model})")
        else:
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  ✓ {name}: {model} ({size:.1f} MB)")
    
    if issues:
        print("\n  发现问题:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("\n  ✓ 未发现明显问题")
    
    return len(issues) == 0

def main():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║           目标跟随系统 - 诊断工具                        ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    # 检查潜在问题
    check_potential_issues()
    
    # 测试模型加载
    try:
        gesture, face_det, face_rec, person = test_model_loading()
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 测试推理循环
    try:
        test_inference_loop(gesture, face_det, face_rec, person, frames=60)
    except Exception as e:
        print(f"[ERROR] 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n[完成] 诊断结束\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
