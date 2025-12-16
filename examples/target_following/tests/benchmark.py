"""
综合性能测试脚本
Performance Benchmark

功能:
1. 测试所有模型的加载时间
2. 测试各模块推理速度
3. 显示内存使用情况
4. 生成性能报告

使用方法:
1. 运行脚本
2. 等待测试完成
3. 查看性能报告
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
import psutil
from config import (
    FaceDetectorConfig, MobileFaceNetConfig, 
    MediaPipePoseConfig, GestureConfig
)


def get_memory_mb():
    """获取当前进程内存使用 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_module(name, load_func, detect_func, frame, n_runs=50):
    """测试单个模块"""
    print(f"\n[测试] {name}")
    print("-" * 40)
    
    # 记录加载前内存
    mem_before = get_memory_mb()
    
    # 加载模型
    t0 = time.time()
    module = load_func()
    load_time = (time.time() - t0) * 1000
    
    if module is None:
        print(f"  ❌ 加载失败")
        return None
    
    # 记录加载后内存
    mem_after = get_memory_mb()
    mem_used = mem_after - mem_before
    
    print(f"  ✓ 加载时间: {load_time:.1f}ms")
    print(f"  ✓ 内存占用: {mem_used:.1f}MB")
    
    # 预热
    for _ in range(5):
        detect_func(module, frame)
    
    # 推理速度测试
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        detect_func(module, frame)
        times.append((time.time() - t0) * 1000)
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    print(f"  ✓ 推理时间: {avg_time:.1f}ms (min: {min_time:.1f}, max: {max_time:.1f}, std: {std_time:.1f})")
    print(f"  ✓ 理论帧率: {1000/avg_time:.1f} FPS")
    
    return {
        "name": name,
        "load_time": load_time,
        "memory": mem_used,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "fps": 1000/avg_time
    }


def main():
    print("=" * 60)
    print("    性能基准测试 (Performance Benchmark)")
    print("=" * 60)
    print()
    
    # 系统信息
    print("[系统信息]")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print(f"  RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"  初始内存: {get_memory_mb():.1f} MB")
    print()
    
    # 准备测试图像
    print("[准备] 打开摄像头获取测试帧...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] 无法读取帧")
        return
    
    print(f"  测试帧尺寸: {frame.shape}")
    print()
    
    results = []
    
    # 1. 人脸检测 (SCRFD)
    def load_face_detector():
        from detectors.face_detector import FaceDetector
        detector = FaceDetector(FaceDetectorConfig())
        if detector.load():
            return detector
        return None
    
    def detect_face(detector, frame):
        return detector.detect(frame)
    
    r = test_module("人脸检测 (SCRFD)", load_face_detector, detect_face, frame)
    if r:
        results.append(r)
    
    # 2. 人脸识别 (MobileFaceNet)
    def load_face_recognizer():
        from detectors.mobilefacenet_recognizer import MobileFaceNetRecognizer
        recognizer = MobileFaceNetRecognizer(MobileFaceNetConfig())
        if recognizer.load():
            return recognizer
        return None
    
    # 先获取人脸
    face_detector = load_face_detector()
    faces = face_detector.detect(frame)
    face_detector.release()
    
    def detect_face_feature(recognizer, frame):
        if faces:
            face = faces[0]
            return recognizer.extract_feature(frame, face.bbox, face.keypoints)
        return None
    
    r = test_module("人脸识别 (MobileFaceNet)", load_face_recognizer, detect_face_feature, frame)
    if r:
        results.append(r)
    
    # 3. 人体检测 (MediaPipe Pose)
    def load_person_detector():
        from detectors.mediapipe_pose_detector import MediaPipePoseDetector
        detector = MediaPipePoseDetector(MediaPipePoseConfig())
        if detector.load():
            return detector
        return None
    
    def detect_person(detector, frame):
        return detector.detect(frame)
    
    r = test_module("人体检测 (MediaPipe Pose)", load_person_detector, detect_person, frame)
    if r:
        results.append(r)
    
    # 4. 手势检测 (MediaPipe Hands)
    def load_gesture_detector():
        from detectors.gesture_detector import GestureDetector
        detector = GestureDetector(GestureConfig())
        if detector.load():
            return detector
        return None
    
    def detect_gesture(detector, frame):
        return detector.detect(frame)
    
    r = test_module("手势检测 (MediaPipe Hands)", load_gesture_detector, detect_gesture, frame)
    if r:
        results.append(r)
    
    # 生成报告
    print()
    print("=" * 60)
    print("    性能报告")
    print("=" * 60)
    print()
    
    # 表格
    print(f"{'模块':<25} {'加载(ms)':<10} {'内存(MB)':<10} {'推理(ms)':<10} {'FPS':<10}")
    print("-" * 65)
    
    total_load = 0
    total_mem = 0
    total_infer = 0
    
    for r in results:
        print(f"{r['name']:<25} {r['load_time']:<10.1f} {r['memory']:<10.1f} {r['avg_time']:<10.1f} {r['fps']:<10.1f}")
        total_load += r['load_time']
        total_mem += r['memory']
        total_infer += r['avg_time']
    
    print("-" * 65)
    print(f"{'合计':<25} {total_load:<10.1f} {total_mem:<10.1f} {total_infer:<10.1f} -")
    print()
    
    # 帧率估算
    print("[帧率估算]")
    print(f"  所有模块串行: {1000/total_infer:.1f} FPS (每帧 {total_infer:.1f}ms)")
    print()
    
    # 基于检测间隔的估算
    gesture_interval = 4
    face_interval = 8
    person_interval = 6
    
    # 每帧平均耗时
    gesture_time = results[3]['avg_time'] if len(results) > 3 else 15
    face_time = (results[0]['avg_time'] + results[1]['avg_time']) if len(results) > 1 else 50
    person_time = results[2]['avg_time'] if len(results) > 2 else 15
    
    avg_frame_time = (
        gesture_time / gesture_interval +
        face_time / face_interval +
        person_time / person_interval +
        5  # 摄像头读取 + 显示
    )
    
    print(f"  使用检测间隔优化: ~{1000/avg_frame_time:.0f} FPS (每帧 ~{avg_frame_time:.1f}ms)")
    print(f"    - 手势: 每{gesture_interval}帧检测")
    print(f"    - 人脸: 每{face_interval}帧检测")
    print(f"    - 人体: 每{person_interval}帧检测")
    print()
    
    print("[内存使用]")
    print(f"  当前总内存: {get_memory_mb():.1f} MB")
    print()
    
    print("[建议]")
    if total_infer > 100:
        print("  ⚠️ 推理总耗时较长，建议增大检测间隔")
    if total_mem > 500:
        print("  ⚠️ 内存占用较高，注意系统资源")
    if len(results) == 4:
        print("  ✓ 所有模块正常运行")


if __name__ == "__main__":
    main()
