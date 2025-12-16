"""
性能分析脚本 - 测量各模块耗时
Performance Profiler - Measure Module Latency
"""

import cv2
import numpy as np
import time
import sys

def profile_detectors():
    """分析各检测器耗时"""
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║           目标跟随系统 - 性能分析                        ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    # 加载模型
    print("[加载模型...]")
    from detectors.gesture_detector import GestureDetector
    from detectors.face_detector import FaceDetector
    from detectors.mobilefacenet_recognizer import MobileFaceNetRecognizer
    from detectors.person_detector import PersonDetector
    from config import GestureConfig, FaceDetectorConfig, MobileFaceNetConfig, PersonDetectorConfig
    
    gesture = GestureDetector(GestureConfig())
    face_det = FaceDetector(FaceDetectorConfig())
    face_rec = MobileFaceNetRecognizer(MobileFaceNetConfig())
    person = PersonDetector(PersonDetectorConfig())
    
    gesture.load()
    face_det.load()
    face_rec.load()
    person.load()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    
    print("\n[开始性能分析 (100帧)...]\n")
    
    # 统计
    times = {
        'camera_read': [],
        'gesture': [],
        'face_detect': [],
        'face_recognize': [],
        'person_detect': [],
        'visualize': [],
        'total': []
    }
    
    for i in range(100):
        total_start = time.time()
        
        # 摄像头读取
        t0 = time.time()
        ret, frame = cap.read()
        times['camera_read'].append((time.time() - t0) * 1000)
        if not ret:
            continue
        
        # 手势检测
        t0 = time.time()
        gesture.detect(frame)
        times['gesture'].append((time.time() - t0) * 1000)
        
        # 人脸检测
        t0 = time.time()
        faces = face_det.detect(frame)
        times['face_detect'].append((time.time() - t0) * 1000)
        
        # 人脸识别 (如果有人脸)
        t0 = time.time()
        if faces:
            face_rec.extract_feature(frame, bbox=faces[0].bbox, keypoints=faces[0].keypoints)
        times['face_recognize'].append((time.time() - t0) * 1000)
        
        # 人体检测
        t0 = time.time()
        person.detect(frame)
        times['person_detect'].append((time.time() - t0) * 1000)
        
        # 可视化 (简单绘制)
        t0 = time.time()
        for face in faces:
            cv2.rectangle(frame, 
                         (int(face.bbox[0]), int(face.bbox[1])),
                         (int(face.bbox[2]), int(face.bbox[3])),
                         (0, 255, 0), 2)
        times['visualize'].append((time.time() - t0) * 1000)
        
        times['total'].append((time.time() - total_start) * 1000)
        
        # 显示
        cv2.imshow("Profile", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 打印结果
    print("=" * 60)
    print("  各模块平均耗时 (毫秒)")
    print("=" * 60)
    print(f"  {'模块':<20} {'平均':<10} {'最小':<10} {'最大':<10}")
    print("-" * 60)
    
    total_avg = 0
    for name, values in times.items():
        if values:
            avg = np.mean(values)
            min_v = np.min(values)
            max_v = np.max(values)
            if name != 'total':
                total_avg += avg
            print(f"  {name:<20} {avg:>8.1f}ms {min_v:>8.1f}ms {max_v:>8.1f}ms")
    
    print("-" * 60)
    total_time = np.mean(times['total'])
    theoretical_fps = 1000 / total_time if total_time > 0 else 0
    print(f"  总计: {total_time:.1f}ms/帧 = {theoretical_fps:.1f} FPS (理论最大)")
    print()
    
    # 优化建议
    print("  [优化建议]")
    gesture_avg = np.mean(times['gesture'])
    face_det_avg = np.mean(times['face_detect'])
    person_avg = np.mean(times['person_detect'])
    
    if gesture_avg > 20:
        print(f"  ⚠️ 手势检测较慢 ({gesture_avg:.1f}ms), 建议增加 gesture_detect_interval")
    if face_det_avg > 30:
        print(f"  ⚠️ 人脸检测较慢 ({face_det_avg:.1f}ms), 建议增加 face_detect_interval")
    if person_avg > 50:
        print(f"  ⚠️ 人体检测较慢 ({person_avg:.1f}ms), 建议增加 person_detect_interval")
    
    # 按跳帧计算实际帧率
    print()
    print("  [按当前跳帧配置估算]")
    # 当前配置: gesture=4, face=8, person=12
    # 假设24帧中: gesture=6次, face=3次, person=2次
    # 注意：不是每帧都做所有检测
    from config import AppConfig
    cfg = AppConfig()
    
    # 计算24帧周期内的检测耗时 (取最小公倍数)
    frames_in_cycle = 24
    gesture_count = frames_in_cycle // cfg.gesture_detect_interval
    face_count = frames_in_cycle // cfg.face_detect_interval
    person_count = frames_in_cycle // cfg.person_detect_interval
    
    print(f"  配置: gesture每{cfg.gesture_detect_interval}帧, face每{cfg.face_detect_interval}帧, person每{cfg.person_detect_interval}帧")
    print(f"  24帧周期内: gesture={gesture_count}次, face={face_count}次, person={person_count}次")
    
    # 实际每帧平均检测耗时
    effective_detect_time = (
        gesture_count * gesture_avg +
        face_count * face_det_avg +
        person_count * person_avg
    ) / frames_in_cycle
    
    # 加上固定开销
    fixed_time = np.mean(times['camera_read']) + np.mean(times['visualize'])
    
    # 但是检测是异步的，实际是每帧只做对应的检测
    # 最差情况：同时做 gesture + face + person
    worst_case = gesture_avg + face_det_avg + person_avg + fixed_time
    
    # 平均情况：大部分帧只有 camera_read
    avg_case = fixed_time + effective_detect_time
    
    effective_fps = 1000 / avg_case if avg_case > 0 else 0
    worst_fps = 1000 / worst_case if worst_case > 0 else 0
    
    print(f"  平均每帧检测耗时: {effective_detect_time:.1f}ms")
    print(f"  估算帧率: 平均 {effective_fps:.1f} FPS, 最差 {worst_fps:.1f} FPS")
    print()

if __name__ == "__main__":
    profile_detectors()
