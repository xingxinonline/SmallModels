"""
人体检测模型测试脚本
Person Detection Model Test

功能:
1. 实时显示人体检测结果
2. 显示骨架关键点
3. 显示检测置信度
4. 显示检测耗时

使用方法:
1. 运行脚本
2. 观察检测效果
3. 调整参数
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from config import MediaPipePoseConfig


def main():
    print("=" * 60)
    print("    人体检测测试工具 (MediaPipe Pose)")
    print("=" * 60)
    print()
    
    # 配置
    config = MediaPipePoseConfig()
    
    # 加载检测器
    print("[INFO] 加载人体检测器 (MediaPipe Pose)...")
    from detectors.mediapipe_pose_detector import MediaPipePoseDetector
    detector = MediaPipePoseDetector(config)
    if not detector.load():
        print("[ERROR] 人体检测器加载失败")
        return
    print("      ✓ 人体检测器就绪")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print()
    print("-" * 60)
    print("  控制说明:")
    print("  - 'd': 切换骨架显示")
    print("  - 'b': 切换边界框显示")
    print("  - 'q': 退出")
    print("-" * 60)
    print()
    
    # 显示选项
    show_skeleton = True
    show_bbox = True
    
    # 性能统计
    detect_times = []
    fps_time = time.time()
    fps_count = 0
    fps = 0
    
    # 骨架连接
    POSE_CONNECTIONS = [
        # 躯干
        (11, 12), (11, 23), (12, 24), (23, 24),
        # 左臂
        (11, 13), (13, 15),
        # 右臂
        (12, 14), (14, 16),
        # 左腿
        (23, 25), (25, 27),
        # 右腿
        (24, 26), (26, 28),
        # 面部
        (0, 1), (0, 2), (1, 3), (2, 4),  # 眼睛和耳朵
        (9, 10),  # 嘴巴
    ]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        
        # 检测人体 (计时)
        t0 = time.time()
        detections = detector.detect(frame)
        detect_time = (time.time() - t0) * 1000
        detect_times.append(detect_time)
        if len(detect_times) > 30:
            detect_times.pop(0)
        avg_time = np.mean(detect_times)
        
        # 显示
        display = frame.copy()
        
        # 绘制检测结果
        for det in detections:
            # 边界框
            if show_bbox:
                x1, y1, x2, y2 = det.bbox.astype(int)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 置信度标签
                label = f"Person: {det.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 10, y1), (255, 0, 0), -1)
                cv2.putText(display, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 骨架
            if show_skeleton and det.keypoints is not None:
                keypoints = det.keypoints
                
                # 绘制关键点
                for i, (x, y, conf) in enumerate(keypoints):
                    if conf > 0.5:
                        # 颜色根据位置变化
                        if i < 5:  # 头部
                            color = (0, 255, 255)
                        elif i < 11:  # 上半身
                            color = (0, 255, 0)
                        else:  # 下半身
                            color = (255, 0, 255)
                        cv2.circle(display, (int(x), int(y)), 4, color, -1)
                
                # 绘制骨架连接
                for i, j in POSE_CONNECTIONS:
                    if i < len(keypoints) and j < len(keypoints):
                        if keypoints[i, 2] > 0.5 and keypoints[j, 2] > 0.5:
                            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                            cv2.line(display, pt1, pt2, (0, 200, 0), 2)
        
        # 显示状态信息
        info_y = 30
        cv2.putText(display, f"FPS: {fps}", (display.shape[1] - 100, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, f"Detector: MediaPipe Pose (Lite)", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(display, f"Persons: {len(detections)}", (10, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, f"Detect: {avg_time:.1f}ms", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
        
        # 显示开关状态
        cv2.putText(display, f"Skeleton: {'ON' if show_skeleton else 'OFF'}", (10, info_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"BBox: {'ON' if show_bbox else 'OFF'}", (10, info_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 底部说明
        cv2.putText(display, "d: Skeleton  b: BBox  q: Quit", 
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Person Detection Test", display)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_skeleton = not show_skeleton
            print(f"[INFO] 骨架显示: {'开' if show_skeleton else '关'}")
        elif key == ord('b'):
            show_bbox = not show_bbox
            print(f"[INFO] 边界框显示: {'开' if show_bbox else '关'}")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    print()
    print(f"[结果] 平均检测耗时: {np.mean(detect_times):.1f}ms")


if __name__ == "__main__":
    main()
