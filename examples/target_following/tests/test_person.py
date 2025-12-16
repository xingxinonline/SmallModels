"""
人体检测单独验证测试
Person Detection Test (YOLOv8-Pose)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config import PersonDetectorConfig, CameraConfig
from detectors.person_detector import PersonDetector
from core.camera import CameraCapture


# COCO 骨架连接
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6),  # 肩膀
    (5, 7), (7, 9),  # 左臂
    (6, 8), (8, 10),  # 右臂
    (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (13, 15),  # 左腿
    (12, 14), (14, 16)  # 右腿
]


def draw_skeleton(image, keypoints, color=(0, 255, 255)):
    """绘制骨架"""
    for i, j in SKELETON:
        if keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3:
            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(image, pt1, pt2, color, 2)
    
    for i, kp in enumerate(keypoints):
        if kp[2] > 0.3:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)


def main():
    print("=" * 60)
    print("    人体检测测试 (YOLOv8-Pose)")
    print("=" * 60)
    print()
    
    # 初始化
    camera = CameraCapture(CameraConfig())
    detector = PersonDetector(PersonDetectorConfig())
    
    # 加载模型
    if not detector.load():
        print("[ERROR] 人体检测器加载失败")
        print("[INFO] 请先运行 download_models.py 下载模型")
        return 1
    
    if not camera.open():
        print("[ERROR] 摄像头打开失败")
        return 1
    
    print()
    print("[INFO] 测试开始! 按 'q' 退出")
    print("-" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                continue
            
            # 人体检测
            t0 = time.time()
            detections = detector.detect(frame)
            detect_time = time.time() - t0
            
            # 绘制结果
            output = frame.copy()
            
            for i, det in enumerate(detections):
                bbox = det.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # 绘制边框
                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 绘制骨架
                if det.keypoints is not None:
                    draw_skeleton(output, det.keypoints)
                
                # 显示信息
                label = f"Person {i+1}: {det.confidence:.2f}"
                cv2.putText(
                    output, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )
            
            # 状态信息
            cv2.putText(
                output, f"Persons: {len(detections)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            cv2.putText(
                output, f"Infer: {detect_time*1000:.1f}ms",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
            
            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                output, f"FPS: {fps:.1f}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
            
            cv2.imshow("Person Detection Test", output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    print(f"\n[INFO] 测试完成，共处理 {frame_count} 帧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
