"""
人脸检测模型测试脚本
Face Detection Model Test

功能:
1. 实时显示人脸检测结果
2. 显示检测置信度
3. 按 '+'/'-' 调整置信度阈值
4. 显示检测耗时

使用方法:
1. 运行脚本
2. 观察不同距离/角度下的检测效果
3. 调整阈值找到最佳设置
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from config import FaceDetectorConfig


def main():
    print("=" * 60)
    print("    人脸检测测试工具 (SCRFD)")
    print("=" * 60)
    print()
    
    # 配置
    config = FaceDetectorConfig()
    threshold = config.confidence_threshold
    
    # 加载人脸检测器
    print("[INFO] 加载人脸检测器 (SCRFD)...")
    from detectors.face_detector import FaceDetector
    detector = FaceDetector(config)
    if not detector.load():
        print("[ERROR] 人脸检测器加载失败")
        return
    print("      ✓ 人脸检测器就绪")
    print(f"[INFO] 初始阈值: {threshold}")
    
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
    print("  - '+': 提高阈值 (+0.05)")
    print("  - '-': 降低阈值 (-0.05)")
    print("  - 'q': 退出")
    print("-" * 60)
    print()
    
    # 性能统计
    detect_times = []
    fps_time = time.time()
    fps_count = 0
    fps = 0
    
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
        
        # 更新检测器阈值
        detector.config.confidence_threshold = threshold
        
        # 检测人脸 (计时)
        t0 = time.time()
        faces = detector.detect(frame)
        detect_time = (time.time() - t0) * 1000
        detect_times.append(detect_time)
        if len(detect_times) > 30:
            detect_times.pop(0)
        avg_time = np.mean(detect_times)
        
        # 显示
        display = frame.copy()
        
        # 绘制人脸
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = face.confidence
            
            # 颜色根据置信度变化
            if conf > 0.8:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif conf > 0.6:
                color = (0, 255, 255)  # 黄色 - 中置信度
            else:
                color = (0, 165, 255)  # 橙色 - 低置信度
            
            # 绘制边界框
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # 绘制置信度
            label = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(display, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 绘制关键点 (如果有)
            if face.keypoints is not None:
                for lm in face.keypoints:
                    cv2.circle(display, (int(lm[0]), int(lm[1])), 2, (255, 0, 0), -1)
        
        # 显示状态信息
        info_y = 30
        cv2.putText(display, f"FPS: {fps}", (display.shape[1] - 100, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, f"Detector: SCRFD", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display, f"Threshold: {threshold:.2f}", (10, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(display, f"Faces: {len(faces)}", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, f"Detect: {avg_time:.1f}ms", (10, info_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
        
        # 底部说明
        cv2.putText(display, "+/-: Threshold  q: Quit", 
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Face Detection Test", display)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            threshold = min(0.95, threshold + 0.05)
            print(f"[INFO] 阈值: {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(0.1, threshold - 0.05)
            print(f"[INFO] 阈值: {threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    print()
    print(f"[结果] 推荐阈值: {threshold:.2f}")
    print(f"[结果] 平均检测耗时: {np.mean(detect_times):.1f}ms")


if __name__ == "__main__":
    main()
