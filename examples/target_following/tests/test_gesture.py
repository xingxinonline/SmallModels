"""
手势识别单独验证测试
Gesture Detection Test
"""

import sys
import os
import time

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config import GestureConfig, CameraConfig, GestureType
from detectors.gesture_detector import GestureDetector
from core.camera import CameraCapture


def main():
    print("=" * 60)
    print("    手势识别测试 (MediaPipe Hands)")
    print("=" * 60)
    print()
    
    # 初始化
    camera = CameraCapture(CameraConfig())
    detector = GestureDetector(GestureConfig())
    
    # 加载
    if not detector.load():
        print("[ERROR] 手势检测器加载失败")
        return 1
    
    if not camera.open():
        print("[ERROR] 摄像头打开失败")
        return 1
    
    print()
    print("[INFO] 测试开始! 按 'q' 退出")
    print("[INFO] 支持的手势: 张开手掌, 握拳, 剪刀手")
    print("-" * 60)
    
    gesture_names = {
        GestureType.NONE: "None",
        GestureType.OPEN_PALM: "Open Palm (Start)",
        GestureType.CLOSED_FIST: "Closed Fist (Stop)",
        GestureType.VICTORY: "Victory (Pause)",
        GestureType.THUMB_UP: "Thumb Up",
        GestureType.THUMB_DOWN: "Thumb Down",
    }
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                continue
            
            # 检测手势
            t0 = time.time()
            result = detector.detect(frame)
            infer_time = time.time() - t0
            
            # 绘制结果
            output = frame.copy()
            
            # 绘制手部边框和关键点
            if result.hand_bbox is not None:
                bbox = result.hand_bbox.astype(int)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if result.hand_landmarks is not None:
                    for pt in result.hand_landmarks.landmarks:
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(output, (x, y), 4, (255, 0, 0), -1)
            
            # 显示手势
            gesture_name = gesture_names.get(result.gesture_type, "未知")
            color = (0, 255, 0) if result.gesture_type != GestureType.NONE else (128, 128, 128)
            
            cv2.putText(
                output, f"Gesture: {gesture_name}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
            )
            
            cv2.putText(
                output, f"Infer: {infer_time*1000:.1f}ms",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1
            )
            
            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                output, f"FPS: {fps:.1f}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
            )
            
            cv2.imshow("Gesture Test", output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        camera.release()
        detector.release()
        cv2.destroyAllWindows()
    
    print(f"\n[INFO] 测试完成，共处理 {frame_count} 帧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
