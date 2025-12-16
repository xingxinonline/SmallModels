"""
手势检测模型测试脚本
Gesture Detection Model Test

功能:
1. 实时显示手势检测结果
2. 显示手部关键点
3. 识别各种手势
4. 显示检测耗时

使用方法:
1. 运行脚本
2. 做出各种手势观察识别效果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from config import GestureConfig, GestureType


def main():
    print("=" * 60)
    print("    手势检测测试工具 (MediaPipe Hands)")
    print("=" * 60)
    print()
    
    # 配置
    config = GestureConfig()
    
    # 加载检测器
    print("[INFO] 加载手势检测器 (MediaPipe Hands)...")
    from detectors.gesture_detector import GestureDetector
    detector = GestureDetector(config)
    if not detector.load():
        print("[ERROR] 手势检测器加载失败")
        return
    print("      ✓ 手势检测器就绪")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print()
    print("-" * 60)
    print("  支持的手势:")
    print("  - 张开手掌 (Open Palm)")
    print("  - 握拳 (Closed Fist)")
    print("  - 剪刀手 (Victory)")
    print("  - 竖起大拇指 (Thumb Up)")
    print()
    print("  控制说明:")
    print("  - 'q': 退出")
    print("-" * 60)
    print()
    
    # 性能统计
    detect_times = []
    fps_time = time.time()
    fps_count = 0
    fps = 0
    
    # 手势颜色映射
    gesture_colors = {
        GestureType.NONE: (128, 128, 128),      # 灰色
        GestureType.OPEN_PALM: (0, 255, 0),     # 绿色
        GestureType.CLOSED_FIST: (0, 0, 255),   # 红色
        GestureType.VICTORY: (255, 0, 255),     # 紫色
        GestureType.THUMB_UP: (255, 255, 0),    # 青色
        GestureType.THUMB_DOWN: (0, 165, 255),  # 橙色
    }
    
    gesture_names = {
        GestureType.NONE: "None",
        GestureType.OPEN_PALM: "Open Palm (张开手掌)",
        GestureType.CLOSED_FIST: "Closed Fist (握拳)",
        GestureType.VICTORY: "Victory (剪刀手)",
        GestureType.THUMB_UP: "Thumb Up (竖大拇指)",
        GestureType.THUMB_DOWN: "Thumb Down",
    }
    
    # 手势统计
    gesture_history = []
    
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
        
        # 检测手势 (计时)
        t0 = time.time()
        result = detector.detect(frame)
        detect_time = (time.time() - t0) * 1000
        detect_times.append(detect_time)
        if len(detect_times) > 30:
            detect_times.pop(0)
        avg_time = np.mean(detect_times)
        
        # 记录手势历史
        if result.gesture_type != GestureType.NONE:
            gesture_history.append(result.gesture_type.value)
            if len(gesture_history) > 10:
                gesture_history.pop(0)
        
        # 显示
        display = frame.copy()
        
        # 获取当前手势颜色
        current_color = gesture_colors.get(result.gesture_type, (128, 128, 128))
        
        # 绘制手部关键点
        hand_lm = result.hand_landmarks
        landmarks = hand_lm.landmarks if hand_lm is not None else None
        if landmarks is not None:
            # 手部连接
            connections = [
                # 大拇指
                (0, 1), (1, 2), (2, 3), (3, 4),
                # 食指
                (0, 5), (5, 6), (6, 7), (7, 8),
                # 中指
                (0, 9), (9, 10), (10, 11), (11, 12),
                # 无名指
                (0, 13), (13, 14), (14, 15), (15, 16),
                # 小指
                (0, 17), (17, 18), (18, 19), (19, 20),
                # 手掌
                (5, 9), (9, 13), (13, 17),
            ]
            
            # 绘制连接
            for i, j in connections:
                if i < len(landmarks) and j < len(landmarks):
                    pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
                    pt2 = (int(landmarks[j][0]), int(landmarks[j][1]))
                    cv2.line(display, pt1, pt2, current_color, 2)
            
            # 绘制关键点
            for i in range(len(landmarks)):
                x, y = landmarks[i][0], landmarks[i][1]
                if i in [4, 8, 12, 16, 20]:  # 指尖
                    cv2.circle(display, (int(x), int(y)), 6, current_color, -1)
                else:
                    cv2.circle(display, (int(x), int(y)), 3, current_color, -1)
        
        # 显示手势名称（大字）
        gesture_name = gesture_names.get(result.gesture_type, "Unknown")
        cv2.putText(display, gesture_name, (display.shape[1]//2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, current_color, 3)
        
        # 显示置信度条
        if result.confidence > 0:
            bar_width = int(result.confidence * 200)
            cv2.rectangle(display, (display.shape[1]//2 - 100, 60), 
                          (display.shape[1]//2 - 100 + bar_width, 80), current_color, -1)
            cv2.rectangle(display, (display.shape[1]//2 - 100, 60), 
                          (display.shape[1]//2 + 100, 80), (255, 255, 255), 2)
            cv2.putText(display, f"{result.confidence:.1%}", (display.shape[1]//2 + 110, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示状态信息
        info_y = 120
        cv2.putText(display, f"FPS: {fps}", (display.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, f"Detector: MediaPipe Hands", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(display, f"Detect: {avg_time:.1f}ms", (10, info_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        
        cv2.putText(display, f"Hands: {1 if landmarks is not None else 0}", (10, info_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示手势历史
        if gesture_history:
            history_str = " -> ".join(gesture_history[-5:])
            cv2.putText(display, f"History: {history_str}", (10, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 底部说明
        cv2.putText(display, "q: Quit", 
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Gesture Detection Test", display)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    
    print()
    print(f"[结果] 平均检测耗时: {np.mean(detect_times):.1f}ms")


if __name__ == "__main__":
    main()
