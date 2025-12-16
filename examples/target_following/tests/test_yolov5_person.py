"""
YOLOv5-Nano 人体检测测试
Test YOLOv5-Nano Person Detection

功能:
  - 实时多人检测
  - 显示检测框和置信度
  - 可调阈值

控制:
  - '+'/'-': 调整置信度阈值
  - 'q': 退出
"""

import cv2
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import YOLOv5PersonConfig, MODELS_DIR
from detectors.yolov5_person_detector import YOLOv5PersonDetector


def main():
    print("=" * 60)
    print("    YOLOv5-Nano 人体检测测试")
    print("=" * 60)
    
    # 检查模型文件
    model_path = os.path.join(MODELS_DIR, "yolov5n.onnx")
    if not os.path.exists(model_path):
        print(f"\n[错误] 模型文件不存在: {model_path}")
        print("\n请先运行下载脚本:")
        print("  uv run python download_person_models.py")
        return
    
    # 配置
    config = YOLOv5PersonConfig(
        model_path=model_path,
        confidence_threshold=0.5,
        nms_threshold=0.45
    )
    
    # 初始化检测器
    detector = YOLOv5PersonDetector(config)
    if not detector.load():
        print("[错误] 检测器加载失败")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n[控制]")
    print("  '+'/'-': 调整置信度阈值")
    print("  'q': 退出")
    print()
    
    threshold = config.confidence_threshold
    detect_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        start_time = time.perf_counter()
        
        # 临时更新阈值
        detector.config.confidence_threshold = threshold
        persons = detector.detect(frame)
        
        detect_time = (time.perf_counter() - start_time) * 1000
        detect_times.append(detect_time)
        if len(detect_times) > 30:
            detect_times.pop(0)
        
        # 绘制结果
        for person in persons:
            x1, y1, x2, y2 = person.bbox.astype(int)
            conf = person.confidence
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 置信度标签
            label = f"Person {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 统计信息
        avg_time = np.mean(detect_times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        info_lines = [
            f"Persons: {len(persons)}",
            f"Threshold: {threshold:.2f}",
            f"Detect: {detect_time:.1f}ms (avg: {avg_time:.1f}ms)",
            f"FPS: {fps:.1f}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv5-Nano Person Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            threshold = min(0.95, threshold + 0.05)
            print(f"[阈值] {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(0.1, threshold - 0.05)
            print(f"[阈值] {threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[结果] 平均检测时间: {np.mean(detect_times):.1f}ms")


if __name__ == "__main__":
    main()
