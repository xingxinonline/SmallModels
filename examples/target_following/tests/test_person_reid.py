"""
MobileNetV2-ReID 人体识别测试
Test MobileNetV2 Person Re-Identification

功能:
  - 实时人体检测 + ReID 特征提取
  - 保存目标人体特征
  - 实时匹配验证

控制:
  - 's': 保存当前检测到的人体作为目标
  - 'c': 清除目标
  - '+'/'-': 调整相似度阈值
  - 'q': 退出
"""

import cv2
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import YOLOv5PersonConfig, PersonReIDConfig, MODELS_DIR
from detectors.yolov5_person_detector import YOLOv5PersonDetector
from detectors.person_reid_recognizer import PersonReIDRecognizer, PersonFeature


def main():
    print("=" * 60)
    print("    MobileNetV2-ReID 人体识别测试")
    print("=" * 60)
    
    # 检查模型文件
    yolo_path = os.path.join(MODELS_DIR, "yolov5n.onnx")
    reid_path = os.path.join(MODELS_DIR, "mobilenetv2_reid.onnx")
    
    missing = []
    if not os.path.exists(yolo_path):
        missing.append(f"YOLOv5-Nano: {yolo_path}")
    if not os.path.exists(reid_path):
        missing.append(f"MobileNetV2-ReID: {reid_path}")
    
    if missing:
        print("\n[错误] 模型文件不存在:")
        for m in missing:
            print(f"  - {m}")
        print("\n请先运行下载/导出脚本:")
        print("  uv run python download_person_models.py")
        print("  uv run python convert_reid_to_onnx.py")
        return
    
    # 配置
    detector_config = YOLOv5PersonConfig(
        model_path=yolo_path,
        confidence_threshold=0.5
    )
    reid_config = PersonReIDConfig(
        model_path=reid_path,
        similarity_threshold=0.5
    )
    
    # 初始化
    detector = YOLOv5PersonDetector(detector_config)
    recognizer = PersonReIDRecognizer(reid_config)
    
    if not detector.load():
        print("[错误] 人体检测器加载失败")
        return
    
    if not recognizer.load():
        print("[错误] ReID 识别器加载失败")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n[控制]")
    print("  's': 保存当前人体为目标")
    print("  'c': 清除目标")
    print("  '+'/'-': 调整相似度阈值")
    print("  'q': 退出")
    print()
    
    target_feature: PersonFeature = None
    threshold = reid_config.similarity_threshold
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测人体
        start_time = time.perf_counter()
        persons = detector.detect(frame)
        detect_time = (time.perf_counter() - start_time) * 1000
        
        # 处理每个检测到的人体
        for person in persons:
            x1, y1, x2, y2 = person.bbox.astype(int)
            
            # 提取特征
            feature = recognizer.extract_feature(frame, person.bbox)
            
            if feature is None:
                color = (128, 128, 128)  # 灰色 - 无法提取特征
                label = "No Feature"
            elif target_feature is None:
                color = (255, 165, 0)  # 橙色 - 无目标
                label = f"Person (no target)"
            else:
                # 计算相似度
                is_match, similarity = recognizer.is_same_person(
                    target_feature, feature, threshold
                )
                
                if is_match:
                    color = (0, 255, 0)  # 绿色 - 匹配
                    label = f"TARGET {similarity:.3f}"
                else:
                    color = (0, 0, 255)  # 红色 - 不匹配
                    label = f"Other {similarity:.3f}"
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 统计信息
        info_lines = [
            f"Persons: {len(persons)}",
            f"Threshold: {threshold:.2f}",
            f"Detect: {detect_time:.1f}ms",
            f"Target: {'SET' if target_feature else 'NONE'}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("MobileNetV2-ReID Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存第一个检测到的人体作为目标
            if persons:
                target_feature = recognizer.extract_feature(frame, persons[0].bbox)
                if target_feature:
                    print(f"[保存] 目标人体特征已保存 (embedding dim: {len(target_feature.embedding)})")
                else:
                    print("[错误] 无法提取特征")
            else:
                print("[提示] 未检测到人体")
        elif key == ord('c'):
            target_feature = None
            print("[清除] 目标已清除")
        elif key == ord('+') or key == ord('='):
            threshold = min(0.9, threshold + 0.05)
            print(f"[阈值] {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(0.1, threshold - 0.05)
            print(f"[阈值] {threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
