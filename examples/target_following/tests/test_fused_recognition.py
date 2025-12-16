"""
融合目标识别测试 - 人脸 + 颜色直方图
Test Fused Target Recognition (Face + Color Histogram)

优势:
  - 正面: 人脸识别准确区分
  - 背面: 颜色直方图补充
  - 相同衣服: 人脸可区分
  - 不同光照: 两者互补

控制:
  - 's': 保存当前人为目标
  - 'c': 清除目标
  - '+'/'-': 调整阈值
  - 'q': 退出
"""

import cv2
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    YOLOv5PersonConfig, FaceDetectorConfig, MobileFaceNetConfig, 
    MODELS_DIR
)
from detectors.yolov5_person_detector import YOLOv5PersonDetector
from detectors.face_detector import FaceDetector
from detectors.mobilefacenet_recognizer import MobileFaceNetRecognizer
from detectors.color_histogram_reid import ColorHistogramReID, ColorHistogramConfig
from detectors.fused_recognizer import (
    FusedTargetRecognizer, FusedTargetFeature, FusedRecognizerConfig
)


def main():
    print("=" * 60)
    print("    融合目标识别测试 (人脸 + 颜色直方图)")
    print("=" * 60)
    
    # 检查模型
    yolo_path = os.path.join(MODELS_DIR, "yolov5n.onnx")
    scrfd_path = os.path.join(MODELS_DIR, "scrfd_500m_bnkps.onnx")
    mobilefacenet_path = os.path.join(MODELS_DIR, "mobilefacenet.onnx")
    
    missing = []
    if not os.path.exists(yolo_path):
        missing.append("yolov5n.onnx")
    if not os.path.exists(scrfd_path):
        missing.append("scrfd_500m_bnkps.onnx")
    if not os.path.exists(mobilefacenet_path):
        missing.append("mobilefacenet.onnx")
    
    if missing:
        print(f"\n[错误] 缺少模型: {missing}")
        return
    
    # 初始化检测器
    person_detector = YOLOv5PersonDetector(YOLOv5PersonConfig(model_path=yolo_path))
    face_detector = FaceDetector(FaceDetectorConfig(model_path=scrfd_path))
    face_recognizer = MobileFaceNetRecognizer(MobileFaceNetConfig(model_path=mobilefacenet_path))
    color_reid = ColorHistogramReID(ColorHistogramConfig(similarity_threshold=0.7))
    fused_recognizer = FusedTargetRecognizer(FusedRecognizerConfig())
    
    # 加载模型
    if not person_detector.load():
        print("[错误] 人体检测器加载失败")
        return
    if not face_detector.load():
        print("[错误] 人脸检测器加载失败")
        return
    if not face_recognizer.load():
        print("[错误] 人脸识别器加载失败")
        return
    color_reid.load()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n[控制]")
    print("  's': 保存当前人为目标 (需要看到人脸)")
    print("  'c': 清除目标")
    print("  '+'/'-': 调整阈值")
    print("  'q': 退出")
    print("\n[策略]")
    print("  - 正面: 人脸识别 (权重 0.7)")
    print("  - 背面: 颜色直方图 (权重 0.3)")
    print("  - 相同衣服但不同人: 人脸可区分")
    print()
    
    target_feature: FusedTargetFeature = None
    threshold_offset = 0.0  # 用于手动调整
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测人体
        persons = person_detector.detect(frame)
        
        # 检测人脸
        faces = face_detector.detect(frame)
        
        # 为每个人体匹配人脸并提取特征
        for person in persons:
            px1, py1, px2, py2 = person.bbox.astype(int)
            
            # 查找在这个人体区域内的人脸
            matched_face = None
            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                # 人脸中心在人体区域内
                fc_x = (fx1 + fx2) // 2
                fc_y = (fy1 + fy2) // 2
                if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                    matched_face = face
                    break
            
            # 提取特征
            candidate = FusedTargetFeature(timestamp=time.time())
            
            # 人脸特征
            if matched_face is not None:
                face_feature = face_recognizer.extract_feature(
                    frame, matched_face.bbox, matched_face.keypoints
                )
                if face_feature:
                    candidate.face_embedding = face_feature.embedding
                    candidate.face_bbox = matched_face.bbox.copy()
            
            # 颜色特征
            color_feature = color_reid.extract_feature(frame, person.bbox)
            if color_feature:
                candidate.color_histogram = color_feature.combined_hist
            
            candidate.bbox = person.bbox.copy()
            
            # 匹配
            if target_feature is None:
                color = (255, 165, 0)  # 橙色 - 无目标
                label = "No Target"
                method = ""
            else:
                is_match, similarity, method = fused_recognizer.is_same_target(
                    target_feature, candidate
                )
                
                # 根据匹配方式选择不同阈值
                if "color_only" in method:
                    base_threshold = 0.70  # 只有颜色时用更高阈值
                elif "fused" in method:
                    base_threshold = 0.55  # 融合时用中等阈值
                else:
                    base_threshold = 0.45  # 只有人脸时用较低阈值
                
                # 应用手动阈值调整
                adjusted_threshold = base_threshold - threshold_offset
                is_match = similarity >= adjusted_threshold
                
                if is_match:
                    color = (0, 255, 0)  # 绿色
                    label = f"TARGET {similarity:.2f}"
                else:
                    color = (0, 0, 255)  # 红色
                    label = f"Other {similarity:.2f}"
            
            # 绘制人体框
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            
            # 绘制人脸框 (如果有)
            if matched_face is not None:
                fx1, fy1, fx2, fy2 = matched_face.bbox.astype(int)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
            
            # 标签
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (px1, py1 - label_size[1] - 5), 
                         (px1 + label_size[0], py1), color, -1)
            cv2.putText(frame, label, (px1, py1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示匹配方式
            if method:
                cv2.putText(frame, method[:30], (px1, py2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 统计信息
        has_face_str = "YES" if (target_feature and target_feature.has_face) else "NO"
        has_body_str = "YES" if (target_feature and target_feature.has_body) else "NO"
        
        info_lines = [
            f"Persons: {len(persons)}, Faces: {len(faces)}",
            f"Target: Face={has_face_str}, Body={has_body_str}",
            f"Threshold Offset: {threshold_offset:+.2f}",
            "Fused: Face(0.7) + Color(0.3)"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Fused Target Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存第一个检测到的人
            if persons:
                target_feature = FusedTargetFeature(timestamp=time.time())
                
                # 找这个人的人脸
                px1, py1, px2, py2 = persons[0].bbox.astype(int)
                for face in faces:
                    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                    fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                        face_feature = face_recognizer.extract_feature(
                            frame, face.bbox, face.keypoints
                        )
                        if face_feature:
                            target_feature.face_embedding = face_feature.embedding
                            target_feature.face_bbox = face.bbox.copy()
                            print(f"  - 人脸特征已保存")
                        break
                
                # 颜色特征
                color_feature = color_reid.extract_feature(frame, persons[0].bbox)
                if color_feature:
                    target_feature.color_histogram = color_feature.combined_hist
                    print(f"  - 颜色特征已保存")
                
                target_feature.bbox = persons[0].bbox.copy()
                
                print(f"[保存] 目标特征: Face={target_feature.has_face}, Body={target_feature.has_body}")
            else:
                print("[提示] 未检测到人体")
                
        elif key == ord('c'):
            target_feature = None
            threshold_offset = 0.0
            print("[清除] 目标已清除")
            
        elif key == ord('+') or key == ord('='):
            threshold_offset = min(0.3, threshold_offset + 0.05)
            print(f"[阈值偏移] {threshold_offset:+.2f}")
            
        elif key == ord('-'):
            threshold_offset = max(-0.3, threshold_offset - 0.05)
            print(f"[阈值偏移] {threshold_offset:+.2f}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
