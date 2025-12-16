"""
增强版融合目标识别测试
Test Enhanced Fused Target Recognition

特征组成:
  1. 人脸识别 (MobileFaceNet)
  2. 分区颜色直方图 (6段 LAB+HSV)
  3. LBP 纹理特征
  4. 几何特征 (宽高比、相对高度)

控制:
  - 's': 保存当前人为目标 (可背对镜头保存!)
  - 'c': 清除目标
  - '+'/'-': 调整阈值
  - 'q': 退出
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODELS_DIR,
    YOLOv5PersonConfig, FaceDetectorConfig, MobileFaceNetConfig
)
from detectors.yolov5_person_detector import YOLOv5PersonDetector
from detectors.face_detector import FaceDetector
from detectors.mobilefacenet_recognizer import MobileFaceNetRecognizer
from detectors.enhanced_reid import EnhancedReIDExtractor, EnhancedReIDConfig
from detectors.fused_recognizer import (
    FusedTargetRecognizer, FusedTargetFeature, FusedRecognizerConfig
)


def main():
    print("=" * 60)
    print("    增强版融合目标识别测试")
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
    
    # 增强版 ReID
    enhanced_reid = EnhancedReIDExtractor(EnhancedReIDConfig(
        num_horizontal_parts=6,
        use_lbp=True,
        use_geometry=True,
        similarity_threshold=0.55
    ))
    
    # 融合识别器
    fused_recognizer = FusedTargetRecognizer(FusedRecognizerConfig(
        face_weight=0.6,
        body_weight=0.4,
        face_only_threshold=0.45,
        body_only_threshold=0.55,
        fused_threshold=0.50
    ))
    
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
    
    enhanced_reid.load()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n[控制]")
    print("  's': 保存当前人为目标 (可背对镜头!)")
    print("  'c': 清除目标")
    print("  '+'/'-': 调整阈值偏移")
    print("  'q': 退出")
    print("\n[特征]")
    print("  - 人脸: MobileFaceNet (128D)")
    print("  - 人体: 6段颜色(LAB+HSV) + LBP纹理 + 几何")
    print("  - 融合: Face(0.6) + Body(0.4)")
    print()
    
    target_feature: FusedTargetFeature = None
    threshold_offset = 0.0
    
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
            
            # 增强版人体特征
            body_feature = enhanced_reid.extract_feature(frame, person.bbox)
            if body_feature:
                candidate.body_feature = body_feature.combined_feature
                candidate.part_color_hists = body_feature.part_color_hists
                candidate.part_lbp_hists = body_feature.part_lbp_hists
                candidate.geometry = body_feature.geometry
            
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
                
                # 应用阈值偏移
                if "body_only" in method:
                    adjusted_threshold = 0.55 - threshold_offset
                elif "fused" in method:
                    adjusted_threshold = 0.50 - threshold_offset
                else:
                    adjusted_threshold = 0.45 - threshold_offset
                
                is_match = similarity >= adjusted_threshold
                
                if is_match:
                    color = (0, 255, 0)  # 绿色
                    label = f"TARGET {similarity:.2f}"
                else:
                    color = (0, 0, 255)  # 红色
                    label = f"Other {similarity:.2f}"
            
            # 绘制人体框
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            
            # 绘制分区线 (6段)
            crop_h = py2 - py1
            part_ratios = [0.12, 0.08, 0.15, 0.15, 0.25, 0.25]
            y_pos = py1
            for ratio in part_ratios[:-1]:
                y_pos += int(crop_h * ratio)
                cv2.line(frame, (px1, y_pos), (px2, y_pos), (100, 100, 100), 1)
            
            # 绘制人脸框 (如果有)
            if matched_face is not None:
                fx1, fy1, fx2, fy2 = matched_face.bbox.astype(int)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
                cv2.putText(frame, "Face", (fx1, fy1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 标签
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (px1, py1 - label_size[1] - 5), 
                         (px1 + label_size[0], py1), color, -1)
            cv2.putText(frame, label, (px1, py1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示匹配方式
            if method:
                method_short = method[:40]
                cv2.putText(frame, method_short, (px1, py2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # 统计信息
        has_face_str = "YES" if (target_feature and target_feature.has_face) else "NO"
        has_body_str = "YES" if (target_feature and target_feature.has_body) else "NO"
        
        info_lines = [
            f"Persons: {len(persons)}, Faces: {len(faces)}",
            f"Target: Face={has_face_str}, Body={has_body_str}",
            f"Threshold Offset: {threshold_offset:+.2f}",
            "Enhanced: 6-Part Color + LBP + Geometry"
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Enhanced Fused Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存第一个检测到的人 (可以没有人脸!)
            if persons:
                target_feature = FusedTargetFeature(timestamp=time.time())
                px1, py1, px2, py2 = persons[0].bbox.astype(int)
                
                # 找这个人的人脸
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
                            print(f"  - 人脸特征已保存 (128D)")
                        break
                
                # 增强版人体特征 (总是保存)
                body_feature = enhanced_reid.extract_feature(frame, persons[0].bbox)
                if body_feature:
                    target_feature.body_feature = body_feature.combined_feature
                    target_feature.part_color_hists = body_feature.part_color_hists
                    target_feature.part_lbp_hists = body_feature.part_lbp_hists
                    target_feature.geometry = body_feature.geometry
                    print(f"  - 人体特征已保存 (6段颜色+LBP+几何)")
                
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
            print(f"[阈值偏移] {threshold_offset:+.2f} (降低阈值，更容易匹配)")
        elif key == ord('-'):
            threshold_offset = max(-0.3, threshold_offset - 0.05)
            print(f"[阈值偏移] {threshold_offset:+.2f} (提高阈值，更严格)")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
