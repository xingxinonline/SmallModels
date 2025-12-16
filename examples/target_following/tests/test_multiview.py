"""
多视角目标跟踪测试
Test Multi-View Target Tracking

核心功能:
  1. 多视角特征库 - 自动积累正面/背面/侧面特征
  2. 运动一致性 - 转身时用位置连续性维持跟踪
  3. 自动学习 - 跟踪中自动学习新角度
  4. 时域平滑 - 多帧投票，避免闪烁

控制:
  - 's': 保存当前人为目标
  - 'a': 手动添加当前视角到目标特征库
  - 'c': 清除目标
  - 'm': 切换自动学习开关
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
from detectors.multiview_recognizer import (
    MultiViewRecognizer, MultiViewConfig, MultiViewTarget, ViewFeature
)


def extract_view_feature(
    frame: np.ndarray,
    person_bbox: np.ndarray,
    faces: list,
    face_recognizer,
    enhanced_reid
) -> ViewFeature:
    """提取视角特征"""
    view = ViewFeature(timestamp=time.time())
    
    px1, py1, px2, py2 = person_bbox.astype(int)
    
    # 查找人脸
    for face in faces:
        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
        fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
        
        if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
            face_feature = face_recognizer.extract_feature(
                frame, face.bbox, face.keypoints
            )
            if face_feature:
                view.has_face = True
                view.face_embedding = face_feature.embedding
            break
    
    # 人体特征
    body_feature = enhanced_reid.extract_feature(frame, person_bbox)
    if body_feature:
        view.part_color_hists = body_feature.part_color_hists
        view.part_lbp_hists = body_feature.part_lbp_hists
        view.geometry = body_feature.geometry
    
    return view


def main():
    print("=" * 60)
    print("    多视角目标跟踪测试")
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
    
    # 增强版 ReID (降低头部权重，因为正反面差异大)
    enhanced_reid = EnhancedReIDExtractor(EnhancedReIDConfig(
        num_horizontal_parts=6,
        use_lbp=True,
        use_geometry=True,
        similarity_threshold=0.50
    ))
    
    # 多视角识别器
    mv_config = MultiViewConfig(
        face_weight=0.6,
        body_weight=0.4,
        face_threshold=0.45,
        body_threshold=0.45,  # 多视角可以降低
        fused_threshold=0.45,
        motion_weight=0.20,   # 运动一致性权重
        auto_learn=True,
        learn_interval=2.0,
        smooth_window=5,
        confirm_threshold=3,
        # 降低头部权重 (正反面差异大)
        part_weights=[0.05, 0.12, 0.20, 0.20, 0.25, 0.18]
    )
    mv_recognizer = MultiViewRecognizer(mv_config)
    
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
    print("  's': 保存当前人为目标")
    print("  'a': 手动添加当前视角 (转身后按)")
    print("  'c': 清除目标")
    print("  'm': 切换自动学习")
    print("  '+'/'-': 调整阈值偏移")
    print("  'q': 退出")
    print("\n[策略]")
    print("  - 多视角特征库: 自动积累不同角度")
    print("  - 运动一致性: 短时转身维持跟踪")
    print("  - 时域平滑: 5帧投票，3帧确认")
    print()
    
    threshold_offset = 0.0
    last_match_person_idx = -1  # 上次匹配的人体索引
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        persons = person_detector.detect(frame)
        faces = face_detector.detect(frame)
        
        # 处理每个人
        for idx, person in enumerate(persons):
            px1, py1, px2, py2 = person.bbox.astype(int)
            
            # 提取特征
            view = extract_view_feature(
                frame, person.bbox, faces, face_recognizer, enhanced_reid
            )
            
            # 匹配
            if mv_recognizer.target is None:
                color = (255, 165, 0)  # 橙色
                label = "No Target"
                method = ""
            else:
                is_match, similarity, method = mv_recognizer.is_same_target(
                    view, person.bbox
                )
                
                # 应用阈值偏移
                adjusted_sim = similarity + threshold_offset
                is_match = adjusted_sim >= mv_config.body_threshold
                
                if is_match:
                    color = (0, 255, 0)  # 绿色
                    label = f"TARGET {similarity:.2f}"
                    last_match_person_idx = idx
                    
                    # 更新跟踪
                    mv_recognizer.update_tracking(person.bbox)
                    
                    # 自动学习
                    if mv_recognizer.auto_learn(view, person.bbox, is_match):
                        print(f"[自动学习] 新视角已添加, 总视角: {mv_recognizer.target.num_views}")
                else:
                    color = (0, 0, 255)  # 红色
                    label = f"Other {similarity:.2f}"
            
            # 绘制人体框
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            
            # 绘制分区线
            crop_h = py2 - py1
            part_ratios = [0.12, 0.08, 0.15, 0.15, 0.25, 0.25]
            y_pos = py1
            for ratio in part_ratios[:-1]:
                y_pos += int(crop_h * ratio)
                cv2.line(frame, (px1, y_pos), (px2, y_pos), (100, 100, 100), 1)
            
            # 人脸框
            if view.has_face:
                for face in faces:
                    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                    fc_x = (fx1 + fx2) // 2
                    if px1 <= fc_x <= px2:
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
                        break
            
            # 标签
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (px1, py1 - label_size[1] - 5),
                         (px1 + label_size[0], py1), color, -1)
            cv2.putText(frame, label, (px1, py1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 匹配方式
            if method:
                method_short = method[:50]
                cv2.putText(frame, method_short, (px1, py2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 状态信息
        target_info = "None"
        if mv_recognizer.target:
            num_views = mv_recognizer.target.num_views
            has_face = "Yes" if mv_recognizer.target.has_face_view else "No"
            target_info = f"Views={num_views}, Face={has_face}"
        
        info_lines = [
            f"Persons: {len(persons)}, Faces: {len(faces)}",
            f"Target: {target_info}",
            f"Auto Learn: {'ON' if mv_config.auto_learn else 'OFF'}",
            f"Motion Weight: {mv_config.motion_weight:.2f}",
            f"Threshold Offset: {threshold_offset:+.2f}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        cv2.imshow("Multi-View Tracking", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        elif key == ord('s'):
            # 保存目标
            if persons:
                view = extract_view_feature(
                    frame, persons[0].bbox, faces, face_recognizer, enhanced_reid
                )
                mv_recognizer.set_target(view, persons[0].bbox)
                face_str = "有人脸" if view.has_face else "无人脸"
                print(f"[保存] 目标已设置 ({face_str})")
            else:
                print("[提示] 未检测到人体")
                
        elif key == ord('a'):
            # 手动添加视角
            if mv_recognizer.target and persons:
                # 使用最近匹配的人，或第一个人
                person_idx = last_match_person_idx if 0 <= last_match_person_idx < len(persons) else 0
                
                view = extract_view_feature(
                    frame, persons[person_idx].bbox, faces, face_recognizer, enhanced_reid
                )
                
                # 强制添加 (绕过时间间隔检查)
                view.timestamp = time.time()
                if mv_recognizer.target._is_different_view(view, threshold=0.75):
                    mv_recognizer.target.view_features.append(view)
                    if len(mv_recognizer.target.view_features) > mv_recognizer.target.max_views:
                        mv_recognizer.target.view_features = (
                            [mv_recognizer.target.view_features[0]] + 
                            mv_recognizer.target.view_features[-mv_recognizer.target.max_views+1:]
                        )
                    face_str = "有人脸" if view.has_face else "无人脸"
                    print(f"[添加] 新视角已添加 ({face_str}), 总视角: {mv_recognizer.target.num_views}")
                else:
                    print("[提示] 视角太相似，未添加")
            else:
                print("[提示] 请先按's'保存目标")
                
        elif key == ord('c'):
            mv_recognizer.clear_target()
            last_match_person_idx = -1
            threshold_offset = 0.0
            print("[清除] 目标已清除")
            
        elif key == ord('m'):
            mv_config.auto_learn = not mv_config.auto_learn
            print(f"[自动学习] {'开启' if mv_config.auto_learn else '关闭'}")
            
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
