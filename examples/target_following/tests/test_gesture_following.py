"""
手势控制目标跟随测试
Gesture-Controlled Target Following

手势控制:
  - 👋 张开手掌持续3秒: Toggle 启动/停止跟随
    - 空闲状态 → 启动跟随 (锁定最近的人)
    - 跟踪状态 → 停止跟随 (清除目标)

系统状态:
  - IDLE: 空闲状态，等待手势启动
  - TRACKING: 跟随中，持续跟踪目标
  - LOST_TARGET: 目标丢失，等待重新出现或手势停止

键盘控制 (备用):
  - 's': 手动保存目标
  - 'a': 手动添加视角
  - 'c': 手动清除目标
  - 'm': 切换自动学习
  - 'q': 退出
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODELS_DIR, GestureType, GestureConfig, SystemState,
    YOLOv5PersonConfig, FaceDetectorConfig, MobileFaceNetConfig
)
from detectors.yolov5_person_detector import YOLOv5PersonDetector
from detectors.face_detector import FaceDetector
from detectors.mobilefacenet_recognizer import MobileFaceNetRecognizer
from detectors.gesture_detector import GestureDetector, GestureResult
from detectors.enhanced_reid import EnhancedReIDExtractor, EnhancedReIDConfig
from detectors.multiview_recognizer import (
    MultiViewRecognizer, MultiViewConfig, ViewFeature
)
from core.state_machine import StateMachine


# 手势配置
GESTURE_HOLD_DURATION = 3.0  # 触发需要保持的秒数
GESTURE_COOLDOWN_SECONDS = 10.0  # 触发后冷却秒数 (防止连续触发)


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


def find_nearest_person(persons: list, frame_center: tuple):
    """找到离画面中心最近的人"""
    if not persons:
        return None, -1
    
    min_dist = float('inf')
    nearest_idx = 0
    
    for i, person in enumerate(persons):
        px1, py1, px2, py2 = person.bbox
        cx = (px1 + px2) / 2
        cy = (py1 + py2) / 2
        dist = (cx - frame_center[0])**2 + (cy - frame_center[1])**2
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    
    return persons[nearest_idx], nearest_idx


def draw_gesture_indicator(frame, gesture: GestureResult, state: SystemState, hold_progress: float = 0.0):
    """绘制手势指示器"""
    h, w = frame.shape[:2]
    
    # 绘制手部框
    if gesture.hand_bbox is not None:
        hx1, hy1, hx2, hy2 = gesture.hand_bbox.astype(int)
        
        if gesture.gesture_type == GestureType.OPEN_PALM:
            color = (0, 255, 255)  # 黄色
            if state == SystemState.IDLE:
                text = "HOLD TO START"
            else:
                text = "HOLD TO STOP"
        else:
            color = (255, 165, 0)
            text = ""
        
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
        if text:
            cv2.putText(frame, text, (hx1, hy1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制持续进度条
        if hold_progress > 0:
            bar_width = hx2 - hx1
            bar_height = 8
            bar_y = hy2 + 5
            
            # 背景
            cv2.rectangle(frame, (hx1, bar_y), (hx2, bar_y + bar_height), (50, 50, 50), -1)
            # 进度
            progress_width = int(bar_width * hold_progress)
            progress_color = (0, 255, 0) if hold_progress < 1.0 else (0, 255, 255)
            cv2.rectangle(frame, (hx1, bar_y), (hx1 + progress_width, bar_y + bar_height), progress_color, -1)
            # 边框
            cv2.rectangle(frame, (hx1, bar_y), (hx2, bar_y + bar_height), (255, 255, 255), 1)
            
            # 进度百分比
            pct_text = f"{int(hold_progress * 100)}%"
            cv2.putText(frame, pct_text, (hx1, bar_y + bar_height + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 状态指示器 (右上角)
    state_colors = {
        SystemState.IDLE: (128, 128, 128),          # 灰色
        SystemState.TRACKING: (0, 255, 0),          # 绿色
        SystemState.LOST_TARGET: (0, 165, 255)      # 橙色
    }
    state_color = state_colors.get(state, (255, 255, 255))
    
    cv2.circle(frame, (w - 30, 30), 15, state_color, -1)
    cv2.putText(frame, state.value, (w - 120, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)


def main():
    print("=" * 60)
    print("    手势控制目标跟随系统")
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
    
    # 手势检测器 (confirm_frames=1，因为持续时间检测在状态机中)
    gesture_config = GestureConfig(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        gesture_confirm_frames=1  # 立即响应，持续时间由状态机控制
    )
    gesture_detector = GestureDetector(gesture_config)
    
    # 增强版 ReID
    enhanced_reid = EnhancedReIDExtractor(EnhancedReIDConfig(
        num_horizontal_parts=6,
        use_lbp=True,
        use_geometry=True
    ))
    
    # 多视角识别器
    mv_config = MultiViewConfig(
        face_weight=0.6,
        body_weight=0.4,
        face_threshold=0.45,
        body_threshold=0.45,
        fused_threshold=0.45,
        motion_weight=0.20,
        auto_learn=True,
        learn_interval=2.0,
        smooth_window=5,
        confirm_threshold=3,
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
    if not gesture_detector.load():
        print("[错误] 手势检测器加载失败")
        return
    
    enhanced_reid.load()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n[手势控制]")
    print(f"  👋 张开手掌持续 {GESTURE_HOLD_DURATION:.0f} 秒: Toggle 启动/停止跟随")
    print("\n[键盘控制]")
    print("  's': 手动保存目标")
    print("  'a': 添加视角")
    print("  'c': 清除目标")
    print("  'm': 切换自动学习")
    print("  'q': 退出")
    print()
    
    # 状态机 (使用之前实现的持续时间检测)
    state_machine = StateMachine(
        lost_timeout_frames=30,
        gesture_hold_duration=GESTURE_HOLD_DURATION,
        gesture_cooldown_seconds=GESTURE_COOLDOWN_SECONDS
    )
    
    lost_frames = 0
    max_lost_frames = 30
    
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)
        
        # 计算 FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
        
        # 检测
        persons = person_detector.detect(frame)
        faces = face_detector.detect(frame)
        gesture = gesture_detector.detect(frame)
        
        # ============== 手势状态机 (持续时间检测) ==============
        current_time = time.time()
        old_state = state_machine.state
        
        # 处理手势 (需要持续 GESTURE_HOLD_DURATION 秒)
        state_changed = state_machine.process_gesture(gesture.gesture_type, current_time, debug=False)
        
        # 获取持续进度
        hold_progress = state_machine.get_gesture_hold_progress()
        
        # 状态变更处理
        if state_changed:
            if state_machine.state == SystemState.TRACKING and old_state == SystemState.IDLE:
                # 启动跟随
                nearest_person, idx = find_nearest_person(persons, frame_center)
                if nearest_person is not None:
                    view = extract_view_feature(
                        frame, nearest_person.bbox, faces, 
                        face_recognizer, enhanced_reid
                    )
                    mv_recognizer.set_target(view, nearest_person.bbox)
                    lost_frames = 0
                    face_str = "有人脸" if view.has_face else "无人脸"
                    print(f"[手势启动] 目标已锁定 ({face_str})")
                else:
                    # 没有人，回到空闲
                    state_machine.state = SystemState.IDLE
                    print("[提示] 未检测到人体，无法启动")
            
            elif state_machine.state == SystemState.IDLE and old_state in [SystemState.TRACKING, SystemState.LOST_TARGET]:
                # 停止跟随
                mv_recognizer.clear_target()
                lost_frames = 0
                print("[手势停止] 跟随已停止")
        
        # ============== 目标跟踪 ==============
        target_person_idx = -1
        
        if state_machine.state == SystemState.TRACKING:
            matched_any = False
            
            for idx, person in enumerate(persons):
                view = extract_view_feature(
                    frame, person.bbox, faces, face_recognizer, enhanced_reid
                )
                
                is_match, similarity, method = mv_recognizer.is_same_target(
                    view, person.bbox
                )
                
                if is_match:
                    matched_any = True
                    target_person_idx = idx
                    lost_frames = 0
                    
                    # 更新跟踪
                    mv_recognizer.update_tracking(person.bbox)
                    
                    # 自动学习
                    if mv_recognizer.auto_learn(view, person.bbox, True):
                        print(f"[自动学习] 新视角, 总数: {mv_recognizer.target.num_views}")
                    break
            
            if not matched_any:
                lost_frames += 1
                if lost_frames >= max_lost_frames:
                    state_machine.state = SystemState.LOST_TARGET
                    print("[目标丢失] 等待重新出现或手势停止")
        
        elif state_machine.state == SystemState.LOST_TARGET:
            # 尝试重新匹配
            for idx, person in enumerate(persons):
                view = extract_view_feature(
                    frame, person.bbox, faces, face_recognizer, enhanced_reid
                )
                
                is_match, similarity, method = mv_recognizer.is_same_target(
                    view, person.bbox
                )
                
                if is_match:
                    state_machine.state = SystemState.TRACKING
                    target_person_idx = idx
                    lost_frames = 0
                    mv_recognizer.update_tracking(person.bbox)
                    print("[重新锁定] 目标已恢复")
                    break
        
        # ============== 绘制 ==============
        for idx, person in enumerate(persons):
            px1, py1, px2, py2 = person.bbox.astype(int)
            
            if state_machine.state == SystemState.IDLE:
                color = (255, 165, 0)  # 橙色
                label = "Candidate"
            elif idx == target_person_idx:
                color = (0, 255, 0)  # 绿色
                label = "TARGET"
            else:
                color = (0, 0, 255)  # 红色
                label = "Other"
            
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (px1, py1 - label_size[1] - 5),
                         (px1 + label_size[0], py1), color, -1)
            cv2.putText(frame, label, (px1, py1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制手势指示器 (含进度条)
        draw_gesture_indicator(frame, gesture, state_machine.state, hold_progress)
        
        # 状态信息
        target_info = "None"
        if mv_recognizer.target:
            num_views = mv_recognizer.target.num_views
            has_face = "Y" if mv_recognizer.target.has_face_view else "N"
            target_info = f"Views={num_views}, Face={has_face}"
        
        info_lines = [
            f"FPS: {fps:.1f}",
            f"State: {state_machine.state.value}",
            f"Persons: {len(persons)}",
            f"Target: {target_info}",
            f"Gesture: {gesture.gesture_type.value}",
            f"Hold: {hold_progress*100:.0f}%" if hold_progress > 0 else ""
        ]
        
        for i, line in enumerate(info_lines):
            if line:
                cv2.putText(frame, line, (10, 25 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        # 手势提示
        if state_machine.state == SystemState.IDLE:
            cv2.putText(frame, f"Hold OPEN PALM {GESTURE_HOLD_DURATION:.0f}s to START", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif state_machine.state == SystemState.TRACKING:
            cv2.putText(frame, f"Hold OPEN PALM {GESTURE_HOLD_DURATION:.0f}s to STOP", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif state_machine.state == SystemState.LOST_TARGET:
            cv2.putText(frame, f"Target LOST - Hold PALM to STOP", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        cv2.imshow("Gesture-Controlled Following", frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if persons:
                nearest, _ = find_nearest_person(persons, frame_center)
                if nearest:
                    view = extract_view_feature(
                        frame, nearest.bbox, faces, face_recognizer, enhanced_reid
                    )
                    mv_recognizer.set_target(view, nearest.bbox)
                    state_machine.state = SystemState.TRACKING
                    print("[手动保存] 目标已锁定")
        elif key == ord('a'):
            if mv_recognizer.target and target_person_idx >= 0:
                person = persons[target_person_idx]
                view = extract_view_feature(
                    frame, person.bbox, faces, face_recognizer, enhanced_reid
                )
                if mv_recognizer.target._is_different_view(view, 0.75):
                    mv_recognizer.target.view_features.append(view)
                    print(f"[手动添加] 新视角, 总数: {mv_recognizer.target.num_views}")
        elif key == ord('c'):
            mv_recognizer.clear_target()
            state_machine.state = SystemState.IDLE
            print("[手动清除] 目标已清除")
        elif key == ord('m'):
            mv_config.auto_learn = not mv_config.auto_learn
            print(f"[自动学习] {'开启' if mv_config.auto_learn else '关闭'}")
    
    gesture_detector.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
