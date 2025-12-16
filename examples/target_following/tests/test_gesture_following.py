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
GESTURE_COOLDOWN_SECONDS = 3.0  # 触发后冷却秒数 (防止连续触发)

# 仅人脸匹配阈值
# 问题：不同人之间也可能有 0.55-0.65 的相似度
# 解决：提高阈值到 0.70，牺牲一些召回率换取精确率
FACE_ONLY_THRESHOLD = 0.70

# 自动学习阈值 - 只有非常高信心时才学习，避免污染
FACE_LEARN_THRESHOLD = 0.80

# 重新锁定阈值 - 从丢失状态恢复需要更高信心
RELOCK_FACE_THRESHOLD = 0.75


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
        face_threshold=0.65,      # 提高人脸阈值，防止误匹配
        body_threshold=0.60,      # 人体阈值
        fused_threshold=0.55,     # 融合阈值
        motion_weight=0.10,       # 降低运动权重
        auto_learn=True,
        learn_interval=3.0,       # 学习间隔
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
        
        # 调试日志 (每30帧输出一次)
        if frame_count % 30 == 0:
            print(f"[DEBUG] Frame {frame_count}: persons={len(persons)}, faces={len(faces)}, gesture={gesture.gesture_type.value}")
            if faces:
                for i, face in enumerate(faces):
                    print(f"        Face[{i}]: bbox={face.bbox.astype(int).tolist()}, conf={face.confidence:.2f}")
            if persons:
                for i, person in enumerate(persons):
                    print(f"        Person[{i}]: bbox={person.bbox.astype(int).tolist()}, conf={person.confidence:.2f}")
        
        # ============== 手势状态机 (持续时间检测) ==============
        current_time = time.time()
        old_state = state_machine.state
        
        # 处理手势 (需要持续 GESTURE_HOLD_DURATION 秒)
        state_changed = state_machine.process_gesture(gesture.gesture_type, current_time, debug=False)
        
        # 获取持续进度
        hold_progress = state_machine.get_gesture_hold_progress()
        
        # 状态机调试日志
        if hold_progress > 0 and frame_count % 10 == 0:
            print(f"[STATE] gesture={gesture.gesture_type.value}, hold={hold_progress*100:.0f}%, state={state_machine.state.value}")
        
        # 状态变更处理
        if state_changed:
            if state_machine.state == SystemState.TRACKING and old_state == SystemState.IDLE:
                # 启动跟随 - 优先使用人体，其次使用人脸
                nearest_person, idx = find_nearest_person(persons, frame_center)
                
                if nearest_person is not None:
                    # 有人体检测结果
                    print(f"[DEBUG] 锁定人体: bbox={nearest_person.bbox.astype(int).tolist()}")
                    view = extract_view_feature(
                        frame, nearest_person.bbox, faces, 
                        face_recognizer, enhanced_reid
                    )
                    print(f"[DEBUG] 提取特征: has_face={view.has_face}, has_body={view.part_color_hists is not None}")
                    if view.has_face and view.face_embedding is not None:
                        print(f"[DEBUG] 人脸embedding: shape={view.face_embedding.shape}, norm={np.linalg.norm(view.face_embedding):.3f}")
                    mv_recognizer.set_target(view, nearest_person.bbox)
                    lost_frames = 0
                    face_str = "有人脸" if view.has_face else "无人脸"
                    print(f"[手势启动] 目标已锁定 (人体+{face_str})")
                elif faces:
                    # 没有人体但有人脸 - 用人脸框作为临时目标
                    # 找离画面中心最近的人脸
                    min_dist = float('inf')
                    nearest_face = None
                    for face in faces:
                        fx1, fy1, fx2, fy2 = face.bbox
                        fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                        dist = (fcx - frame_center[0])**2 + (fcy - frame_center[1])**2
                        if dist < min_dist:
                            min_dist = dist
                            nearest_face = face
                    
                    if nearest_face is not None:
                        # 用人脸框扩展为伪人体框（向下扩展3倍）
                        fx1, fy1, fx2, fy2 = nearest_face.bbox
                        face_h = fy2 - fy1
                        face_w = fx2 - fx1
                        print(f"[DEBUG] 仅人脸模式: face_bbox={nearest_face.bbox.astype(int).tolist()}")
                        # 人脸大约是人体的1/7，向下扩展
                        pseudo_bbox = np.array([
                            max(0, fx1 - face_w * 0.5),
                            fy1,
                            min(w, fx2 + face_w * 0.5),
                            min(h, fy2 + face_h * 5)
                        ])
                        print(f"[DEBUG] 伪人体框: pseudo_bbox={pseudo_bbox.astype(int).tolist()}")
                        
                        view = ViewFeature(timestamp=time.time())
                        view.has_face = True
                        face_feature = face_recognizer.extract_feature(
                            frame, nearest_face.bbox, nearest_face.keypoints
                        )
                        if face_feature:
                            view.face_embedding = face_feature.embedding
                            print(f"[DEBUG] 人脸特征提取成功: embedding_shape={face_feature.embedding.shape}, norm={np.linalg.norm(face_feature.embedding):.3f}")
                        else:
                            print(f"[DEBUG] 人脸特征提取失败!")
                        
                        mv_recognizer.set_target(view, pseudo_bbox)
                        print(f"[DEBUG] 目标已设置: has_face_view={mv_recognizer.target.has_face_view if mv_recognizer.target else False}")
                        lost_frames = 0
                        print(f"[手势启动] 目标已锁定 (仅人脸，等待人体补充)")
                else:
                    # 既没有人体也没有人脸
                    state_machine.state = SystemState.IDLE
                    print("[提示] 未检测到人体或人脸，无法启动")
            
            elif state_machine.state == SystemState.IDLE and old_state == SystemState.TRACKING:
                # 停止跟随 - 只有从 TRACKING 状态才能停止
                mv_recognizer.clear_target()
                lost_frames = 0
                print("[手势停止] 跟随已停止")
        
        # ============== 目标跟踪 ==============
        target_person_idx = -1
        target_face_idx = -1  # 仅人脸匹配时的索引
        current_match_info = None  # 当前帧匹配信息，用于界面显示
        
        if state_machine.state == SystemState.TRACKING:
            matched_any = False
            
            # 调试: 显示目标信息
            if frame_count % 30 == 0 and mv_recognizer.target:
                t = mv_recognizer.target
                print(f"[DEBUG] Target: num_views={t.num_views}, has_face_view={t.has_face_view}")
                for vi, v in enumerate(t.view_features):
                    print(f"        View[{vi}]: has_face={v.has_face}, has_body={v.part_color_hists is not None}")
            
            # 1. 优先通过人体匹配
            for idx, person in enumerate(persons):
                view = extract_view_feature(
                    frame, person.bbox, faces, face_recognizer, enhanced_reid
                )
                
                is_match, similarity, method = mv_recognizer.is_same_target(
                    view, person.bbox
                )
                
                if frame_count % 30 == 0:
                    print(f"[DEBUG] Person[{idx}] match: is_match={is_match}, sim={similarity:.3f}, method={method}")
                
                if is_match:
                    matched_any = True
                    target_person_idx = idx
                    lost_frames = 0
                    
                    # 保存当前匹配信息用于显示
                    current_match_info = {
                        'type': 'person',
                        'similarity': similarity,
                        'method': method,
                        'threshold': mv_recognizer.config.fused_threshold if 'fused' in method else mv_recognizer.config.body_threshold
                    }
                    
                    # 更新跟踪
                    mv_recognizer.update_tracking(person.bbox)
                    
                    # 自动学习策略:
                    # 1. 如果目标只有人脸没有人体 -> 积极学习人体特征（补充多模态）
                    # 2. 否则用高阈值过滤
                    should_learn = False
                    target_has_body = any(v.has_body for v in mv_recognizer.target.view_features)
                    
                    if not target_has_body and view.has_body:
                        # 目标缺少人体特征，积极学习
                        should_learn = True
                        learn_reason = "补充人体特征"
                    elif similarity >= FACE_LEARN_THRESHOLD:
                        # 高置信匹配，学习新角度
                        should_learn = True
                        learn_reason = f"高置信(sim={similarity:.2f})"
                    
                    if should_learn and mv_recognizer.auto_learn(view, person.bbox, True):
                        print(f"[自动学习] {learn_reason}, 总数: {mv_recognizer.target.num_views}")
                    break
            
            # 2. 如果人体没匹配到，尝试仅通过人脸匹配（使用更严格的阈值）
            if not matched_any and faces and mv_recognizer.target and mv_recognizer.target.has_face_view:
                if frame_count % 30 == 0:
                    print(f"[DEBUG] 人体匹配失败，尝试仅人脸匹配 (阈值={FACE_ONLY_THRESHOLD})...")
                
                best_face_match = None
                best_face_sim = 0.0
                best_face_idx = -1
                best_view_idx = -1
                    
                for face_idx, face in enumerate(faces):
                    face_feature = face_recognizer.extract_feature(
                        frame, face.bbox, face.keypoints
                    )
                    if face_feature and face_feature.embedding is not None:
                        # 与目标人脸特征比较，找最高相似度
                        for vi, view in enumerate(mv_recognizer.target.view_features):
                            if view.has_face and view.face_embedding is not None:
                                sim = float(np.dot(face_feature.embedding, view.face_embedding))
                                if frame_count % 30 == 0:
                                    print(f"[DEBUG] Face[{face_idx}] vs View[{vi}]: sim={sim:.3f}, threshold={FACE_ONLY_THRESHOLD}")
                                if sim > best_face_sim:
                                    best_face_sim = sim
                                    best_face_idx = face_idx
                                    best_view_idx = vi
                
                # 使用更严格的阈值判断
                if best_face_sim >= FACE_ONLY_THRESHOLD:
                    matched_any = True
                    target_face_idx = best_face_idx
                    lost_frames = 0
                    
                    # 保存当前匹配信息用于显示
                    current_match_info = {
                        'type': 'face_only',
                        'similarity': best_face_sim,
                        'method': f'face_only (vs View[{best_view_idx}])',
                        'threshold': FACE_ONLY_THRESHOLD
                    }
                    
                    # 用人脸框更新位置
                    mv_recognizer.update_tracking(faces[best_face_idx].bbox)
                    if frame_count % 30 == 0:
                        print(f"[DEBUG] 人脸匹配成功! face_idx={best_face_idx}, sim={best_face_sim:.3f}")
                    
                    # 高相似度时允许自动学习（增加多角度视图）
                    if best_face_sim >= FACE_LEARN_THRESHOLD:
                        face_only_view = ViewFeature(timestamp=time.time())
                        face_feature = face_recognizer.extract_feature(
                            frame, faces[best_face_idx].bbox, faces[best_face_idx].keypoints
                        )
                        if face_feature:
                            face_only_view.has_face = True
                            face_only_view.face_embedding = face_feature.embedding
                            if mv_recognizer.auto_learn(face_only_view, faces[best_face_idx].bbox, True):
                                print(f"[自动学习] 新人脸视角(sim={best_face_sim:.2f}), 总数: {mv_recognizer.target.num_views}")
                elif frame_count % 30 == 0 and best_face_sim > 0:
                    print(f"[DEBUG] 人脸最高相似度 {best_face_sim:.3f} < 阈值 {FACE_ONLY_THRESHOLD}")
            
            if not matched_any:
                lost_frames += 1
                if frame_count % 30 == 0:
                    print(f"[DEBUG] 未匹配, lost_frames={lost_frames}/{max_lost_frames}")
                if lost_frames >= max_lost_frames:
                    state_machine.state = SystemState.LOST_TARGET
                    print("[目标丢失] 等待重新出现或手势停止")
        
        elif state_machine.state == SystemState.LOST_TARGET:
            # 尝试重新匹配 - 必须同时有人脸验证，或人体相似度非常高
            # 这是防止误锁定的关键！
            matched_any = False
            
            # 重新锁定的阈值要求更高
            RELOCK_BODY_THRESHOLD = 0.70  # 仅人体时需要更高相似度
            RELOCK_FUSED_THRESHOLD = 0.65  # 有人脸时可以稍低
            
            for idx, person in enumerate(persons):
                view = extract_view_feature(
                    frame, person.bbox, faces, face_recognizer, enhanced_reid
                )
                
                is_match, similarity, method = mv_recognizer.is_same_target(
                    view, person.bbox
                )
                
                # 重新锁定需要更严格的验证
                if is_match:
                    # 检查匹配类型和阈值
                    if 'fused' in method and view.has_face:
                        # 有人脸的融合匹配 - 使用较高阈值
                        if similarity >= RELOCK_FUSED_THRESHOLD:
                            state_machine.state = SystemState.TRACKING
                            target_person_idx = idx
                            lost_frames = 0
                            mv_recognizer.update_tracking(person.bbox)
                            print(f"[重新锁定] 目标已恢复 (人体+人脸, sim={similarity:.2f})")
                            matched_any = True
                            break
                    elif similarity >= RELOCK_BODY_THRESHOLD:
                        # 仅人体匹配 - 需要更高相似度
                        state_machine.state = SystemState.TRACKING
                        target_person_idx = idx
                        lost_frames = 0
                        mv_recognizer.update_tracking(person.bbox)
                        print(f"[重新锁定] 目标已恢复 (仅人体, sim={similarity:.2f})")
                        matched_any = True
                        break
            
            # 仅人脸匹配
            if not matched_any and faces and mv_recognizer.target and mv_recognizer.target.has_face_view:
                for face_idx, face in enumerate(faces):
                    face_feature = face_recognizer.extract_feature(
                        frame, face.bbox, face.keypoints
                    )
                    if face_feature and face_feature.embedding is not None:
                        for view in mv_recognizer.target.view_features:
                            if view.has_face and view.face_embedding is not None:
                                sim = float(np.dot(face_feature.embedding, view.face_embedding))
                                # 重新锁定用更高阈值，确保是同一人
                                if sim >= RELOCK_FACE_THRESHOLD:
                                    state_machine.state = SystemState.TRACKING
                                    target_face_idx = face_idx
                                    lost_frames = 0
                                    mv_recognizer.update_tracking(face.bbox)
                                    print(f"[重新锁定] 目标已恢复 (人脸, 相似度: {sim:.3f})")
                                    matched_any = True
                                    break
                        if matched_any:
                            break
        
        # ============== 绘制 ==============
        # 绘制人体框
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
        
        # 绘制人脸框 (仅人脸匹配时高亮目标人脸)
        for face_idx, face in enumerate(faces):
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            if face_idx == target_face_idx and target_person_idx < 0:
                # 仅人脸匹配的目标
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(frame, "TARGET(Face)", (fx1, fy1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif state_machine.state == SystemState.IDLE:
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)
        
        # 绘制手势指示器 (含进度条)
        draw_gesture_indicator(frame, gesture, state_machine.state, hold_progress)
        
        # 状态信息
        target_info = "None"
        if mv_recognizer.target:
            num_views = mv_recognizer.target.num_views
            # 统计有人脸和有人体的视角数量
            face_views = sum(1 for v in mv_recognizer.target.view_features if v.has_face)
            body_views = sum(1 for v in mv_recognizer.target.view_features if v.part_color_hists is not None)
            target_info = f"Views={num_views} (F:{face_views} B:{body_views})"
        
        # 匹配信息
        match_info = ""
        if current_match_info:
            sim = current_match_info['similarity']
            thresh = current_match_info['threshold']
            mtype = current_match_info['type']
            match_info = f"Match: {mtype} sim={sim:.2f} (>={thresh:.2f})"
        
        info_lines = [
            f"FPS: {fps:.1f}",
            f"State: {state_machine.state.value}",
            f"Persons: {len(persons)}, Faces: {len(faces)}",
            f"Target: {target_info}",
            match_info,
            f"Gesture: {gesture.gesture_type.value}" + (f" ({hold_progress*100:.0f}%)" if hold_progress > 0 else "")
        ]
        
        for i, line in enumerate(info_lines):
            if line:
                # 匹配信息用不同颜色
                color = (0, 255, 255) if "Match:" in line else (0, 255, 0)
                cv2.putText(frame, line, (10, 25 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
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
