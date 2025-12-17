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

# 自动学习阈值
# - 多人场景：需要人脸验证 + 高阈值（防止学习错误人脸）
# - 单人场景：可放宽
# 关键修复：0.65 太低会在多人场景学习到他人人脸，导致目标切换
FACE_LEARN_THRESHOLD = 0.72  # 人脸匹配学习阈值 (提高以防止学习错误人脸)
FACE_LEARN_THRESHOLD_MULTI = 0.78  # 多人场景下的人脸学习阈值（更严格）
BODY_LEARN_THRESHOLD = 0.68  # 人体匹配学习阈值（提高）

# 重新锁定阈值 - 从丢失状态恢复需要更高信心
RELOCK_FACE_THRESHOLD = 0.70  # 降低以便更容易重新锁定

# 连续帧确认 - 防止瞬间误匹配导致的误锁定
# 重新锁定需要连续N帧都匹配成功才确认
RELOCK_CONFIRM_FRAMES = 2  # 连续帧数要求 (从3降到2)
AUTO_LEARN_CONFIRM_FRAMES = 1  # 自动学习不需要连续帧（高置信度时直接学习）

# 视角库最大容量 - 防止特征库无限膨胀
MAX_VIEW_COUNT = 8  # 最多保存8个视角

# 人脸有效尺寸 - 小人脸embedding质量差，容易误识别
MIN_FACE_SIZE = 40  # 人脸最小边长(像素)
MIN_FACE_SIZE_FOR_LEARN = 50  # 学习时人脸最小边长(更严格)

# ============================================
# 多帧投票机制 - 避免单帧误判
# ============================================
# 连续N帧未匹配才判定丢失，防止瞬时遮挡误判
LOST_CONFIRM_FRAMES = 5  # 连续未匹配帧数才丢失 (默认 max_lost_frames=30)
# 匹配结果缓冲 - 保存最近N帧的匹配情况用于投票
MATCH_HISTORY_SIZE = 5  # 保存最近5帧匹配历史
# 运动权重增益 - 多人场景下增加运动一致性权重
MOTION_WEIGHT_MULTI_PERSON = 0.6  # 多人场景 motion 权重 (body:0.4, motion:0.6)
MOTION_WEIGHT_SINGLE_PERSON = 0.5  # 单人场景 motion 权重 (body:0.5, motion:0.5)

# 侧脸容忍度 - 侧脸角度下人脸embedding差异大，需要更信任运动连续性
# 当运动连续性极高时（motion > 0.95），可以容忍较低的人脸相似度
MOTION_TRUST_THRESHOLD = 0.95  # 运动连续性信任阈值
FACE_SIDE_VIEW_MIN = 0.35  # 侧脸最低接受阈值（配合高运动连续性）


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


def find_person_with_gesture(persons: list, hand_bbox: np.ndarray):
    """找到做手势的那个人（优先手势在人体框内，其次找最近的人体）"""
    if not persons or hand_bbox is None:
        return None, -1
    
    hx1, hy1, hx2, hy2 = hand_bbox
    hand_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
    
    best_person = None
    best_idx = -1
    best_overlap = 0.0
    
    # 策略1：优先找手势中心在人体框内的
    for i, person in enumerate(persons):
        px1, py1, px2, py2 = person.bbox
        
        # 检查手势中心是否在人体框内
        if px1 <= hand_center[0] <= px2 and py1 <= hand_center[1] <= py2:
            # 计算重叠程度（手势框与人体框的IoU）
            inter_x1 = max(hx1, px1)
            inter_y1 = max(hy1, py1)
            inter_x2 = min(hx2, px2)
            inter_y2 = min(hy2, py2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                hand_area = (hx2 - hx1) * (hy2 - hy1)
                overlap = inter_area / hand_area if hand_area > 0 else 0
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_person = person
                    best_idx = i
    
    # 策略2：如果没找到完全包含的，找手势框与人体框边缘最近的
    # 这处理手伸出身体做手势的情况
    if best_person is None:
        min_edge_dist = float('inf')
        for i, person in enumerate(persons):
            px1, py1, px2, py2 = person.bbox
            
            # 计算手势中心到人体框边缘的最短距离
            # 如果手势在框内，距离为0
            dx = max(px1 - hand_center[0], 0, hand_center[0] - px2)
            dy = max(py1 - hand_center[1], 0, hand_center[1] - py2)
            edge_dist = (dx**2 + dy**2) ** 0.5
            
            # 额外检查：手势应该在人体的合理延伸范围内（宽度的50%）
            person_width = px2 - px1
            max_extend = person_width * 0.5
            
            if edge_dist < min_edge_dist and edge_dist < max_extend:
                min_edge_dist = edge_dist
                best_person = person
                best_idx = i
    
    return best_person, best_idx


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
        face_threshold=0.60,      # 人脸阈值（适度降低以容忍侧脸）
        body_threshold=0.58,      # 人体阈值（适度降低以提高连续性）
        fused_threshold=0.52,     # 融合阈值（适度降低）
        motion_weight=0.15,       # 提高运动权重（侧脸时依赖运动连续性）
        auto_learn=True,
        learn_interval=3.0,       # 学习间隔
        smooth_window=5,
        confirm_threshold=3,
        part_weights=[0.05, 0.12, 0.20, 0.20, 0.25, 0.18],
        max_views=MAX_VIEW_COUNT  # 限制视角数量
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
    
    # 创建可调整大小的窗口
    window_name = "Gesture-Controlled Following"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)  # 默认窗口大小
    
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
    
    # 连续帧确认计数器
    relock_confirm_count = 0  # 重新锁定连续匹配帧数
    relock_candidate_idx = -1  # 当前重新锁定候选人索引
    auto_learn_confirm_count = 0  # 自动学习连续匹配帧数
    auto_learn_candidate_view = None  # 待学习的视角
    
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
                # 启动跟随 - 优先锁定做手势的人，其次用最近的人
                target_person = None
                target_idx = -1
                
                # 1. 优先找做手势的那个人
                if gesture.hand_bbox is not None:
                    print(f"[DEBUG] 手势框: {gesture.hand_bbox.astype(int).tolist()}")
                    for pi, p in enumerate(persons):
                        px1, py1, px2, py2 = p.bbox.astype(int)
                        hc = ((gesture.hand_bbox[0] + gesture.hand_bbox[2]) / 2,
                              (gesture.hand_bbox[1] + gesture.hand_bbox[3]) / 2)
                        in_box = px1 <= hc[0] <= px2 and py1 <= hc[1] <= py2
                        print(f"[DEBUG] Person[{pi}] bbox: [{px1}, {py1}, {px2}, {py2}], 手势在框内: {in_box}")
                    gesture_person, gesture_idx = find_person_with_gesture(persons, gesture.hand_bbox)
                    if gesture_person is not None:
                        target_person = gesture_person
                        target_idx = gesture_idx
                        print(f"[DEBUG] 锁定做手势的人 Person[{target_idx}]")
                    else:
                        print(f"[DEBUG] 手势未落在任何人体框内或附近！")
                
                # 2. 如果没找到，使用离画面中心最近的人
                if target_person is None:
                    target_person, target_idx = find_nearest_person(persons, frame_center)
                    if target_person is not None:
                        print(f"[DEBUG] 无法定位手势所在人体，使用最近的人 Person[{target_idx}]")
                
                if target_person is not None:
                    # 有人体检测结果
                    print(f"[DEBUG] 锁定人体: bbox={target_person.bbox.astype(int).tolist()}")
                    view = extract_view_feature(
                        frame, target_person.bbox, faces, 
                        face_recognizer, enhanced_reid
                    )
                    print(f"[DEBUG] 提取特征: has_face={view.has_face}, has_body={view.part_color_hists is not None}")
                    if view.has_face and view.face_embedding is not None:
                        print(f"[DEBUG] 人脸embedding: shape={view.face_embedding.shape}, norm={np.linalg.norm(view.face_embedding):.3f}")
                    mv_recognizer.set_target(view, target_person.bbox)
                    mv_recognizer.clear_match_history()  # 新目标，清空历史
                    lost_frames = 0
                    face_str = "有人脸" if view.has_face else "无人脸"
                    print(f"[手势启动] 目标已锁定 (人体+{face_str})")
                elif faces:
                    # 没有人体但有人脸 - 用人脸框作为临时目标
                    # 优先找离手势最近的人脸，其次找离画面中心最近的人脸
                    target_face = None
                    
                    if gesture.hand_bbox is not None:
                        # 找离手势最近的人脸
                        hx1, hy1, hx2, hy2 = gesture.hand_bbox
                        hand_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
                        min_dist = float('inf')
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox
                            fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                            dist = (fcx - hand_center[0])**2 + (fcy - hand_center[1])**2
                            if dist < min_dist:
                                min_dist = dist
                                target_face = face
                        if target_face is not None:
                            print(f"[DEBUG] 仅人脸模式: 使用离手势最近的人脸")
                    
                    if target_face is None:
                        # 找离画面中心最近的人脸
                        min_dist = float('inf')
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox
                            fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                            dist = (fcx - frame_center[0])**2 + (fcy - frame_center[1])**2
                            if dist < min_dist:
                                min_dist = dist
                                target_face = face
                    
                    if target_face is not None:
                        # 用人脸框扩展为伪人体框（向下扩展3倍）
                        fx1, fy1, fx2, fy2 = target_face.bbox
                        face_h = fy2 - fy1
                        face_w = fx2 - fx1
                        print(f"[DEBUG] 仅人脸模式: face_bbox={target_face.bbox.astype(int).tolist()}")
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
                            frame, target_face.bbox, target_face.keypoints
                        )
                        if face_feature:
                            view.face_embedding = face_feature.embedding
                            print(f"[DEBUG] 人脸特征提取成功: embedding_shape={face_feature.embedding.shape}, norm={np.linalg.norm(face_feature.embedding):.3f}")
                        else:
                            print(f"[DEBUG] 人脸特征提取失败!")
                        
                        mv_recognizer.set_target(view, pseudo_bbox)
                        mv_recognizer.clear_match_history()  # 新目标，清空历史
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
                mv_recognizer.clear_match_history()  # 清空历史
                lost_frames = 0
                print("[手势停止] 跟随已停止")
        
        # ============== 目标跟踪 ==============
        target_person_idx = -1
        target_face_idx = -1  # 仅人脸匹配时的索引
        current_match_info = None  # 当前帧匹配信息，用于界面显示
        
        # ============== 场景判断 ==============
        # 关键：多人场景应该用 max(persons, faces) 判断，而不是仅看 persons
        # 场景分类：
        #   单人: persons<=1 且 faces<=1
        #   多人: persons>1 或 faces>1 (只要有一方>1就是多人风险场景)
        num_persons = len(persons)
        num_faces = len(faces)
        is_multi_person_scene = num_persons > 1 or num_faces > 1
        is_single_person_scene = not is_multi_person_scene
        
        if state_machine.state == SystemState.TRACKING:
            matched_any = False
            
            # 调试: 显示目标信息和场景类型
            if frame_count % 30 == 0:
                scene_type = "多人" if is_multi_person_scene else "单人"
                print(f"[DEBUG] 场景: {scene_type} (persons={num_persons}, faces={num_faces})")
                if mv_recognizer.target:
                    t = mv_recognizer.target
                    print(f"[DEBUG] Target: num_views={t.num_views}, has_face_view={t.has_face_view}")
                    for vi, v in enumerate(t.view_features):
                        print(f"        View[{vi}]: has_face={v.has_face}, has_body={v.part_color_hists is not None}")
            
            # 1. 通过人体匹配 - 使用"最佳匹配"策略（而不是"第一个匹配"）
            # 收集所有候选匹配，选择最高分的
            all_person_matches = []  # [(idx, similarity, method, view, face_in_person, face_verified, face_sim, body_sim)]
            
            # 关键保护：如果目标有人脸特征，候选人也有人脸时必须通过人脸验证
            target_has_face = mv_recognizer.target and mv_recognizer.target.has_face_view
            target_has_body = mv_recognizer.target and any(v.has_body for v in mv_recognizer.target.view_features)
            
            # =====================================================================
            # 场景×目标状态 分析矩阵
            # =====================================================================
            # 
            # 画面内容:
            #   单人场景: persons<=1 且 faces<=1
            #   多人场景: persons>1 或 faces>1
            #
            # 目标在画面中的状态:
            #   A: 目标以 人脸+人体 出现
            #   B: 目标仅以 人脸 出现（人体被遮挡或太远）
            #   C: 目标仅以 人体 出现（背对/低头/遮挡脸）
            #   D: 目标不在画面中
            #
            # 处理策略:
            # ┌──────────────────────────────────────────────────────────────────┐
            # │ Step1: 遍历人体匹配 → 覆盖状态 A, C                              │
            # │   - A: 人脸验证通过 → 确认匹配                                   │
            # │   - C: 无人脸可验证，使用body匹配                                │
            # │                                                                  │
            # │ Step2: 仅人脸匹配 → 覆盖状态 B                                   │
            # │   - 无人体匹配成功时，尝试独立人脸匹配                           │
            # │                                                                  │
            # │ 未匹配 → 状态 D 或匹配失败                                       │
            # │   - 累积 lost_frames → 触发 LOST_TARGET                          │
            # └──────────────────────────────────────────────────────────────────┘
            #
            # 关键风险: 目标不在(D) 但误匹配到衣着相似的他人
            #
            # 保护措施汇总:
            # ┌────────────────────┬────────────────────────────────────────────┐
            # │ 场景               │ 保护策略                                    │
            # ├────────────────────┼────────────────────────────────────────────┤
            # │ 多人+目标有脸+     │ face_sim < FACE_REJECT → 拒绝              │
            # │ 候选有脸           │ face_sim < FACE_UNCERTAIN 且               │
            # │                    │ body_sim < HIGH_BODY → 拒绝                 │
            # ├────────────────────┼────────────────────────────────────────────┤
            # │ 多人+目标有脸+     │ body_sim < BACK_VIEW_BODY → 拒绝           │
            # │ 候选无脸           │ (可能是他人背面)                            │
            # ├────────────────────┼────────────────────────────────────────────┤
            # │ 单人+目标有脸+     │ 同上保护逻辑                                │
            # │ 候选有脸           │ 因为那个"单人"可能不是目标                  │
            # ├────────────────────┼────────────────────────────────────────────┤
            # │ 单人+目标有脸+     │ 如有其他人脸: body_sim < 0.65 → 拒绝       │
            # │ 候选无脸           │ 无其他人脸: 使用标准阈值                    │
            # ├────────────────────┼────────────────────────────────────────────┤
            # │ 仅人脸匹配         │ 多人脸: +0.05 阈值惩罚                      │
            # │                    │ 远处人脸: +0.10 阈值惩罚                    │
            # │                    │ 人脸在不匹配人体框内: 跳过                  │
            # └────────────────────┴────────────────────────────────────────────┘
            # =====================================================================
            
            for idx, person in enumerate(persons):
                view = extract_view_feature(
                    frame, person.bbox, faces, face_recognizer, enhanced_reid
                )
                
                # 使用 return_details=True 获取详细信息（包含 face_sim）
                result = mv_recognizer.is_same_target(
                    view, person.bbox, return_details=True
                )
                # 返回值是 (is_match, similarity, method, details)
                is_match = result[0]
                similarity = result[1]
                method = result[2]
                details = result[3] if len(result) > 3 else {}
                
                # 提取详细相似度
                face_sim = details.get('face_sim')  # 可能为 None（候选人没有人脸）
                body_sim = details.get('body_sim', 0.0)
                
                # 提取运动连续性分数
                motion_score = details.get('motion_sim', 0.0)
                if 'M:' in method:
                    try:
                        motion_str = method.split('M:')[1].split(')')[0].split(' ')[0]
                        motion_score = float(motion_str)
                    except:
                        pass
                
                if frame_count % 30 == 0:
                    face_str = f"F:{face_sim:.2f}" if face_sim is not None else "F:None"
                    print(f"[DEBUG] Person[{idx}] match: is_match={is_match}, sim={similarity:.3f}, {face_str}, B:{body_sim:.2f}, M:{motion_score:.2f}, method={method}")
                
                if is_match:
                    face_in_person = view.has_face and view.face_embedding is not None
                    
                    # ============================================
                    # 简化的匹配逻辑（防止误跟踪他人）
                    # ============================================
                    # 核心思路:
                    #   1. 人脸 > 阈值 且 尺寸够大 → 靠人脸判断
                    #   2. 人脸 < 阈值 或 尺寸太小 → 靠 motion + body 判断
                    #   3. motion + body 都低 → 目标丢失
                    # ============================================
                    
                    FACE_MATCH_THRESHOLD = 0.55  # 人脸匹配阈值
                    BODY_MOTION_THRESHOLD = 0.65  # body + motion 综合阈值
                    MULTI_PERSON_BODY_THRESHOLD = 0.70  # 多人场景下仅body匹配的阈值
                    
                    # 检查人脸尺寸是否足够大
                    face_size_valid = False
                    current_face_size = 0
                    for face in faces:
                        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                        fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                        px1, py1, px2, py2 = person.bbox.astype(int)
                        if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                            face_w = fx2 - fx1
                            face_h = fy2 - fy1
                            current_face_size = min(face_w, face_h)
                            face_size_valid = current_face_size >= MIN_FACE_SIZE
                            break
                    
                    # 计算 body + motion 综合分数
                    # 多人场景下增加 motion 权重，因为运动轨迹更可靠
                    if is_multi_person_scene:
                        motion_weight = MOTION_WEIGHT_MULTI_PERSON
                    else:
                        motion_weight = MOTION_WEIGHT_SINGLE_PERSON
                    body_weight = 1.0 - motion_weight
                    body_motion_score = body_sim * body_weight + motion_score * motion_weight
                    
                    # 判断匹配类型 (人脸有效 = 相似度高 且 尺寸够大)
                    face_matched = (face_sim is not None and 
                                    face_sim >= FACE_MATCH_THRESHOLD and 
                                    face_size_valid)
                    body_motion_matched = body_motion_score >= BODY_MOTION_THRESHOLD
                    
                    if frame_count % 30 == 0 and face_sim is not None:
                        print(f"[DEBUG] Person[{idx}] face_size={current_face_size}px, valid={face_size_valid}, face_matched={face_matched}")
                    
                    # 决策逻辑
                    accept = False
                    match_type = ""
                    
                    if face_matched:
                        # Case 1: 人脸匹配 → 直接信任
                        accept = True
                        match_type = "face"
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] Person[{idx}] 人脸匹配通过 (F:{face_sim:.2f}>={FACE_MATCH_THRESHOLD})")
                    elif target_has_face and face_in_person and face_sim is not None and face_sim < 0.30:
                        # Case 2: 目标有脸 + 候选有脸 + 人脸明确不匹配 → 多人场景拒绝，单人场景看body+motion
                        if is_multi_person_scene:
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] Person[{idx}] 多人场景人脸明确不匹配(F:{face_sim:.2f}<0.30), 拒绝")
                            accept = False
                        elif body_motion_matched:
                            accept = True
                            match_type = "body_motion"
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] Person[{idx}] 单人场景人脸低(F:{face_sim:.2f})但body+motion高({body_motion_score:.2f}), 通过")
                        else:
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] Person[{idx}] 单人场景人脸低且body+motion不足({body_motion_score:.2f}<{BODY_MOTION_THRESHOLD}), 拒绝")
                            accept = False
                    elif body_motion_matched:
                        # Case 3: 人脸不够但 body+motion 够 → 通过
                        # 多人场景需要更高的 body 阈值
                        if is_multi_person_scene and target_has_face and body_sim < MULTI_PERSON_BODY_THRESHOLD:
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] Person[{idx}] 多人场景无人脸验证且body不足({body_sim:.2f}<{MULTI_PERSON_BODY_THRESHOLD}), 拒绝")
                            accept = False
                        else:
                            accept = True
                            match_type = "body_motion"
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] Person[{idx}] body+motion匹配通过 (B:{body_sim:.2f}+M:{motion_score:.2f}={body_motion_score:.2f})")
                    else:
                        # Case 4: 人脸和body+motion都不够 → 拒绝
                        if frame_count % 30 == 0:
                            face_str = f"F:{face_sim:.2f}" if face_sim is not None else "F:None"
                            print(f"[DEBUG] Person[{idx}] 人脸和body+motion都不足 ({face_str}, BM:{body_motion_score:.2f}), 拒绝")
                        accept = False
                    
                    if accept:
                        # tuple: (idx, similarity, method, view, face_in_person, face_matched, face_sim, body_sim, motion_score, match_type)
                        all_person_matches.append((idx, similarity, method, view, face_in_person, face_matched, face_sim, body_sim, motion_score, match_type))
            
            # 选择最佳匹配
            if all_person_matches:
                # 策略: 
                #   1. 优先选人脸匹配的（身份最可靠）
                #   2. 人脸匹配中优先选 motion 高的（轨迹最一致）
                #   3. 其次选 body+motion 匹配的
                #   4. body+motion 中多人场景优先选 motion 高的
                # tuple: (idx, similarity, method, view, face_in_person, face_matched, face_sim, body_sim, motion_score, match_type)
                matches_by_face = [m for m in all_person_matches if m[9] == "face"]  # m[9] = match_type
                matches_by_body_motion = [m for m in all_person_matches if m[9] == "body_motion"]
                
                best_match = None
                if matches_by_face:
                    # 人脸匹配中，优先选 motion 高的（轨迹一致性）
                    # 排序依据: face_sim * 0.6 + motion * 0.4
                    best_match = max(matches_by_face, key=lambda x: (x[6] if x[6] is not None else 0) * 0.6 + x[8] * 0.4)
                    if frame_count % 30 == 0 and len(all_person_matches) > 1:
                        print(f"[DEBUG] 选择人脸匹配 Person[{best_match[0]}] (F:{best_match[6]:.2f}, M:{best_match[8]:.2f}, 共{len(all_person_matches)}候选)")
                elif matches_by_body_motion:
                    # body+motion 匹配中，多人场景强调 motion，单人场景平衡
                    if is_multi_person_scene:
                        # 多人: motion 优先（轨迹一致最重要）
                        best_match = max(matches_by_body_motion, key=lambda x: x[8] * 0.7 + x[7] * 0.3)
                    else:
                        # 单人: 平衡 body 和 motion
                        best_match = max(matches_by_body_motion, key=lambda x: x[7] * 0.5 + x[8] * 0.5)
                    if frame_count % 30 == 0:
                        print(f"[DEBUG] 选择body+motion匹配 Person[{best_match[0]}], B:{best_match[7]:.2f}, M:{best_match[8]:.2f}")
                
                if best_match:
                    # 解包: (idx, similarity, method, view, face_in_person, face_matched, face_sim, body_sim, motion_score, match_type)
                    idx, similarity, method, view, face_in_person, face_matched, match_face_sim, match_body_sim, match_motion_score, match_type = best_match
                    matched_any = True
                    target_person_idx = idx
                    lost_frames = 0
                    
                    # 保存当前匹配信息用于显示
                    current_match_info = {
                        'type': 'person',
                        'similarity': similarity,
                        'method': method,
                        'match_type': match_type,  # "face" or "body_motion"
                        'threshold': FACE_MATCH_THRESHOLD if match_type == "face" else BODY_MOTION_THRESHOLD
                    }
                    
                    # 更新跟踪
                    mv_recognizer.update_tracking(persons[idx].bbox)
                    
                    # ============================================
                    # 简化的自动学习策略
                    # ============================================
                    # 核心原则：
                    #   1. 人脸匹配 + body不匹配但motion+body高 → 学习body（前提：人脸在人体框内）
                    #   2. motion+body匹配 + 人脸低但>某值 → 学习人脸（前提：人脸在人体框内）
                    #   3. 关键约束：有人脸+有人体时，学习必须保证人脸在人体框内
                    # ============================================
                    
                    should_learn = False
                    learn_what = ""  # "body" or "face" or "both"
                    learn_reason = ""
                    
                    target_has_body = (mv_recognizer.target is not None and 
                                       any(v.has_body for v in mv_recognizer.target.view_features))
                    
                    # 容量检查：视角库已满时停止学习
                    current_view_count = mv_recognizer.target.num_views if mv_recognizer.target else 0
                    if current_view_count >= MAX_VIEW_COUNT:
                        if frame_count % 60 == 0:
                            print(f"[DEBUG] 视角库已满({current_view_count}>={MAX_VIEW_COUNT})，停止学习")
                        should_learn = False
                    # 多人场景 + 没有人脸匹配 = 禁止学习
                    elif is_multi_person_scene and match_type != "face":
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] 多人场景无人脸匹配，禁止学习")
                        should_learn = False
                    else:
                        # 提取匹配信息
                        match_face_sim = best_match[6] if best_match[6] is not None else 0.0
                        match_body_sim = best_match[7]
                        match_motion = best_match[8]
                        body_motion_combined = match_body_sim * 0.5 + match_motion * 0.5
                        
                        # 学习阈值
                        FACE_LEARN_THRESHOLD_LOCAL = 0.65  # 人脸学习阈值
                        BODY_MOTION_LEARN_THRESHOLD = 0.70  # body+motion 学习阈值
                        FACE_MIN_FOR_BODY_LEARN = 0.50  # 学习body时人脸的最低要求
                        
                        # 检查当前人脸尺寸是否足够大（用于学习）
                        current_face_size_for_learn = 0
                        face_size_ok_for_learn = False
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                            fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                            px1, py1, px2, py2 = persons[idx].bbox.astype(int)
                            if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                                face_w = fx2 - fx1
                                face_h = fy2 - fy1
                                current_face_size_for_learn = min(face_w, face_h)
                                face_size_ok_for_learn = current_face_size_for_learn >= MIN_FACE_SIZE_FOR_LEARN
                                break
                        
                        # Case 1: 人脸匹配通过 → 可以学习body
                        if match_type == "face":
                            # 人脸匹配 + body+motion高 → 学习body（如果目标还没有body或需要更新）
                            if body_motion_combined >= BODY_MOTION_LEARN_THRESHOLD:
                                # 关键约束：人脸必须在人体框内！
                                if face_in_person:
                                    should_learn = True
                                    learn_what = "body"
                                    learn_reason = f"人脸匹配(F:{match_face_sim:.2f})学习body(BM:{body_motion_combined:.2f})"
                                else:
                                    if frame_count % 30 == 0:
                                        print(f"[DEBUG] 人脸不在人体框内，不学习body")
                            elif match_face_sim >= FACE_LEARN_THRESHOLD_LOCAL and face_size_ok_for_learn:
                                # 人脸够高 + 尺寸够大 → 直接学习当前视角
                                should_learn = True
                                learn_what = "face"
                                learn_reason = f"人脸高置信(F:{match_face_sim:.2f}, size={current_face_size_for_learn}px)"
                        
                        # Case 2: body+motion匹配通过 → 可以学习人脸
                        elif match_type == "body_motion":
                            # body+motion匹配 + 有人脸且>某值 + 人脸尺寸够大 → 学习人脸
                            if face_in_person and match_face_sim >= FACE_MIN_FOR_BODY_LEARN and face_size_ok_for_learn:
                                # 关键约束：人脸必须在人体框内 且 尺寸足够大！
                                should_learn = True
                                learn_what = "face"
                                learn_reason = f"body+motion匹配(BM:{body_motion_combined:.2f})学习face(F:{match_face_sim:.2f}, size={current_face_size_for_learn}px)"
                            elif not face_in_person and body_motion_combined >= BODY_MOTION_LEARN_THRESHOLD:
                                # 纯背面/侧面，学习body视角
                                should_learn = True
                                learn_what = "body"
                                learn_reason = f"背面匹配(BM:{body_motion_combined:.2f})"
                    
                    if should_learn:
                        learned, op_info = mv_recognizer.auto_learn(view, persons[idx].bbox, True)
                        if learned:
                            print(f"[自动学习] {learn_reason} -> {op_info}")
            
            # 2. 如果人体没匹配到，尝试仅通过人脸匹配（使用更严格的阈值）
            # ============================================
            # 这里处理目标状态 B: 目标仅以人脸出现
            # 场景包括:
            #   - 1.2: 画面只有人脸（目标远处/被遮挡）
            #   - 2.1-B/2.2-B: 多人场景，目标人体被遮挡只露脸
            #   - 2.3: 多个远处人脸，无人体
            # ============================================
            if not matched_any and faces and mv_recognizer.target and mv_recognizer.target.has_face_view:
                if frame_count % 30 == 0:
                    print(f"[DEBUG] 人体匹配失败，尝试仅人脸匹配 (阈值={FACE_ONLY_THRESHOLD})...")
                
                best_face_match = None
                best_face_sim = 0.0
                best_face_idx = -1
                best_view_idx = -1
                
                # 多人脸场景需要更严格的阈值
                multi_face_penalty = 0.05 if num_faces > 1 else 0.0
                    
                for face_idx, face in enumerate(faces):
                    fx1, fy1, fx2, fy2 = face.bbox
                    fc_x, fc_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                    
                    # 检查人脸是否在某个人体框内
                    face_in_any_person = False
                    
                    if len(persons) > 0:
                        for p_idx, person in enumerate(persons):
                            px1, py1, px2, py2 = person.bbox
                            if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                                face_in_any_person = True
                                break
                    
                    # 情况1: 多人场景，人脸在不匹配的人体框内 → 跳过（属于别人）
                    # 关键：这是场景 2.1-B/2.2-B 的保护
                    if num_persons > 1 and face_in_any_person:
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] Face[{face_idx}] 在不匹配的人体框内(多人场景)，跳过")
                        continue
                    
                    # 情况2: 有人体但人脸不在任何人体框内 → 远处的人脸，使用更严格阈值
                    # 场景：目标背对镜头（有人体无脸），远处有其他人的脸（有脸无人体）
                    is_distant_face = num_persons > 0 and not face_in_any_person
                    current_threshold = FACE_ONLY_THRESHOLD + 0.1 + multi_face_penalty if is_distant_face else FACE_ONLY_THRESHOLD + multi_face_penalty
                    
                    if is_distant_face and frame_count % 30 == 0:
                        print(f"[DEBUG] Face[{face_idx}] 不在任何人体框内(远处人脸)，使用更高阈值={current_threshold:.2f}")
                    
                    face_feature = face_recognizer.extract_feature(
                        frame, face.bbox, face.keypoints
                    )
                    if face_feature and face_feature.embedding is not None:
                        # 与目标人脸特征比较，找最高相似度
                        for vi, view in enumerate(mv_recognizer.target.view_features):
                            if view.has_face and view.face_embedding is not None:
                                sim = float(np.dot(face_feature.embedding, view.face_embedding))
                                if frame_count % 30 == 0:
                                    print(f"[DEBUG] Face[{face_idx}] vs View[{vi}]: sim={sim:.3f}, threshold={current_threshold:.2f}")
                                # 只有超过当前阈值才记录为候选
                                if sim >= current_threshold and sim > best_face_sim:
                                    best_face_sim = sim
                                    best_face_idx = face_idx
                                    best_view_idx = vi
                
                # 使用更严格的阈值判断（已在上面的循环中过滤）
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
                    
                    # 仅人脸匹配时的自动学习 - 更严格的条件
                    # 只有单人场景+高人脸相似度才学习，避免多人场景误学习
                    # 使用外层定义的 is_single_person_scene
                    if best_face_sim >= 0.80 and is_single_person_scene:
                        face_only_view = ViewFeature(timestamp=time.time())
                        face_feature = face_recognizer.extract_feature(
                            frame, faces[best_face_idx].bbox, faces[best_face_idx].keypoints
                        )
                        if face_feature:
                            face_only_view.has_face = True
                            face_only_view.face_embedding = face_feature.embedding
                            learned, op_info = mv_recognizer.auto_learn(face_only_view, faces[best_face_idx].bbox, True)
                            if learned:
                                print(f"[自动学习] 仅人脸(sim={best_face_sim:.2f}) -> {op_info}")
                    elif frame_count % 30 == 0 and best_face_sim >= 0.70:
                        reason = "多人场景" if not is_single_person_scene else f"相似度不足({best_face_sim:.2f}<0.80)"
                        print(f"[DEBUG] 仅人脸匹配不学习: {reason}")
                elif frame_count % 30 == 0 and best_face_sim > 0:
                    print(f"[DEBUG] 人脸最高相似度 {best_face_sim:.3f} < 阈值 {FACE_ONLY_THRESHOLD}")
            
            if not matched_any:
                lost_frames += 1
                # 清空匹配历史，防止误匹配
                mv_recognizer.clear_match_history()
                if frame_count % 30 == 0:
                    print(f"[DEBUG] 未匹配, lost_frames={lost_frames}/{max_lost_frames}")
                if lost_frames >= max_lost_frames:
                    state_machine.state = SystemState.LOST_TARGET
                    print("[目标丢失] 等待重新出现或手势停止")
        
        elif state_machine.state == SystemState.LOST_TARGET:
            # 尝试重新匹配 - 使用最佳匹配策略 + 连续帧确认
            # 关键：LOST_TARGET 重新锁定需要连续N帧匹配成功才确认
            
            # 重新锁定的阈值 - 适度降低以提高可用性
            RELOCK_BODY_THRESHOLD = 0.75  # 仅人体时的阈值
            RELOCK_FUSED_THRESHOLD = 0.65  # 有人脸时的综合阈值
            RELOCK_FACE_SIM_THRESHOLD = 0.55  # 人脸相似度下限
            
            # 多人场景下，必须有人脸验证才能重新锁定
            # 注意：LOST_TARGET 状态需要重新计算场景类型
            relock_is_multi_person = len(persons) > 1 or len(faces) > 1
            require_face_for_relock = relock_is_multi_person or (mv_recognizer.target and mv_recognizer.target.has_face_view)
            
            # 当前帧最佳匹配
            current_best_match = None
            current_best_idx = -1
            
            for idx, person in enumerate(persons):
                view = extract_view_feature(
                    frame, person.bbox, faces, face_recognizer, enhanced_reid
                )
                
                # 使用 return_details=True 获取详细信息
                result = mv_recognizer.is_same_target(
                    view, person.bbox, return_details=True
                )
                # 返回值是 (is_match, similarity, method, details)
                is_match = result[0]
                similarity = result[1]
                method = result[2]
                details = result[3] if len(result) > 3 else {}
                
                # 重新锁定需要更严格的验证
                if is_match:
                    face_in_person = view.has_face and view.face_embedding is not None
                    
                    # 从 details 中获取人脸相似度
                    face_sim = details.get('face_sim', 0.0) if details else 0.0
                    
                    # 检查是否满足阈值要求
                    if ('fused' in method or 'face_priority' in method) and face_in_person:
                        # 有人脸验证：检查人脸相似度是否足够高
                        if similarity >= RELOCK_FUSED_THRESHOLD and face_sim >= RELOCK_FACE_SIM_THRESHOLD:
                            if current_best_match is None or similarity > current_best_match[1]:
                                current_best_match = (idx, similarity, method, view, True, face_sim)
                                current_best_idx = idx
                    elif not require_face_for_relock and similarity >= RELOCK_BODY_THRESHOLD:
                        # 仅人体匹配：只在单人场景且目标没有人脸特征时允许
                        if current_best_match is None or similarity > current_best_match[1]:
                            current_best_match = (idx, similarity, method, view, False, 0.0)
                            current_best_idx = idx
            
            # 连续帧确认机制
            if current_best_match:
                idx, similarity, method, view, has_face, face_sim = current_best_match
                
                # 检查是否与上一帧候选人相同
                if current_best_idx == relock_candidate_idx:
                    relock_confirm_count += 1
                else:
                    # 候选人变化，重新计数
                    relock_candidate_idx = current_best_idx
                    relock_confirm_count = 1
                
                if frame_count % 30 == 0:
                    print(f"[DEBUG] 重新锁定候选: Person[{idx}], sim={similarity:.2f}, 连续帧={relock_confirm_count}/{RELOCK_CONFIRM_FRAMES}")
                
                # 达到连续帧要求，确认重新锁定
                if relock_confirm_count >= RELOCK_CONFIRM_FRAMES:
                    state_machine.state = SystemState.TRACKING
                    target_person_idx = idx
                    lost_frames = 0
                    relock_confirm_count = 0
                    relock_candidate_idx = -1
                    mv_recognizer.update_tracking(persons[idx].bbox)
                    relock_type = "人体+人脸" if has_face else "仅人体"
                    if has_face:
                        print(f"[重新锁定] 目标已恢复 ({relock_type}, sim={similarity:.2f}, face={face_sim:.2f}, 连续确认)")
                    else:
                        print(f"[重新锁定] 目标已恢复 ({relock_type}, sim={similarity:.2f}, 连续确认)")
            else:
                # 无匹配，重置连续帧计数
                if relock_confirm_count > 0:
                    relock_confirm_count = 0
                    relock_candidate_idx = -1
                    if frame_count % 30 == 0:
                        print(f"[DEBUG] 重新锁定候选丢失，重置计数")
            
            # 禁用仅人脸重新锁定 - 太容易误识别远处的相似人脸
            # 只有当人脸在人体框内时才能通过人体+人脸联合匹配来锁定
            # 原因：仅人脸匹配缺少位置、身体特征等关联信息，容易误匹配
            # if not matched_any and faces and mv_recognizer.target and mv_recognizer.target.has_face_view:
            #     for face_idx, face in enumerate(faces):
            #         ...
        
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
        
        # 绘制人脸框 - 使用之前匹配过程中的结果，避免重复计算
        # target_face_idx 是仅人脸匹配时确定的目标人脸
        # 对于有人体匹配的情况，只有 face_in_person=True 且通过 face_priority 验证的才标记为目标
        for face_idx, face in enumerate(faces):
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            
            # 判断是否为目标人脸
            # 关键：不能只看是否在目标人体框内，必须是匹配过程中验证过的
            # 使用 current_match_info 来判断是否经过了人脸验证
            is_target_face = False
            
            if target_person_idx >= 0 and target_person_idx < len(persons):
                px1, py1, px2, py2 = persons[target_person_idx].bbox
                fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                    # 人脸在目标人体框内
                    # 只有当匹配方法包含 face_priority 或 fused 时，才表示人脸已验证
                    if current_match_info:
                        method = current_match_info.get('method', '')
                        if 'face_priority' in method or 'fused' in method:
                            # 人脸已经通过验证
                            is_target_face = True
                        # 否则是纯人体匹配，不能确定人脸是否属于目标
                    # 如果只有一个人脸在人体框内，也可以认为是目标人脸
                    elif len([f for f in faces if px1 <= (f.bbox[0]+f.bbox[2])//2 <= px2 and py1 <= (f.bbox[1]+f.bbox[3])//2 <= py2]) == 1:
                        is_target_face = True
            
            if face_idx == target_face_idx and target_person_idx < 0:
                # 仅人脸匹配的目标
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(frame, "TARGET(Face)", (fx1, fy1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif is_target_face:
                # 目标人体内的人脸 - 用绿色高亮
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            elif state_machine.state == SystemState.IDLE:
                # 空闲状态显示所有人脸
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)
            else:
                # 跟踪状态显示非目标人脸（淡色）
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (128, 128, 128), 1)
        
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
        
        cv2.imshow(window_name, frame)
        
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
