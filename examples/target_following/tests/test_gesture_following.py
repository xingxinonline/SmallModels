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

# ============================================
# 人脸质量状态定义（核心状态机）
# ============================================
# 人脸状态分为三级：稳定 / 不稳定 / 丢失
# 不同状态使用不同的匹配策略

# 人脸稳定状态阈值
FACE_STABLE_CONF = 0.70      # 置信度 >= 0.70
FACE_STABLE_SIZE = 64        # 尺寸 >= 64px
FACE_STABLE_SIM = 0.60       # 相似度 >= 0.60
FACE_STABLE_FRAMES = 3       # 连续帧 >= 3

# 人脸不稳定状态阈值（侧脸/模糊）
FACE_UNSTABLE_CONF = 0.40    # 置信度 >= 0.40
FACE_UNSTABLE_SIZE = 48      # 尺寸 >= 48px
FACE_UNSTABLE_SIM = 0.30     # 相似度 >= 0.30
FACE_UNSTABLE_FRAMES = 2     # 连续帧 >= 2

# 人脸丢失阈值
FACE_LOST_CONF = 0.40        # 置信度 < 0.40
FACE_LOST_SIZE = 48          # 尺寸 < 48px
FACE_LOST_FRAMES = 3         # 连续丢失帧 >= 3

# 仅人脸匹配阈值（无人体时的备用）
FACE_ONLY_THRESHOLD = 0.70           # 稳定人脸
FACE_ONLY_THRESHOLD_UNSTABLE = 0.50  # 不稳定人脸 + motion辅助

# 自动学习阈值
FACE_LEARN_THRESHOLD = 0.72  # 人脸匹配学习阈值
FACE_LEARN_THRESHOLD_MULTI = 0.78  # 多人场景下的人脸学习阈值
BODY_LEARN_THRESHOLD = 0.68  # 人体匹配学习阈值

# 重新锁定阈值
RELOCK_FACE_THRESHOLD = 0.70
RELOCK_CONFIRM_FRAMES = 2
AUTO_LEARN_CONFIRM_FRAMES = 1

# 视角库最大容量（有脸3-4 + 无脸2 = 侧身+背面）
MAX_VIEW_COUNT = 6

# 人脸有效尺寸（匹配用）
MIN_FACE_SIZE = 40
MIN_FACE_SIZE_FOR_LEARN = 50

# ============================================
# 多帧投票机制
# ============================================
LOST_CONFIRM_FRAMES = 5
MATCH_HISTORY_SIZE = 5
MOTION_WEIGHT_MULTI_PERSON = 0.6
MOTION_WEIGHT_SINGLE_PERSON = 0.5

# 侧脸容忍度
MOTION_TRUST_THRESHOLD = 0.95
FACE_SIDE_VIEW_MIN = 0.35


# ============================================
# 人脸质量评估函数
# ============================================
def evaluate_face_quality(face_conf: float, face_size: int, face_sim: float) -> str:
    """
    评估人脸质量，返回状态: 'stable', 'unstable', 'lost'
    
    stable: 高置信度+大尺寸，或 超大尺寸可弥补低置信度，或 高相似度可弥补
    unstable: 中等质量 → motion辅助判断
    lost: 低质量或无人脸 → 切换到人体+motion
    
    关键改进：
    1. 大尺寸人脸（>=100px）即使置信度较低也应视为stable
    2. 高相似度（>=0.60）可以弥补小尺寸/低置信度（说明embedding质量好）
    """
    if face_conf is None or face_size is None:
        return 'lost'
    
    # 关键改进：高相似度说明 embedding 质量好，可以提升评级
    # 即使人脸小/检测置信度低，高相似度也说明是同一个人
    HIGH_SIM_THRESHOLD = 0.60
    MEDIUM_SIM_THRESHOLD = 0.45
    
    if face_sim is not None and face_sim >= HIGH_SIM_THRESHOLD:
        # 高相似度：只要尺寸不是太小（>=20px）就算 stable
        if face_size >= 20:
            return 'stable'
    
    if face_sim is not None and face_sim >= MEDIUM_SIM_THRESHOLD:
        # 中等相似度：只要尺寸不是太小（>=20px）就算 unstable
        if face_size >= 20:
            return 'unstable'
    
    # 大尺寸人脸可以弥补低置信度
    # size >= 100px 时，只要 conf >= 0.50 就算 stable
    LARGE_FACE_SIZE = 100
    LARGE_FACE_MIN_CONF = 0.50
    
    if face_size >= LARGE_FACE_SIZE and face_conf >= LARGE_FACE_MIN_CONF:
        # 大尺寸人脸：只要相似度不太低就算stable
        if face_sim is None or face_sim >= FACE_UNSTABLE_SIM:
            return 'stable'
    
    # 正常判断
    if (face_conf >= FACE_STABLE_CONF and 
        face_size >= FACE_STABLE_SIZE and 
        (face_sim is None or face_sim >= FACE_STABLE_SIM)):
        return 'stable'
    elif (face_conf >= FACE_UNSTABLE_CONF and 
          face_size >= FACE_UNSTABLE_SIZE and
          (face_sim is None or face_sim >= FACE_UNSTABLE_SIM)):
        return 'unstable'
    else:
        return 'lost'


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
        
        # ============== 手势有效性过滤 ==============
        # 1. 手势必须足够大（避免误识别远处的小手势）
        # 2. 优先识别屏幕中央区域的手势
        MIN_HAND_SIZE_FOR_GESTURE = 30  # 手势最小像素尺寸（降低到30px）
        CENTER_REGION_RATIO = 0.85  # 中央区域占比（扩大到85%）
        
        gesture_valid = False
        gesture_reject_reason = None
        
        if gesture.hand_bbox is not None and gesture.gesture_type in (GestureType.OPEN_PALM, GestureType.CLOSED_FIST):
            hx1, hy1, hx2, hy2 = gesture.hand_bbox
            hand_w = hx2 - hx1
            hand_h = hy2 - hy1
            hand_size = min(hand_w, hand_h)
            hand_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
            
            # 检查1：手势尺寸
            if hand_size < MIN_HAND_SIZE_FOR_GESTURE:
                gesture_reject_reason = f"手势太小({hand_size:.0f}px<{MIN_HAND_SIZE_FOR_GESTURE}px)"
            else:
                # 检查2：手势是否在中央区域
                center_x_min = w * (1 - CENTER_REGION_RATIO) / 2
                center_x_max = w * (1 + CENTER_REGION_RATIO) / 2
                center_y_min = h * (1 - CENTER_REGION_RATIO) / 2
                center_y_max = h * (1 + CENTER_REGION_RATIO) / 2
                
                in_center = (center_x_min <= hand_center[0] <= center_x_max and 
                            center_y_min <= hand_center[1] <= center_y_max)
                
                if in_center:
                    gesture_valid = True
                else:
                    gesture_reject_reason = f"手势不在中央区域"
        
        # 如果手势无效，重置为 none
        if not gesture_valid and gesture.gesture_type in (GestureType.OPEN_PALM, GestureType.CLOSED_FIST):
            if gesture_reject_reason and frame_count % 30 == 0:
                print(f"[DEBUG] 手势过滤: {gesture_reject_reason}")
            # 创建一个无效手势结果
            gesture = GestureResult(gesture_type=GestureType.NONE, confidence=0.0, hand_bbox=None)
        
        # 调试日志 (每30帧输出一次)
        if frame_count % 30 == 0:
            print(f"[DEBUG] Frame {frame_count}: persons={len(persons)}, faces={len(faces)}, gesture={gesture.gesture_type.value}")
            if faces:
                for i, face in enumerate(faces):
                    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                    face_size = min(fx2-fx1, fy2-fy1)
                    print(f"        Face[{i}]: bbox={[fx1,fy1,fx2,fy2]}, conf={face.confidence:.2f}, size={face_size}px")
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
                # ============================================
                # 启动跟随 - 方案D：必须有人脸，可以没人体
                # ============================================
                # 优先级1: 有人体 + 手势在框内 + 框内有人脸 → 锁定
                # 优先级2: 无人体 + 有人脸(质量够) → 锁定（直播场景）
                # 其他情况 → 拒绝启动
                # ============================================
                
                MIN_FACE_CONF_FOR_START = 0.65  # 启动时人脸最低置信度
                MIN_FACE_SIZE_FOR_START = 50    # 启动时人脸最小尺寸（标准）
                MIN_FACE_SIZE_FOR_START_RELAXED = 30  # 高置信度时可放宽到30px
                HIGH_CONF_FOR_RELAXED_SIZE = 0.75     # 置信度>=0.75时放宽尺寸要求
                
                target_locked = False
                
                # ========== 场景1: 有人体检测 ==========
                if persons:
                    target_person = None
                    target_idx = -1
                    face_in_target = None
                    
                    # 1. 找手势所在的人体
                    if gesture.hand_bbox is not None:
                        print(f"[DEBUG] 手势框: {gesture.hand_bbox.astype(int).tolist()}")
                        for pi, p in enumerate(persons):
                            px1, py1, px2, py2 = p.bbox.astype(int)
                            hc = ((gesture.hand_bbox[0] + gesture.hand_bbox[2]) / 2,
                                  (gesture.hand_bbox[1] + gesture.hand_bbox[3]) / 2)
                            in_box = px1 <= hc[0] <= px2 and py1 <= hc[1] <= py2
                            print(f"[DEBUG] Person[{pi}] bbox: [{px1}, {py1}, {px2}, {py2}], 手势在框内: {in_box}")
                            
                            if in_box:
                                target_person = p
                                target_idx = pi
                                break
                    
                    if target_person is None:
                        # 手势不在任何人体框内
                        state_machine.state = SystemState.IDLE
                        print("[提示] 手势未落在任何人体框内，请将手放在身体前方再做手势")
                    else:
                        # 2. 检查该人体框内是否有人脸
                        px1, py1, px2, py2 = target_person.bbox.astype(int)
                        best_face_in_person = None
                        best_face_info = None
                        
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                            fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                            face_w, face_h = fx2 - fx1, fy2 - fy1
                            face_size = min(face_w, face_h)
                            
                            # 检查人脸是否在人体框内
                            if not (px1 <= fc_x <= px2 and py1 <= fc_y <= py2):
                                continue
                            
                            # 记录这个人脸的信息用于调试
                            best_face_info = {
                                'conf': face.confidence,
                                'size': face_size,
                                'bbox': [fx1, fy1, fx2, fy2]
                            }
                            
                            # 人脸中心在人体框内 + 质量达标
                            # 高置信度(>=0.80)可接受更小尺寸(30px)
                            size_ok = face_size >= MIN_FACE_SIZE_FOR_START
                            size_ok_relaxed = (face.confidence >= HIGH_CONF_FOR_RELAXED_SIZE and 
                                               face_size >= MIN_FACE_SIZE_FOR_START_RELAXED)
                            
                            if (face.confidence >= MIN_FACE_CONF_FOR_START and
                                (size_ok or size_ok_relaxed)):
                                face_in_target = face
                                break
                        
                        if face_in_target is None:
                            # 人体框内没有合格的人脸 - 打印详细调试信息
                            state_machine.state = SystemState.IDLE
                            if best_face_info:
                                conf = best_face_info['conf']
                                size = best_face_info['size']
                                # 计算差距
                                conf_gap = MIN_FACE_CONF_FOR_START - conf if conf < MIN_FACE_CONF_FOR_START else 0
                                size_gap = MIN_FACE_SIZE_FOR_START - size if size < MIN_FACE_SIZE_FOR_START else 0
                                size_gap_relaxed = MIN_FACE_SIZE_FOR_START_RELAXED - size if size < MIN_FACE_SIZE_FOR_START_RELAXED else 0
                                
                                print(f"[启动检测] 当前人脸: conf={conf:.2f}, size={size}px")
                                print(f"           标准条件: conf>={MIN_FACE_CONF_FOR_START} ({'+' if conf>=MIN_FACE_CONF_FOR_START else '✗'}) + size>={MIN_FACE_SIZE_FOR_START}px ({'+' if size>=MIN_FACE_SIZE_FOR_START else '✗'})")
                                print(f"           放宽条件: conf>={HIGH_CONF_FOR_RELAXED_SIZE} ({'+' if conf>=HIGH_CONF_FOR_RELAXED_SIZE else '✗'}) + size>={MIN_FACE_SIZE_FOR_START_RELAXED}px ({'+' if size>=MIN_FACE_SIZE_FOR_START_RELAXED else '✗'})")
                                
                                # 给出具体建议
                                if conf < MIN_FACE_CONF_FOR_START:
                                    print(f"           💡 建议: 正面朝向镜头 (conf差{conf_gap:.2f})")
                                elif size < MIN_FACE_SIZE_FOR_START and conf < HIGH_CONF_FOR_RELAXED_SIZE:
                                    print(f"           💡 建议: 靠近镜头 (size差{size_gap}px) 或正面朝向 (conf差{HIGH_CONF_FOR_RELAXED_SIZE-conf:.2f})")
                                elif size < MIN_FACE_SIZE_FOR_START_RELAXED:
                                    print(f"           💡 建议: 靠近镜头 (size差{size_gap_relaxed}px)")
                            else:
                                print(f"[启动检测] 人体框内未检测到人脸，请面对镜头")
                        else:
                            # 3. 锁定目标（人体+人脸）
                            print(f"[DEBUG] 锁定 Person[{target_idx}]: bbox={target_person.bbox.astype(int).tolist()}")
                            view = extract_view_feature(
                                frame, target_person.bbox, faces, 
                                face_recognizer, enhanced_reid
                            )
                            print(f"[DEBUG] 提取特征: has_face={view.has_face}, has_body={view.part_color_hists is not None}")
                            if view.has_face and view.face_embedding is not None:
                                print(f"[DEBUG] 人脸embedding: shape={view.face_embedding.shape}, norm={np.linalg.norm(view.face_embedding):.3f}")
                                mv_recognizer.set_target(view, target_person.bbox)
                                mv_recognizer.clear_match_history()
                                lost_frames = 0
                                target_locked = True
                                print(f"[手势启动] 目标已锁定 (人体+人脸)")
                            else:
                                state_machine.state = SystemState.IDLE
                                print("[提示] 人脸特征提取失败，请重试")
                
                # ========== 场景2: 无人体，仅人脸（直播场景）==========
                elif faces:
                    # 找最佳人脸
                    best_face = None
                    best_face_score = -1
                    
                    for face in faces:
                        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                        face_w, face_h = fx2 - fx1, fy2 - fy1
                        face_size = min(face_w, face_h)
                        face_conf = face.confidence
                        
                        # 高置信度(>=0.80)可接受更小尺寸(30px)
                        size_ok = face_size >= MIN_FACE_SIZE_FOR_START
                        size_ok_relaxed = (face_conf >= HIGH_CONF_FOR_RELAXED_SIZE and 
                                           face_size >= MIN_FACE_SIZE_FOR_START_RELAXED)
                        
                        if face_conf >= MIN_FACE_CONF_FOR_START and (size_ok or size_ok_relaxed):
                            score = face_conf + face_size / 200.0
                            if score > best_face_score:
                                best_face_score = score
                                best_face = face
                    
                    if best_face is not None:
                        fx1, fy1, fx2, fy2 = best_face.bbox.astype(int)
                        face_w, face_h = fx2 - fx1, fy2 - fy1
                        face_size = min(face_w, face_h)
                        print(f"[DEBUG] 仅人脸模式: bbox={best_face.bbox.astype(int).tolist()}, conf={best_face.confidence:.2f}, size={face_size}px")
                        
                        # 用人脸框扩展为伪人体框
                        pseudo_bbox = np.array([
                            max(0, fx1 - face_w * 0.5),
                            fy1,
                            min(w, fx2 + face_w * 0.5),
                            min(h, fy2 + face_h * 5)
                        ])
                        print(f"[DEBUG] 伪人体框: {pseudo_bbox.astype(int).tolist()}")
                        
                        # 提取人脸特征
                        view = ViewFeature(timestamp=time.time())
                        view.has_face = True
                        face_feature = face_recognizer.extract_feature(
                            frame, best_face.bbox, best_face.keypoints
                        )
                        if face_feature and face_feature.embedding is not None:
                            view.face_embedding = face_feature.embedding
                            print(f"[DEBUG] 人脸特征: shape={face_feature.embedding.shape}, norm={np.linalg.norm(face_feature.embedding):.3f}")
                            
                            mv_recognizer.set_target(view, pseudo_bbox)
                            mv_recognizer.clear_match_history()
                            lost_frames = 0
                            target_locked = True
                            print(f"[手势启动] 目标已锁定 (仅人脸模式，等待人体补充)")
                        else:
                            state_machine.state = SystemState.IDLE
                            print("[提示] 人脸特征提取失败，请重试")
                    else:
                        state_machine.state = SystemState.IDLE
                        print(f"[提示] 人脸质量不足 (需要conf>={MIN_FACE_CONF_FOR_START}+size>={MIN_FACE_SIZE_FOR_START}px, 或conf>={HIGH_CONF_FOR_RELAXED_SIZE}+size>={MIN_FACE_SIZE_FOR_START_RELAXED}px)")
                
                # ========== 场景3: 无检测 ==========
                else:
                    state_machine.state = SystemState.IDLE
                    print("[提示] 未检测到人脸，无法启动")
            
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
        # 单人场景的严格定义:
        #   1. 只有单脸（无人体）
        #   2. 只有单人体（无脸）
        #   3. 单脸 + 单人体，且脸在人体框内
        # 多人场景:
        #   1. 多个人体
        #   2. 多个人脸
        #   3. 单脸 + 单人体，但脸不在人体框内（两个不同的人）
        num_persons = len(persons)
        num_faces = len(faces)
        
        # 检查单脸+单人体时，脸是否在人体框内
        face_in_person_for_scene = False
        if num_faces == 1 and num_persons == 1:
            fx1, fy1, fx2, fy2 = faces[0].bbox.astype(int)
            fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
            px1, py1, px2, py2 = persons[0].bbox.astype(int)
            face_in_person_for_scene = (px1 <= fc_x <= px2 and py1 <= fc_y <= py2)
        
        # 判断是否为单人场景
        if num_persons == 0 and num_faces == 0:
            is_single_person_scene = True  # 没人
        elif num_persons == 0 and num_faces == 1:
            is_single_person_scene = True  # 只有单脸
        elif num_persons == 1 and num_faces == 0:
            is_single_person_scene = True  # 只有单人体
        elif num_persons == 1 and num_faces == 1 and face_in_person_for_scene:
            is_single_person_scene = True  # 单脸+单人体，脸在框内
        else:
            is_single_person_scene = False  # 其他都是多人场景
        
        is_multi_person_scene = not is_single_person_scene
        
        # ============================================
        # 交汇检测：两人框重叠时需要特殊处理
        # ============================================
        # 场景：两人交叉走过，非目标站到前面，遮挡目标
        # 风险：如果人脸太小无法验证，可能误跟踪到非目标
        # 策略：检测到交汇时，提高匹配阈值，宁可丢失也不误跟踪
        is_crossing_scene = False
        crossing_iou = 0.0
        
        if num_persons >= 2:
            # 计算所有人体框之间的最大IoU
            for i in range(num_persons):
                for j in range(i + 1, num_persons):
                    box1 = persons[i].bbox.astype(int)
                    box2 = persons[j].bbox.astype(int)
                    
                    # 计算IoU
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])
                    
                    if x1 < x2 and y1 < y2:
                        inter_area = (x2 - x1) * (y2 - y1)
                        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                        union_area = area1 + area2 - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        if iou > crossing_iou:
                            crossing_iou = iou
            
            # IoU > 0.15 认为是交汇场景
            CROSSING_IOU_THRESHOLD = 0.15
            is_crossing_scene = crossing_iou > CROSSING_IOU_THRESHOLD
            
            if is_crossing_scene and frame_count % 30 == 0:
                print(f"[DEBUG] ⚠️ 检测到交汇场景 (IoU={crossing_iou:.2f}), 启用严格匹配模式")
        
        if state_machine.state == SystemState.TRACKING:
            matched_any = False
            
            # 调试: 显示目标信息和场景类型
            if frame_count % 30 == 0:
                scene_type = "多人" if is_multi_person_scene else "单人"
                extra_info = ""
                if num_persons == 1 and num_faces == 1:
                    extra_info = f", 脸在框内={face_in_person_for_scene}"
                print(f"[DEBUG] 场景: {scene_type} (persons={num_persons}, faces={num_faces}{extra_info})")
                if mv_recognizer.target:
                    t = mv_recognizer.target
                    print(f"[DEBUG] Target: num_views={t.num_views}, has_face_view={t.has_face_view}")
                    for vi, v in enumerate(t.view_features):
                        print(f"        View[{vi}]: has_face={v.has_face}, has_body={v.part_color_hists is not None}")
            
            # 1. 通过人体匹配 - 使用"最佳匹配"策略（而不是"第一个匹配"）
            # 收集所有候选匹配，选择最高分的
            all_person_matches = []  # [(idx, similarity, method, view, face_in_person, face_verified, face_sim, body_sim)]
            
            # 记录人体被拒绝的原因，用于决定是否允许仅人脸匹配
            persons_rejected_by_face_mismatch = 0  # 因"人脸明确不匹配"被拒绝的人体数
            persons_total_checked = 0
            
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
                    # 分层匹配逻辑（基于人脸质量分级）
                    # ============================================
                    # 对于【有效人脸】(size>=50px, conf>=0.65):
                    #   F >= 0.65: face_priority (高置信度，仅靠人脸)
                    #   0.45 <= F < 0.65: face + motion (中等置信度)
                    #   0.30 <= F < 0.45: body + motion (低置信度人脸)
                    #   F < 0.30: 明确拒绝 (即使body+motion高也拒绝)
                    # 
                    # 对于【无效人脸】(小/低置信度/无人脸):
                    #   只能靠 body + motion
                    # ============================================
                    
                    # 人脸相似度分层阈值
                    FACE_HIGH_THRESHOLD = 0.65      # 高置信度：仅靠人脸
                    FACE_MEDIUM_THRESHOLD = 0.45    # 中等置信度：人脸+motion
                    FACE_LOW_THRESHOLD = 0.30       # 低置信度临界值
                    FACE_REJECT_THRESHOLD = 0.30    # 低于此值明确拒绝
                    
                    FACE_MATCH_THRESHOLD = 0.55     # 人脸匹配阈值（兼容旧逻辑）
                    BODY_MOTION_THRESHOLD = 0.65    # body + motion 综合阈值
                    MULTI_PERSON_BODY_THRESHOLD = 0.70  # 多人场景下仅body匹配的阈值
                    
                    # 有效人脸的定义
                    MIN_FACE_SIZE_FOR_VALID = 30    # 有效人脸最小尺寸（测试验证30px即可准确识别）
                    MIN_FACE_CONF_FOR_VALID = 0.65  # 有效人脸最低置信度
                    MIN_FACE_SIZE_RELAXED = 30      # 放宽条件的最小尺寸
                    
                    # 检查人脸尺寸是否足够大
                    face_size_valid = False
                    face_size_valid_relaxed = False  # 放宽条件（单人+高相似度）
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
                            face_size_valid_relaxed = current_face_size >= MIN_FACE_SIZE_RELAXED
                            break
                    
                    # 计算 body + motion 综合分数
                    # 多人场景下增加 motion 权重，因为运动轨迹更可靠
                    if is_multi_person_scene:
                        motion_weight = MOTION_WEIGHT_MULTI_PERSON
                    else:
                        motion_weight = MOTION_WEIGHT_SINGLE_PERSON
                    body_weight = 1.0 - motion_weight
                    body_motion_score = body_sim * body_weight + motion_score * motion_weight
                    
                    # 判断匹配类型
                    # 人脸有效条件：
                    #   - 标准: 相似度>=0.55 且 尺寸>=40px
                    #   - 放宽(单人+高置信): 相似度>=0.65 且 尺寸>=20px
                    face_matched_standard = (face_sim is not None and 
                                             face_sim >= FACE_MATCH_THRESHOLD and 
                                             face_size_valid)
                    face_matched_relaxed = (face_sim is not None and 
                                            face_sim >= FACE_HIGH_THRESHOLD and  # 修复：使用正确的常量名
                                            face_size_valid_relaxed and 
                                            is_single_person_scene)
                    face_matched = face_matched_standard or face_matched_relaxed
                    body_motion_matched = body_motion_score >= BODY_MOTION_THRESHOLD
                    
                    if frame_count % 30 == 0 and face_sim is not None:
                        # 只有当通过放宽条件而非标准条件时才显示 relaxed
                        relaxed_info = ""
                        if face_matched_relaxed and not face_matched_standard:
                            relaxed_info = ", relaxed=True"
                        print(f"[DEBUG] Person[{idx}] face_size={current_face_size}px, valid={face_size_valid}, face_matched={face_matched}{relaxed_info}")
                    
                    # 决策逻辑
                    accept = False
                    match_type = ""
                    persons_total_checked += 1
                    
                    # ============================================
                    # 分层决策：基于人脸有效性和相似度
                    # ============================================
                    # 1. 先判断人脸是否"有效"（可用于判断身份）
                    # 2. 有效人脸：根据相似度分层决策
                    # 3. 无效人脸：只能靠 body + motion
                    # ============================================
                    
                    # 获取人脸置信度（用于判断有效性）
                    current_face_conf = 0.0
                    for face in faces:
                        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                        fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                        px1, py1, px2, py2 = person.bbox.astype(int)
                        if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                            current_face_conf = face.confidence
                            break
                    
                    # 判断人脸是否"有效"（可用于身份判断）
                    face_is_valid = (face_in_person and 
                                    face_sim is not None and 
                                    current_face_size >= MIN_FACE_SIZE_FOR_VALID and
                                    current_face_conf >= MIN_FACE_CONF_FOR_VALID)
                    
                    # 放宽的有效条件（高相似度时可接受较小人脸）
                    face_is_valid_relaxed = (face_in_person and 
                                            face_sim is not None and 
                                            current_face_size >= MIN_FACE_SIZE_RELAXED and
                                            current_face_conf >= 0.60)
                    
                    if frame_count % 30 == 0:
                        face_str = f"F:{face_sim:.2f}" if face_sim is not None else "F:None"
                        print(f"[DEBUG] Person[{idx}] 人脸有效性: size={current_face_size}px, conf={current_face_conf:.2f}, valid={face_is_valid}, relaxed_valid={face_is_valid_relaxed}")
                    
                    if face_is_valid:
                        # ========== 有效人脸：基于相似度分层 ==========
                        if face_sim >= FACE_HIGH_THRESHOLD:
                            # Layer 1: F >= 0.65 → 高置信度，仅靠人脸
                            accept = True
                            match_type = "face"
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] Person[{idx}] ✓ 有效人脸高置信度 (F:{face_sim:.2f}>=0.65) → face_priority")
                        elif face_sim >= FACE_MEDIUM_THRESHOLD:
                            # Layer 2: 0.45 <= F < 0.65 → 中等置信度，需要motion辅助
                            # 要求 motion >= 0.5 或 综合分数够高
                            if motion_score >= 0.5 or body_motion_score >= BODY_MOTION_THRESHOLD:
                                accept = True
                                match_type = "face_motion"
                                if frame_count % 30 == 0:
                                    print(f"[DEBUG] Person[{idx}] ✓ 有效人脸中等置信度 (F:{face_sim:.2f}, M:{motion_score:.2f}) → face+motion")
                            else:
                                accept = False
                                if frame_count % 30 == 0:
                                    print(f"[DEBUG] Person[{idx}] ✗ 有效人脸中等但motion不足 (F:{face_sim:.2f}, M:{motion_score:.2f}<0.5)")
                        elif face_sim >= FACE_LOW_THRESHOLD:
                            # Layer 3: 0.30 <= F < 0.45 → 低置信度，需要body+motion
                            if body_motion_score >= BODY_MOTION_THRESHOLD:
                                accept = True
                                match_type = "body_motion"
                                if frame_count % 30 == 0:
                                    print(f"[DEBUG] Person[{idx}] ✓ 有效人脸低置信度 (F:{face_sim:.2f}) + body+motion高 → body+motion")
                            else:
                                accept = False
                                if frame_count % 30 == 0:
                                    print(f"[DEBUG] Person[{idx}] ✗ 有效人脸低置信度 (F:{face_sim:.2f}) 且body+motion不足")
                        else:
                            # Layer 4: F < 0.30 → 明确不匹配，拒绝！
                            # ★★★ 核心修复：即使body+motion高也拒绝 ★★★
                            accept = False
                            persons_rejected_by_face_mismatch += 1
                            if frame_count % 30 == 0:
                                scene_type = "多人" if is_multi_person_scene else "单人"
                                print(f"[DEBUG] Person[{idx}] ✗✗ {scene_type}有效人脸明确不匹配 (F:{face_sim:.2f}<0.30) → 直接拒绝")
                    
                    elif face_is_valid_relaxed and face_sim is not None and face_sim >= FACE_HIGH_THRESHOLD:
                        # ========== 放宽条件：较小人脸但高相似度 ==========
                        accept = True
                        match_type = "face"
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] Person[{idx}] ✓ 放宽有效人脸高置信度 (F:{face_sim:.2f}>=0.65, size={current_face_size}px)")
                    
                    elif face_is_valid_relaxed and face_sim is not None and face_sim < FACE_REJECT_THRESHOLD:
                        # ========== 放宽条件：较小人脸但明确不匹配 ==========
                        accept = False
                        persons_rejected_by_face_mismatch += 1
                        if frame_count % 30 == 0:
                            scene_type = "多人" if is_multi_person_scene else "单人"
                            print(f"[DEBUG] Person[{idx}] ✗✗ {scene_type}较小人脸明确不匹配 (F:{face_sim:.2f}<0.30, size={current_face_size}px) → 拒绝")
                    
                    elif body_motion_matched:
                        # ========== 无有效人脸：靠 body + motion ==========
                        
                        # ★★★ 重要：小人脸（<30px）的F值不可靠，不能用于拒绝决策！★★★
                        # 只有"放宽有效"的人脸（>=30px）才能用F<0.30来判断不匹配
                        # 小人脸的低F值可能是特征提取不准，而不是真的不匹配
                        
                        if face_is_valid_relaxed and face_sim is not None and face_sim < FACE_REJECT_THRESHOLD:
                            # 放宽有效的人脸（>=30px），F<0.30 → 明确不匹配
                            accept = False
                            persons_rejected_by_face_mismatch += 1
                            if frame_count % 30 == 0:
                                scene_type = "多人" if is_multi_person_scene else "单人"
                                print(f"[DEBUG] Person[{idx}] ✗✗ {scene_type}人脸明确不匹配 (F:{face_sim:.2f}<0.30, size={current_face_size}px>=30) → 拒绝")
                        # 交汇场景特殊处理
                        elif is_crossing_scene and target_has_face:
                            # 交汇时没有有效人脸验证 → 宁可短暂丢失
                            if frame_count % 30 == 0:
                                face_str = f"F:{face_sim:.2f}" if face_sim is not None else "F:None"
                                print(f"[DEBUG] Person[{idx}] ⚠️ 交汇场景无有效人脸({face_str}, size={current_face_size}px), 暂停匹配")
                            accept = False
                        # 多人场景：body阈值提高
                        elif is_multi_person_scene and target_has_face and body_sim < MULTI_PERSON_BODY_THRESHOLD - 0.01:
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] Person[{idx}] ✗ 多人场景无有效人脸且body不足({body_sim:.2f}<{MULTI_PERSON_BODY_THRESHOLD-0.01:.2f})")
                            accept = False
                        else:
                            accept = True
                            match_type = "body_motion"
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] Person[{idx}] ✓ 无有效人脸，body+motion通过 (B:{body_sim:.2f}+M:{motion_score:.2f}={body_motion_score:.2f})")
                    else:
                        # ========== 什么都不够 ==========
                        if frame_count % 30 == 0:
                            face_str = f"F:{face_sim:.2f}" if face_sim is not None else "F:None"
                            print(f"[DEBUG] Person[{idx}] ✗ 无有效人脸且body+motion不足 ({face_str}, BM:{body_motion_score:.2f})")
                        accept = False
                    
                    if accept:
                        # tuple: (idx, similarity, method, view, face_in_person, face_matched, face_sim, body_sim, motion_score, match_type)
                        all_person_matches.append((idx, similarity, method, view, face_in_person, face_matched, face_sim, body_sim, motion_score, match_type))
            
            # 选择最佳匹配
            if all_person_matches:
                # 策略: 
                #   1. 优先选人脸匹配的（身份最可靠）：face, face_motion
                #   2. 人脸匹配中优先选 motion 高的（轨迹最一致）
                #   3. 其次选 body+motion 匹配的
                #   4. body+motion 中多人场景优先选 motion 高的
                # tuple: (idx, similarity, method, view, face_in_person, face_matched, face_sim, body_sim, motion_score, match_type)
                matches_by_face = [m for m in all_person_matches if m[9] in ("face", "face_motion")]  # m[9] = match_type
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
                    
                    # ★★★ 关键日志：标注最终选择的目标 ★★★
                    if frame_count % 30 == 0:
                        px1, py1, px2, py2 = persons[idx].bbox.astype(int)
                        face_str = f"F:{match_face_sim:.2f}" if match_face_sim is not None else "F:None"
                        print(f"[★目标★] Person[{idx}] 被选为目标 (绿框)")
                        print(f"         bbox=[{px1},{py1},{px2},{py2}], {face_str}, B:{match_body_sim:.2f}, M:{match_motion_score:.2f}")
                        print(f"         匹配类型: {match_type}, 方法: {method[:50]}")
                    
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
                    #   4. 视角库满时：用替换策略而非停止学习
                    # ============================================
                    
                    should_learn = False
                    learn_what = ""  # "body" or "face" or "both"
                    learn_reason = ""
                    use_replace_strategy = False  # 是否使用替换策略
                    
                    target_has_body = (mv_recognizer.target is not None and 
                                       any(v.has_body for v in mv_recognizer.target.view_features))
                    
                    # 容量检查：视角库满时改用替换策略
                    current_view_count = mv_recognizer.target.num_views if mv_recognizer.target else 0
                    if current_view_count >= MAX_VIEW_COUNT:
                        # 不停止学习，而是检查是否值得替换
                        use_replace_strategy = True
                        if frame_count % 60 == 0:
                            print(f"[DEBUG] 视角库已满({current_view_count})，启用替换策略")
                    
                    # 多人场景 + 没有人脸匹配 = 禁止学习
                    # 多人场景 + 人脸-人体不一致 = 禁止学习（防止关联错误导致学习污染）
                    if is_multi_person_scene and match_type != "face":
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] 多人场景无人脸匹配，禁止学习")
                        should_learn = False
                    elif is_multi_person_scene and match_type == "face":
                        # 多人场景下人脸匹配：检查人脸-人体一致性
                        # 如果人脸高匹配(F>=0.55)但身体低匹配(B<0.60)，可能是关联错误
                        match_face_sim_check = best_match[6] if best_match[6] is not None else 0.0
                        match_body_sim_check = best_match[7]
                        
                        # 计算差距：人脸相似度 - 身体相似度
                        face_body_gap = match_face_sim_check - match_body_sim_check
                        
                        # 如果差距过大（>=0.25），或者身体相似度太低（<0.55），禁止学习
                        FACE_BODY_CONSISTENCY_GAP = 0.25  # 允许的最大差距
                        BODY_MIN_FOR_LEARN_MULTI = 0.55   # 多人场景下学习需要的最低body相似度
                        
                        if match_body_sim_check < BODY_MIN_FOR_LEARN_MULTI:
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] 多人场景人脸-人体不一致(F:{match_face_sim_check:.2f}, B:{match_body_sim_check:.2f}<{BODY_MIN_FOR_LEARN_MULTI})，禁止学习")
                            should_learn = False
                        elif face_body_gap > FACE_BODY_CONSISTENCY_GAP:
                            if frame_count % 30 == 0:
                                print(f"[DEBUG] 多人场景人脸-人体差距过大(F:{match_face_sim_check:.2f}-B:{match_body_sim_check:.2f}={face_body_gap:.2f}>{FACE_BODY_CONSISTENCY_GAP})，禁止学习")
                            should_learn = False
                    
                    # 只有未被禁止学习时才继续学习逻辑
                    if should_learn is not False or (not is_multi_person_scene):
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
                            # 检查目标是否还没有人体视角（仅人脸模式启动的情况）
                            target_has_body_view = mv_recognizer.target is not None and any(v.has_body for v in mv_recognizer.target.view_features)
                            
                            # Case 1a: 目标没有人体视角（仅人脸模式启动）+ 人脸匹配成功 + body+motion高 → 升级初始视角
                            if not target_has_body_view and face_in_person and body_motion_combined >= 0.70:
                                initial_view = mv_recognizer.target.view_features[0] if mv_recognizer.target.view_features else None
                                if initial_view and initial_view.has_face and not initial_view.has_body:
                                    # 升级初始视角：把人体特征加到初始视角上
                                    # 合并：保留初始的人脸特征 + 新的人体特征
                                    if view.part_color_hists is not None:
                                        initial_view.part_color_hists = view.part_color_hists
                                        initial_view.timestamp = time.time()
                                        print(f"[初始视角升级] 仅人脸→有人体(F:{match_face_sim:.2f}, BM:{body_motion_combined:.2f})")
                                        should_learn = False  # 已经升级，不需要再学习
                                    else:
                                        should_learn = True
                                        learn_what = "body"
                                        learn_reason = f"首次学习人体(F:{match_face_sim:.2f}, BM:{body_motion_combined:.2f})"
                                else:
                                    should_learn = True
                                    learn_what = "body"
                                    learn_reason = f"人脸匹配(F:{match_face_sim:.2f})学习body(BM:{body_motion_combined:.2f})"
                            
                            # Case 1b: 目标已有人体视角 + 人脸匹配 + body+motion高 → 学习body
                            elif target_has_body_view and body_motion_combined >= BODY_MOTION_LEARN_THRESHOLD:
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
                        # 注意：方案D确保启动时一定有人脸，所以不需要"首次学习人脸"逻辑
                        elif match_type == "body_motion":
                            # Case 2a: 人脸相似度够高 → 学习/更新人脸
                            if face_in_person and match_face_sim >= FACE_MIN_FOR_BODY_LEARN and face_size_ok_for_learn:
                                # 关键约束：人脸必须在人体框内 且 尺寸足够大！
                                should_learn = True
                                learn_what = "face"
                                learn_reason = f"body+motion匹配(BM:{body_motion_combined:.2f})学习face(F:{match_face_sim:.2f}, size={current_face_size_for_learn}px)"
                            
                            # Case 2b: 无人脸/背面/侧面 → 学习body视角
                            elif not face_in_person and body_motion_combined >= BODY_MOTION_LEARN_THRESHOLD:
                                # 纯背面/侧面，学习body视角
                                # 注意：这里 face_in_person=False 可能是：
                                #   1. 真正的背面（没有人脸检测）
                                #   2. 人脸检测漏检（瞬时）
                                #   3. 人脸不在人体框内（检测偏移）
                                should_learn = True
                                learn_what = "body"
                                reason_detail = "背面/无脸" if len(faces) == 0 else "脸不在框内"
                                learn_reason = f"{reason_detail}匹配(BM:{body_motion_combined:.2f})"
                    
                    # 执行学习（普通或替换模式）
                    if should_learn:
                        if use_replace_strategy:
                            # 替换策略：找到最差/最老的视角替换
                            # 评估当前视角质量
                            current_quality = 0.0
                            if learn_what == "face" and match_face_sim >= FACE_LEARN_THRESHOLD_LOCAL:
                                current_quality = match_face_sim
                            elif learn_what == "body" and body_motion_combined >= BODY_MOTION_LEARN_THRESHOLD:
                                current_quality = body_motion_combined
                            
                            # 只有当前视角质量够高才考虑替换
                            if current_quality >= 0.75:  # 替换门槛要高
                                learned, op_info = mv_recognizer.auto_learn(view, persons[idx].bbox, True, replace_mode=True)
                                if learned:
                                    print(f"[替换学习] {learn_reason} (quality={current_quality:.2f}) -> {op_info}")
                            else:
                                if frame_count % 60 == 0:
                                    print(f"[DEBUG] 当前质量({current_quality:.2f})<0.75，不替换")
                        else:
                            # 普通学习模式
                            learned, op_info = mv_recognizer.auto_learn(view, persons[idx].bbox, True)
                            if learned:
                                print(f"[自动学习 F{frame_count}] {learn_reason} -> {op_info}")
            
            # 2. 如果人体没匹配到，尝试仅通过人脸匹配
            # ============================================
            # 根据人脸质量使用不同策略:
            #   - stable (高质量): 纯人脸匹配，阈值0.70
            #   - unstable (中等): 人脸+motion辅助，阈值0.50
            #   - lost (低质量): 无法匹配，等待人体出现
            # ============================================
            if not matched_any and faces and mv_recognizer.target and mv_recognizer.target.has_face_view:
                
                best_face_match = None
                best_face_sim = 0.0
                best_face_idx = -1
                best_view_idx = -1
                best_face_quality = 'lost'
                best_face_conf = 0.0
                best_face_size = 0
                
                # 多人脸场景需要更严格的阈值
                multi_face_penalty = 0.05 if num_faces > 1 else 0.0
                    
                for face_idx, face in enumerate(faces):
                    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                    fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    face_w = fx2 - fx1
                    face_h = fy2 - fy1
                    face_size = min(face_w, face_h)
                    face_conf = float(face.score) if hasattr(face, 'score') else 0.5
                    
                    # 检查人脸是否在某个人体框内
                    face_in_any_person = False
                    if len(persons) > 0:
                        for p_idx, person in enumerate(persons):
                            px1, py1, px2, py2 = person.bbox.astype(int)
                            if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                                face_in_any_person = True
                                break
                    
                    # 多人场景，人脸在不匹配的人体框内 → 通常跳过
                    # 但例外：如果所有人体都是因为"人脸明确不匹配"被拒绝的，
                    # 说明目标可能是另一个人脸（检测错位），应该继续尝试匹配
                    all_rejected_by_face_mismatch = (persons_total_checked > 0 and 
                                                     persons_rejected_by_face_mismatch == persons_total_checked)
                    
                    if num_persons > 1 and face_in_any_person and not all_rejected_by_face_mismatch:
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] Face[{face_idx}] 在不匹配的人体框内(多人场景)，跳过")
                        continue
                    elif num_persons > 1 and face_in_any_person and all_rejected_by_face_mismatch:
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] Face[{face_idx}] 所有人体因人脸不匹配被拒绝，尝试仅人脸匹配")
                    
                    # 远处人脸使用更高阈值
                    is_distant_face = num_persons > 0 and not face_in_any_person
                    
                    face_feature = face_recognizer.extract_feature(
                        frame, face.bbox, face.keypoints
                    )
                    if face_feature and face_feature.embedding is not None:
                        # 与目标人脸特征比较，找最高相似度
                        for vi, view in enumerate(mv_recognizer.target.view_features):
                            if view.has_face and view.face_embedding is not None:
                                sim = float(np.dot(face_feature.embedding, view.face_embedding))
                                
                                # 评估人脸质量
                                face_quality = evaluate_face_quality(face_conf, face_size, sim)
                                
                                # 根据质量决定阈值
                                if face_quality == 'stable':
                                    current_threshold = FACE_ONLY_THRESHOLD + multi_face_penalty
                                    if is_distant_face:
                                        # 高相似度(>=0.75)减少distant惩罚
                                        if sim >= 0.75:
                                            current_threshold += 0.05  # 减半惩罚
                                        else:
                                            current_threshold += 0.10
                                elif face_quality == 'unstable':
                                    # 不稳定人脸: 使用更低阈值，但需要motion辅助验证
                                    current_threshold = FACE_ONLY_THRESHOLD_UNSTABLE + multi_face_penalty
                                    if is_distant_face:
                                        current_threshold += 0.05
                                else:
                                    current_threshold = 1.0  # 无法匹配
                                
                                if frame_count % 30 == 0:
                                    print(f"[DEBUG] Face[{face_idx}] vs View[{vi}]: sim={sim:.3f}, conf={face_conf:.2f}, size={face_size}px, quality={face_quality}, threshold={current_threshold:.2f}")
                                
                                if sim >= current_threshold and sim > best_face_sim:
                                    best_face_sim = sim
                                    best_face_idx = face_idx
                                    best_view_idx = vi
                                    best_face_quality = face_quality
                                    best_face_conf = face_conf
                                    best_face_size = face_size
                
                # 根据人脸质量决定是否匹配成功
                face_match_success = False
                
                if best_face_quality == 'stable' and best_face_sim >= FACE_ONLY_THRESHOLD:
                    # 稳定人脸: 纯人脸匹配
                    face_match_success = True
                    if frame_count % 30 == 0:
                        print(f"[DEBUG] 稳定人脸匹配成功! face_idx={best_face_idx}, sim={best_face_sim:.3f}")
                        
                elif best_face_quality == 'unstable' and best_face_sim >= FACE_ONLY_THRESHOLD_UNSTABLE:
                    # 不稳定人脸: 需要motion辅助验证
                    # 获取motion分数（使用最近的位置预测）
                    motion_score = 0.0
                    last_bbox = mv_recognizer.target.last_bbox if mv_recognizer.target else None
                    if last_bbox is not None and best_face_idx >= 0:
                        # 计算人脸框与预测位置的IOU
                        face_bbox = faces[best_face_idx].bbox
                        pred_bbox = last_bbox
                        
                        # 简化: 用中心点距离代替IOU
                        fc_x = (face_bbox[0] + face_bbox[2]) / 2
                        fc_y = (face_bbox[1] + face_bbox[3]) / 2
                        pc_x = (pred_bbox[0] + pred_bbox[2]) / 2
                        pc_y = (pred_bbox[1] + pred_bbox[3]) / 2
                        
                        # 计算归一化距离
                        frame_h, frame_w = frame.shape[:2]
                        dist = np.sqrt((fc_x - pc_x)**2 + (fc_y - pc_y)**2)
                        max_dist = np.sqrt(frame_w**2 + frame_h**2) * 0.3  # 允许30%画面距离
                        motion_score = max(0, 1.0 - dist / max_dist)
                    
                    # 不稳定人脸 + motion辅助
                    combined_score = best_face_sim * 0.6 + motion_score * 0.4
                    if combined_score >= 0.50:  # 综合分数阈值
                        face_match_success = True
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] 不稳定人脸+motion匹配成功! face={best_face_sim:.2f}, motion={motion_score:.2f}, combined={combined_score:.2f}")
                    else:
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] 不稳定人脸+motion不足 (face={best_face_sim:.2f}, motion={motion_score:.2f}, combined={combined_score:.2f}<0.50)")
                
                if face_match_success:
                    matched_any = True
                    target_face_idx = best_face_idx
                    lost_frames = 0
                    
                    current_match_info = {
                        'type': 'face_only',
                        'similarity': best_face_sim,
                        'method': f'face_only_{best_face_quality} (vs View[{best_view_idx}])',
                        'threshold': FACE_ONLY_THRESHOLD if best_face_quality == 'stable' else FACE_ONLY_THRESHOLD_UNSTABLE
                    }
                    
                    mv_recognizer.update_tracking(faces[best_face_idx].bbox)
                    
                    # 仅人脸匹配时的自动学习 - 只有稳定人脸才学习
                    if best_face_quality == 'stable' and best_face_sim >= 0.80 and is_single_person_scene:
                        face_only_view = ViewFeature(timestamp=time.time())
                        face_feature = face_recognizer.extract_feature(
                            frame, faces[best_face_idx].bbox, faces[best_face_idx].keypoints
                        )
                        if face_feature:
                            face_only_view.has_face = True
                            face_only_view.face_embedding = face_feature.embedding
                            learned, op_info = mv_recognizer.auto_learn(face_only_view, faces[best_face_idx].bbox, True)
                            if learned:
                                print(f"[自动学习 F{frame_count}] 仅人脸(sim={best_face_sim:.2f}) -> {op_info}")
                    elif frame_count % 30 == 0 and best_face_sim >= 0.60:
                        if best_face_quality == 'unstable':
                            print(f"[DEBUG] 不稳定人脸不学习")
                        elif not is_single_person_scene:
                            print(f"[DEBUG] 仅人脸匹配不学习: 多人场景")
                        else:
                            print(f"[DEBUG] 仅人脸匹配不学习: 相似度不足({best_face_sim:.2f}<0.80)")
                            
                elif frame_count % 30 == 0 and best_face_sim > 0:
                    print(f"[DEBUG] 人脸匹配失败: sim={best_face_sim:.3f}, quality={best_face_quality}")
            
            if not matched_any:
                lost_frames += 1
                # 清空匹配历史，防止误匹配
                mv_recognizer.clear_match_history()
                if frame_count % 30 == 0:
                    print(f"[DEBUG F{frame_count}] 未匹配, lost_frames={lost_frames}/{max_lost_frames}")
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
        # ★★★ 绘制前日志：明确 target_person_idx 的值 ★★★
        if frame_count % 30 == 0:
            print(f"\n[绘制] target_person_idx={target_person_idx}, state={state_machine.state.value}")
            for idx, person in enumerate(persons):
                px1, py1, px2, py2 = person.bbox.astype(int)
                is_target = (idx == target_person_idx)
                print(f"       Person[{idx}]: bbox=[{px1},{py1},{px2},{py2}], 是目标={is_target}")
        
        # 绘制人体框
        for idx, person in enumerate(persons):
            px1, py1, px2, py2 = person.bbox.astype(int)
            
            if state_machine.state == SystemState.IDLE:
                color = (255, 165, 0)  # 橙色
                label = "Candidate"
            elif idx == target_person_idx:
                color = (0, 255, 0)  # 绿色
                label = f"TARGET[{idx}]"  # 标注索引
            else:
                color = (0, 0, 255)  # 红色
                label = f"Other[{idx}]"  # 标注索引
            
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
                    # 注意：不再使用 "只有一个人脸在框内就认为是目标" 的逻辑
                    # 因为在遮挡场景下，遮挡者的人脸可能正好在目标人体框内
                    # 这会导致错误的绿框
            
            # 人脸框绘制日志
            if frame_count % 30 == 0:
                print(f"       Face[{face_idx}]: is_target_face={is_target_face}, target_face_idx={target_face_idx}")
            
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
