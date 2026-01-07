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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
from core.pan_tilt_controller import (
    PanTiltController, PDConfig, SerialServoController, VirtualServoController
)


# 云台控制配置
PAN_TILT_ENABLED = True   # 是否启用云台控制
USE_VIRTUAL_SERVO = False  # True=虚拟模式(无硬件), False=真实串口
SERVO_PORT = "COM3"        # 串口号 - 请根据实际连接修改 (当前可用: COM1, COM3)
INITIAL_PAN = 0.0          # 初始水平角度
INITIAL_TILT = -40.0       # 初始俯仰角度

# ★★★ 日志函数 (性能优化) ★★★
_debug_verbose = False  # 可在运行时修改

def debug_print(*args, **kwargs):
    """条件调试输出，关闭可提升帧率"""
    if _debug_verbose:
        print(*args, **kwargs)

# 性能优化配置
# ============================================
# 帧率优化策略说明：
# 1. 跳帧检测 - 人体/人脸/手势不需要每帧都检测
# 2. 并行检测 - 人体/人脸/手势三路并行
# 3. 输入缩放 - 缩小输入图像降低计算量
# 4. 异步捕获 - 摄像头采集和处理解耦
# ============================================
PARALLEL_DETECTION = True    # ★★★ 启用并行检测 ★★★
SKIP_FRAME_DETECTION = True  # 启用跳帧检测提升帧率
PERSON_DETECT_INTERVAL = 3   # 人体检测间隔 (每3帧检测一次) [2→3]
FACE_DETECT_INTERVAL = 3     # 人脸检测间隔 (每3帧检测一次)
GESTURE_DETECT_INTERVAL = 2  # 手势检测间隔 (每2帧检测一次) [新增!]

# 输入缩放 (降低检测分辨率以提升速度)
DETECT_SCALE = 1.0           # 检测缩放比例 (1.0=原始, 0.5=半分辨率)
GESTURE_SCALE = 0.75         # 手势检测缩放 (MediaPipe 很慢,用更小的图)

# ★★★ 性能优化开关 ★★★
DEBUG_VERBOSE = False        # 详细调试日志 (关闭可提升帧率)
SERVO_DEBUG = False          # 舵机调试日志 (关闭可提升帧率)
USE_REID_FOR_FACE_ASSIGN = False  # 人脸分配时使用ReID (关闭可大幅提升帧率)
USE_ASYNC_CAPTURE = True     # 异步摄像头捕获 (开启可大幅提升帧率)

# 循环时间跟踪 (用于自适应舵机控制)
_avg_loop_dt = 0.040  # 全局变量，初始值 40ms

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
MOTION_WEIGHT_MULTI_PERSON = 0.5  # 降低motion权重，防止转身时motion低拖累body
MOTION_WEIGHT_SINGLE_PERSON = 0.4

# 侧脸容忍度
MOTION_TRUST_THRESHOLD = 0.95
FACE_SIDE_VIEW_MIN = 0.35

# ============================================
# 转身容忍机制 (背身/侧身时人脸丢失)
# ============================================
# 当 body 很高时，即使 motion 较低也应该接受
# 这解决了用户快速转身导致 motion 下降的问题
BODY_HIGH_THRESHOLD = 0.68       # body >= 0.68 视为"高"，可以弥补 motion 不足
BODY_VERY_HIGH_THRESHOLD = 0.75  # body >= 0.75 视为"极高"，几乎可以独立判断
BODY_ALONE_ACCEPT_MULTI = 0.70   # 多人场景：body >= 0.70 且距离近(<100px)时独立接受
BODY_ALONE_ACCEPT_SINGLE = 0.65  # 单人场景：body >= 0.65 时独立接受


# ============================================
# 异步摄像头捕获类 (解决 cap.read 阻塞问题)
# ============================================
class AsyncVideoCapture:
    """异步摄像头捕获，独立线程读取摄像头，主循环直接获取最新帧"""
    
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区
        
        self.frame = None
        self.ret = False
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
    
    def start(self):
        """启动异步捕获线程"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        # 等待第一帧 (最多等待 2 秒)
        for _ in range(20):
            time.sleep(0.1)
            with self.lock:
                if self.frame is not None:
                    break
        return self
    
    def _capture_loop(self):
        """持续读取摄像头帧"""
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
    
    def read(self):
        """获取最新帧 (非阻塞)"""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def release(self):
        """释放资源"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()


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
    
    # 大尺寸人脸可以弥补低置信度
    # size >= 100px 时，只要 conf >= 0.50 就算 stable
    LARGE_FACE_SIZE = 100
    LARGE_FACE_MIN_CONF = 0.50
    
    if face_size >= LARGE_FACE_SIZE and face_conf >= LARGE_FACE_MIN_CONF:
        # 大尺寸人脸：只要相似度不太低就算stable
        if face_sim is None or face_sim >= FACE_UNSTABLE_SIM:
            return 'stable'

    if face_sim is not None and face_sim >= MEDIUM_SIM_THRESHOLD:
        # 中等相似度：只要尺寸不是太小（>=20px）就算 unstable
        if face_size >= 20:
            return 'unstable'
    
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
    enhanced_reid,
    assigned_face=None,
    skip_body_reid=False,  # ★★★ 跳过身体ReID计算 ★★★
    skip_face_embedding=False  # ★★★ 新增：跳过人脸特征提取 ★★★
) -> ViewFeature:
    """提取视角特征"""
    view = ViewFeature(timestamp=time.time())
    
    px1, py1, px2, py2 = person_bbox.astype(int)
    
    # 优先使用预分配的人脸
    best_face = assigned_face
    
    # 如果没有预分配，则查找人体框内最大的人脸
    if best_face is None:
        max_face_area = 0
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
            
            if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                area = (fx2 - fx1) * (fy2 - fy1)
                if area > max_face_area:
                    max_face_area = area
                    best_face = face
    
    # 人脸特征 (可跳过以提升帧率)
    if best_face and not skip_face_embedding:
        face_feature = face_recognizer.extract_feature(
            frame, best_face.bbox, best_face.keypoints
        )
        if face_feature:
            view.has_face = True
            view.face_embedding = face_feature.embedding
    elif best_face:
        # 跳过特征提取但标记有人脸(用于边界框)
        view.has_face = True
    
    # 人体特征 (可跳过以提升帧率)
    if not skip_body_reid:
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


# ★★★ 全局线程池用于并行检测 ★★★
# 在模块加载时创建，避免每帧创建开销
_detection_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='detect')


def main():
    global _avg_loop_dt  # 声明为全局变量
    
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
    # 优化配置：减少 bins，降低 LBP 半径，确保性能
    enhanced_reid = EnhancedReIDExtractor(EnhancedReIDConfig(
        num_horizontal_parts=6,
        use_lbp=True,
        lbp_radius=1,       # 小半径更快
        lbp_points=8,
        use_geometry=True,
        h_bins=16,          # 减少 bins 加速直方图计算
        s_bins=16,
        use_illumination_normalization=True, # 启用光照归一化 (已优化)
        use_gray_world=True
    ))
    
    # 多视角识别器
    mv_config = MultiViewConfig(
        face_weight=0.6,
        body_weight=0.4,
        face_threshold=0.60,      # 人脸阈值
        body_threshold=0.40,      # Body阈值大幅降低，让上层逻辑决定是否接受
        fused_threshold=0.40,     # 融合阈值降低
        motion_weight=0.25,       # 提高运动权重
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
    
    # 打开摄像头 (支持异步模式)
    if USE_ASYNC_CAPTURE:
        cap = AsyncVideoCapture(0, 640, 480).start()
        print("[INFO] 使用异步摄像头捕获模式")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return
    
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
    print("  'p': 切换云台控制")
    print("  'q': 退出")
    print()
    
    # ============================================
    # 云台控制器初始化
    # ============================================
    pan_tilt_controller = None
    pan_tilt_active = False
    
    if PAN_TILT_ENABLED:
        print("\n[云台控制]")
        
        # 创建舵机控制器
        if USE_VIRTUAL_SERVO:
            servo_controller = VirtualServoController()
            print("  模式: 虚拟舵机 (无硬件)")
        else:
            servo_controller = SerialServoController()
            print(f"  模式: 真实舵机 (串口: {SERVO_PORT})")
        
        # 连接舵机
        if servo_controller.connect(SERVO_PORT if not USE_VIRTUAL_SERVO else None):
            # 启用舵机调试日志 (关闭可提升帧率)
            servo_controller.debug = SERVO_DEBUG
            
            # 创建云台控制器
            pd_config = PDConfig(
                kp=4.5,          # 比例增益
                kd=3.0,          # 微分增益
                deadzone=0.02,   # 死区 2% (减小以提高灵敏度)
                angle_deadzone=0.3,  # 角度死区 0.3度 (减小)
                smoothing_alpha=0.8  # 增加响应速度
            )
            pan_tilt_controller = PanTiltController(
                config=pd_config,
                frame_size=(640, 480),
                servo_controller=servo_controller
            )
            pan_tilt_active = True
            print(f"  状态: 已启用 (调试模式)")
            print(f"  初始角度: Pan={INITIAL_PAN}°, Tilt={INITIAL_TILT}°")
        else:
            print("  状态: 连接失败，已禁用")
    else:
        print("\n[云台控制] 已禁用 (PAN_TILT_ENABLED=False)")
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
    
    # ★★★ 历史窗口：追踪最近N帧检测到的人数 ★★★
    # 用于防止因遮挡/重叠导致误判单人场景
    SCENE_HISTORY_WINDOW = 15  # 追踪最近15帧（约0.5秒）
    person_count_history = []  # 最近N帧检测到的人体数
    face_count_history = []    # 最近N帧检测到的人脸数
    
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    # 性能统计
    t_det_person = 0
    t_det_face = 0
    t_det_gesture = 0
    t_reid = 0
    
    # 详细性能分析计时器
    t_match = 0  # 匹配逻辑
    t_draw = 0   # 绘制逻辑
    t_other = 0  # 其他逻辑
    t_other1 = 0  # 手势过滤
    t_other2 = 0  # 状态机
    t_other3 = 0  # 云台控制
    t_capture = 0  # 摄像头捕获
    
    while True:
        loop_start = time.time()
        t0_capture = time.time()
        ret, frame = cap.read()
        t_capture += time.time() - t0_capture
        if not ret or frame is None:
            # 异步模式下可能暂时没有新帧，短暂等待后重试
            if USE_ASYNC_CAPTURE:
                time.sleep(0.001)
                continue
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)
        
        # 计算 FPS
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30 / (now - fps_start)
            fps_start = now
            avg_dt = _avg_loop_dt * 1000  # 转为 ms
            # 更详细的性能统计
            print(f"[PERF] FPS: {fps:.1f} | LoopDt: {avg_dt:.1f}ms")
            print(f"       Capture: {t_capture*1000/30:.1f}ms")
            print(f"       Detect: Person={t_det_person*1000/30:.1f}ms Face={t_det_face*1000/30:.1f}ms Gesture={t_det_gesture*1000/30:.1f}ms")
            print(f"       Logic:  Match={t_match*1000/30:.1f}ms Draw={t_draw*1000/30:.1f}ms ReID={t_reid*1000/30:.1f}ms")
            print(f"       Other:  GestFilt={t_other1*1000/30:.1f}ms StateMach={t_other2*1000/30:.1f}ms Gimbal={t_other3*1000/30:.1f}ms")
            t_det_person = 0
            t_det_face = 0
            t_det_gesture = 0
            t_reid = 0
            t_match = 0
            t_draw = 0
            t_other = 0
            t_other1 = 0
            t_other2 = 0
            t_other3 = 0
            t_capture = 0
        
        t_section_start = time.time()
        
        # ============== 检测 (并行 + 跳帧优化) ==============
        # 策略: 人体检测、人脸检测、手势检测三路并行
        # 同时支持跳帧：只有需要检测的任务才提交到线程池
        # 这可以显著提升帧率
        
        # 判断哪些检测需要执行
        if SKIP_FRAME_DETECTION:
            do_person_detect = (frame_count % PERSON_DETECT_INTERVAL == 0)
            do_face_detect = (frame_count % FACE_DETECT_INTERVAL == 0)
            do_gesture_detect = (frame_count % GESTURE_DETECT_INTERVAL == 0)
        else:
            do_person_detect = True
            do_face_detect = True
            do_gesture_detect = True
        
        # 准备缩放后的图像
        if DETECT_SCALE < 1.0:
            detect_frame = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
        else:
            detect_frame = frame
        
        if GESTURE_SCALE < 1.0:
            gesture_frame = cv2.resize(frame, None, fx=GESTURE_SCALE, fy=GESTURE_SCALE)
        else:
            gesture_frame = frame
        
        if PARALLEL_DETECTION:
            # ★★★ 并行检测模式 (结合跳帧) ★★★
            # 使用全局线程池避免每帧创建开销
            futures = {}
            
            t0_parallel = time.time()
            
            if do_person_detect:
                futures['person'] = _detection_executor.submit(person_detector.detect, detect_frame)
            if do_face_detect:
                futures['face'] = _detection_executor.submit(face_detector.detect, detect_frame)
            if do_gesture_detect:
                futures['gesture'] = _detection_executor.submit(gesture_detector.detect, gesture_frame)
            
            # 收集结果
            if 'person' in futures:
                persons = futures['person'].result()
                if DETECT_SCALE < 1.0:
                    for p in persons:
                        p.bbox = p.bbox / DETECT_SCALE
                _last_persons = persons
                t_det_person = time.time() - t0_parallel  # 近似并行时间
            else:
                persons = _last_persons if '_last_persons' in dir() else []
            
            if 'face' in futures:
                faces = futures['face'].result()
                if DETECT_SCALE < 1.0:
                    for f in faces:
                        f.bbox = f.bbox / DETECT_SCALE
                        if f.keypoints is not None:
                            f.keypoints = f.keypoints / DETECT_SCALE
                _last_faces = faces
                t_det_face = time.time() - t0_parallel  # 近似并行时间
            else:
                faces = _last_faces if '_last_faces' in dir() else []
            
            if 'gesture' in futures:
                gesture = futures['gesture'].result()
                if GESTURE_SCALE < 1.0 and gesture.hand_bbox is not None:
                    gesture.hand_bbox = gesture.hand_bbox / GESTURE_SCALE
                _last_gesture = gesture
                t_det_gesture = time.time() - t0_parallel  # 近似并行时间
            else:
                gesture = _last_gesture if '_last_gesture' in dir() else GestureResult(gesture_type=GestureType.NONE, confidence=0.0, hand_bbox=None)
            
            # 总并行检测时间
            t_parallel_total = time.time() - t0_parallel
            # 调整时间统计：并行时总时间 ≈ max(各检测时间)
            # 但我们展示的是实际耗时
            if do_person_detect or do_face_detect or do_gesture_detect:
                # 重新分配时间以反映真实情况
                if do_person_detect: t_det_person = t_parallel_total
                if do_face_detect: t_det_face = 0  # 并行执行，不额外计时
                if do_gesture_detect: t_det_gesture = 0  # 并行执行，不额外计时
        else:
            # 串行检测模式 (保留作为回退)
            # 人体检测
            if do_person_detect:
                t0 = time.time()
                persons = person_detector.detect(detect_frame)
                t_det_person += time.time() - t0
                if DETECT_SCALE < 1.0:
                    for p in persons:
                        p.bbox = p.bbox / DETECT_SCALE
                _last_persons = persons
            else:
                persons = _last_persons if '_last_persons' in dir() else []
            
            # 人脸检测
            if do_face_detect:
                t0 = time.time()
                faces = face_detector.detect(detect_frame)
                t_det_face += time.time() - t0
                if DETECT_SCALE < 1.0:
                    for f in faces:
                        f.bbox = f.bbox / DETECT_SCALE
                        if f.keypoints is not None:
                            f.keypoints = f.keypoints / DETECT_SCALE
                _last_faces = faces
            else:
                faces = _last_faces if '_last_faces' in dir() else []
            
            # 手势检测
            if do_gesture_detect:
                t0 = time.time()
                gesture = gesture_detector.detect(gesture_frame)
                t_det_gesture += time.time() - t0
                if GESTURE_SCALE < 1.0 and gesture.hand_bbox is not None:
                    gesture.hand_bbox = gesture.hand_bbox / GESTURE_SCALE
                _last_gesture = gesture
            else:
                gesture = _last_gesture if '_last_gesture' in dir() else GestureResult(gesture_type=GestureType.NONE, confidence=0.0, hand_bbox=None)
        
        # ★★★ 检测阶段结束，重新设置计时起点 ★★★
        t_section_start = time.time()
        
        # 重置 ReID 计时 (在循环内累加)
        # t_reid 在 extract_view_feature 中被间接调用，这里无法直接统计
        # 我们可以在 extract_view_feature 前后加计时，或者只统计总循环时间
        
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
            if DEBUG_VERBOSE and gesture_reject_reason and frame_count % 30 == 0:
                if DEBUG_VERBOSE: print(f"[DEBUG] 手势过滤: {gesture_reject_reason}")
            # 创建一个无效手势结果
            gesture = GestureResult(gesture_type=GestureType.NONE, confidence=0.0, hand_bbox=None)
        
        # 调试日志 (每30帧输出一次) - 仅在 DEBUG_VERBOSE 时输出
        if DEBUG_VERBOSE and frame_count % 30 == 0:
            if DEBUG_VERBOSE: print(f"[DEBUG] Frame {frame_count}: persons={len(persons)}, faces={len(faces)}, gesture={gesture.gesture_type.value}")
            if faces:
                for i, face in enumerate(faces):
                    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                    face_size = min(fx2-fx1, fy2-fy1)
                    print(f"        Face[{i}]: bbox={[fx1,fy1,fx2,fy2]}, conf={face.confidence:.2f}, size={face_size}px")
            if persons:
                for i, person in enumerate(persons):
                    print(f"        Person[{i}]: bbox={person.bbox.astype(int).tolist()}, conf={person.confidence:.2f}")
        
        # 检测阶段结束，记录时间
        t_other1 += time.time() - t_section_start
        t_other += time.time() - t_section_start
        t_section_start = time.time()
        
        # ============== 手势状态机 (持续时间检测) ==============
        current_time = time.time()
        old_state = state_machine.state
        
        # 处理手势 (需要持续 GESTURE_HOLD_DURATION 秒)
        state_changed = state_machine.process_gesture(gesture.gesture_type, current_time, debug=False)
        
        # 获取持续进度
        hold_progress = state_machine.get_gesture_hold_progress()
        
        # 状态机调试日志 - 仅在 DEBUG_VERBOSE 时输出
        if DEBUG_VERBOSE and hold_progress > 0 and frame_count % 10 == 0:
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
                        if DEBUG_VERBOSE: print(f"[DEBUG] 手势框: {gesture.hand_bbox.astype(int).tolist()}")
                        for pi, p in enumerate(persons):
                            px1, py1, px2, py2 = p.bbox.astype(int)
                            hc = ((gesture.hand_bbox[0] + gesture.hand_bbox[2]) / 2,
                                  (gesture.hand_bbox[1] + gesture.hand_bbox[3]) / 2)
                            in_box = px1 <= hc[0] <= px2 and py1 <= hc[1] <= py2
                            if DEBUG_VERBOSE: print(f"[DEBUG] Person[{pi}] bbox: [{px1}, {py1}, {px2}, {py2}], 手势在框内: {in_box}")
                            
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
                            if DEBUG_VERBOSE: print(f"[DEBUG] 锁定 Person[{target_idx}]: bbox={target_person.bbox.astype(int).tolist()}")
                            view = extract_view_feature(
                                frame, target_person.bbox, faces, 
                                face_recognizer, enhanced_reid,
                                assigned_face=face_in_target
                            )
                            if DEBUG_VERBOSE: print(f"[DEBUG] 提取特征: has_face={view.has_face}, has_body={view.part_color_hists is not None}")
                            if view.has_face and view.face_embedding is not None:
                                if DEBUG_VERBOSE: print(f"[DEBUG] 人脸embedding: shape={view.face_embedding.shape}, norm={np.linalg.norm(view.face_embedding):.3f}")
                                mv_recognizer.set_target(view, target_person.bbox)
                                mv_recognizer.clear_match_history()
                                # 清空历史窗口，新目标开始重新统计
                                person_count_history.clear()
                                face_count_history.clear()
                                lost_frames = 0
                                target_locked = True
                                print(f"[手势启动] 目标已锁定 (人体+人脸)")
                                
                                # 启动云台追踪
                                if pan_tilt_controller and pan_tilt_active:
                                    pan_tilt_controller.start_tracking(INITIAL_PAN, INITIAL_TILT)
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
                        if DEBUG_VERBOSE: print(f"[DEBUG] 仅人脸模式: bbox={best_face.bbox.astype(int).tolist()}, conf={best_face.confidence:.2f}, size={face_size}px")
                        
                        # 用人脸框扩展为伪人体框
                        pseudo_bbox = np.array([
                            max(0, fx1 - face_w * 0.5),
                            fy1,
                            min(w, fx2 + face_w * 0.5),
                            min(h, fy2 + face_h * 5)
                        ])
                        if DEBUG_VERBOSE: print(f"[DEBUG] 伪人体框: {pseudo_bbox.astype(int).tolist()}")
                        
                        # 提取人脸特征
                        view = ViewFeature(timestamp=time.time())
                        view.has_face = True
                        face_feature = face_recognizer.extract_feature(
                            frame, best_face.bbox, best_face.keypoints
                        )
                        if face_feature and face_feature.embedding is not None:
                            view.face_embedding = face_feature.embedding
                            if DEBUG_VERBOSE: print(f"[DEBUG] 人脸特征: shape={face_feature.embedding.shape}, norm={np.linalg.norm(face_feature.embedding):.3f}")
                            
                            # 设置目标：用伪人体框作为视角范围，但用人脸框作为 last_bbox（用于motion计算）
                            mv_recognizer.set_target(view, pseudo_bbox)
                            # 覆盖 last_bbox 为人脸框（更精确的motion计算）
                            mv_recognizer.target.last_bbox = best_face.bbox.copy()
                            mv_recognizer.clear_match_history()
                            # 清空历史窗口，新目标开始重新统计
                            person_count_history.clear()
                            face_count_history.clear()
                            lost_frames = 0
                            target_locked = True
                            print(f"[手势启动] 目标已锁定 (仅人脸模式，等待人体补充)")
                            
                            # 启动云台追踪
                            if pan_tilt_controller and pan_tilt_active:
                                pan_tilt_controller.start_tracking(INITIAL_PAN, INITIAL_TILT)
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
                
                # 停止云台追踪
                if pan_tilt_controller and pan_tilt_active:
                    pan_tilt_controller.stop_tracking()
                
                print("[手势停止] 跟随已停止")
        
        # 手势状态机结束，记录时间
        t_other2 += time.time() - t_section_start
        t_other += time.time() - t_section_start
        t_section_start = time.time()
        
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
        
        # ★★★ 更新历史窗口 ★★★
        person_count_history.append(num_persons)
        face_count_history.append(num_faces)
        if len(person_count_history) > SCENE_HISTORY_WINDOW:
            person_count_history.pop(0)
            face_count_history.pop(0)
        
        # 检查单脸+单人体时，脸是否在人体框内
        face_in_person_for_scene = False
        if num_faces == 1 and num_persons == 1:
            fx1, fy1, fx2, fy2 = faces[0].bbox.astype(int)
            fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
            px1, py1, px2, py2 = persons[0].bbox.astype(int)
            face_in_person_for_scene = (px1 <= fc_x <= px2 and py1 <= fc_y <= py2)
        
        # ★★★ 安全的单人场景判断 ★★★
        # 核心原则：只要最近检测到过多人，就不能轻易认为是单人场景
        # 这样可以防止因遮挡/重叠导致误判单人场景，从而学习错误特征
        
        # 基础判断（仅看当前帧）
        current_frame_single = False
        if num_persons == 0 and num_faces == 0:
            current_frame_single = True  # 没人
        elif num_persons == 0 and num_faces == 1:
            current_frame_single = True  # 只有单脸
        elif num_persons == 1 and num_faces == 0:
            current_frame_single = True  # 只有单人体
        elif num_persons == 1 and num_faces == 1 and face_in_person_for_scene:
            current_frame_single = True  # 单脸+单人体，脸在框内
        
        # 历史感知：检查最近是否检测到过多人
        recent_max_persons = max(person_count_history) if person_count_history else num_persons
        recent_max_faces = max(face_count_history) if face_count_history else num_faces
        recently_had_multi = recent_max_persons > 1 or recent_max_faces > 1
        
        # 最终判断：
        # - 当前帧是单人 且 最近没有检测到多人 → 单人场景
        # - 否则 → 多人场景（保守策略）
        if current_frame_single and not recently_had_multi:
            is_single_person_scene = True
        else:
            is_single_person_scene = False
        
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
                if DEBUG_VERBOSE: print(f"[DEBUG] ⚠️ 检测到交汇场景 (IoU={crossing_iou:.2f}), 启用严格匹配模式")
        
        # ============================================
        # 浮动人脸检测：有大脸不在任何人体框内
        # ============================================
        # 场景：目标靠近摄像头，人体检测失败但脸部可见
        # 风险：如果信任motion匹配其他人体，会发生ID switch
        # 策略：检测到浮动大脸时，禁止仅motion匹配人体
        floating_face_detected = False
        floating_face_idx = -1
        FLOATING_FACE_MIN_SIZE = 100  # 大脸定义：>=100px
        
        if faces:
            for fi, face in enumerate(faces):
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                face_size = min(fx2 - fx1, fy2 - fy1)
                
                if face_size >= FLOATING_FACE_MIN_SIZE:
                    # 检查这个大脸是否在任何人体框内
                    face_in_any_person = False
                    face_cx, face_cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                    
                    for person in persons:
                        px1, py1, px2, py2 = person.bbox.astype(int)
                        if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
                            face_in_any_person = True
                            break
                    
                    if not face_in_any_person:
                        floating_face_detected = True
                        floating_face_idx = fi
                        if DEBUG_VERBOSE and frame_count % 30 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] ⚠️ 检测到浮动大脸 Face[{fi}] (size={face_size}px, 不在任何人体框内)")
                        break  # 只检测第一个浮动大脸
        
        if state_machine.state == SystemState.TRACKING:
            matched_any = False
            
            # ★★★ ReID 跳帧优化 (极致性能) ★★★
            # 核心思路：ReID 很慢(~50ms)，尽量少算
            # 单人单脸：不需要ReID，完全依靠人脸+motion
            # 多人/多脸：需要ReID辅助区分
            REID_INTERVAL_SINGLE = 30   # 单人场景每30帧算一次 (更激进)
            REID_INTERVAL_MULTI = 10    # 多人场景每10帧算一次
            
            # 极致优化：只有真正需要区分多人时才算 ReID
            # 条件：有多个 person 或 多个 face
            real_multi_person = num_persons > 1 or num_faces > 1
            
            if real_multi_person:
                do_reid = (frame_count % REID_INTERVAL_MULTI == 0)
            else:
                do_reid = (frame_count % REID_INTERVAL_SINGLE == 0)
            
            # ★★★ 新增：跳过人脸特征提取的条件 ★★★
            # 如果：单人场景 + 目标有人脸 + 不需要ReID → 完全跳过特征提取，仅用motion
            skip_all_features = (not real_multi_person) and (not do_reid)
            
            # ★★★ 单人快速路径：完全跳过匹配逻辑 ★★★
            # 当只有 1 个 person 且 1 个 face 时，直接假设匹配成功
            # 这是最常见的场景，可以节省大量计算
            use_fast_path = (num_persons == 1 and num_faces <= 1 and not do_reid)
            
            # 调试：每30帧输出一次场景判断和 ReID 状态
            if frame_count % 30 == 0:
                print(f"[场景] persons={num_persons}, faces={num_faces}, fast_path={use_fast_path}, do_reid={do_reid}")
            
            # ★★★ 单人快速路径：直接使用唯一的 person，跳过所有匹配逻辑 ★★★
            if use_fast_path and num_persons == 1:
                # 直接设置匹配结果，跳过所有复杂的特征提取和匹配逻辑
                matched_any = True
                target_person_idx = 0
                lost_frames = 0
                
                # 更新跟踪状态
                mv_recognizer.update_tracking(persons[0].bbox)
                
                # 设置匹配信息（简化版）- 单人场景假设完美匹配
                current_match_info = {
                    'type': 'person',
                    'similarity': 1.0,
                    'method': 'fast_path',
                    'match_type': 'fast_path',
                    'threshold': 0.0,
                    'face_sim': 1.0,      # 单人假设人脸匹配
                    'body_sim': 1.0,      # 单人假设身体匹配
                    'motion_score': 1.0   # 单人假设运动连续
                }
                
                # 重置 lost_frames（目标在视野中）
                lost_frames = 0
                
            # else 分支：原有的复杂匹配逻辑（多人场景或需要 ReID 时）
            if not use_fast_path or num_persons != 1:
                # 调试: 显示目标信息和场景类型 (仅 DEBUG_VERBOSE 时)
                if DEBUG_VERBOSE and frame_count % 30 == 0:
                    scene_type = "多人" if is_multi_person_scene else "单人"
                    extra_info = ""
                    if num_persons == 1 and num_faces == 1:
                        extra_info = f", 脸在框内={face_in_person_for_scene}"
                    # 显示历史感知信息
                    history_info = f", 历史最大P={recent_max_persons}/F={recent_max_faces}" if recently_had_multi else ""
                    if DEBUG_VERBOSE: print(f"[DEBUG] 场景: {scene_type} (persons={num_persons}, faces={num_faces}{extra_info}{history_info})")
                    if mv_recognizer.target:
                        t = mv_recognizer.target
                        if DEBUG_VERBOSE: print(f"[DEBUG] Target: num_views={t.num_views}, has_face_view={t.has_face_view}")
                        for vi, v in enumerate(t.view_features):
                            print(f"        View[{vi}]: has_face={v.has_face}, has_body={v.part_color_hists is not None}")
                        # Motion调试：显示last_bbox和position_history
                        if t.last_bbox is not None:
                            lx1, ly1, lx2, ly2 = t.last_bbox.astype(int)
                            last_cx, last_cy = (lx1 + lx2) // 2, (ly1 + ly2) // 2
                            if DEBUG_VERBOSE: print(f"[DEBUG] Motion基准: last_bbox=[{lx1},{ly1},{lx2},{ly2}], center=({last_cx},{last_cy})")
                            print(f"        position_history长度={len(t.position_history)}, last_seen={time.time() - t.last_seen_time:.1f}秒前")
            
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
            
            # ============================================
            # 预处理：将人脸唯一分配给最匹配的人体框
            # ============================================
            # 解决多人重叠时，一个人脸被多个人体框同时认领的问题
            # ★★★ 改进策略：不仅考虑距离，还要考虑身体特征匹配度 ★★★
            # 如果目标已有身体特征，优先将人脸分配给身体特征更匹配的人体框
            face_to_person_map = {}  # face_idx -> person_idx
            
            # 先计算每个人体的身体相似度（用于辅助人脸分配）
            # ★★★ 优化：使用 do_reid 条件控制是否计算 ReID ★★★
            # ★★★ 新增：USE_REID_FOR_FACE_ASSIGN 开关控制是否在人脸分配时使用 ReID ★★★
            person_body_sims = {}  # p_idx -> body_sim
            if USE_REID_FOR_FACE_ASSIGN and do_reid and mv_recognizer.target is not None and any(v.has_body for v in mv_recognizer.target.view_features):
                for p_idx, person in enumerate(persons):
                    temp_view = extract_view_feature(
                        frame, person.bbox, [], face_recognizer, enhanced_reid, None
                    )
                    # 只计算身体相似度
                    if temp_view.part_color_hists is not None and len(temp_view.part_color_hists) > 0:
                        body_sim = 0.0
                        for tv in mv_recognizer.target.view_features:
                            if tv.has_body and tv.part_color_hists is not None and len(tv.part_color_hists) > 0:
                                try:
                                    num_parts = min(len(temp_view.part_color_hists), len(tv.part_color_hists))
                                    color_sims = []
                                    for i in range(num_parts):
                                        hist1 = temp_view.part_color_hists[i]
                                        hist2 = tv.part_color_hists[i]
                                        if hist1 is not None and hist2 is not None and len(hist1) > 0 and len(hist2) > 0:
                                            sim = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
                                            color_sims.append(sim)
                                    if color_sims:
                                        color_sim = float(np.mean(color_sims))
                                        if color_sim > body_sim:
                                            body_sim = color_sim
                                except Exception as e:
                                    pass
                        person_body_sims[p_idx] = max(0.0, body_sim)
            
            for f_idx, face in enumerate(faces):
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                
                candidates = []
                for p_idx, person in enumerate(persons):
                    px1, py1, px2, py2 = person.bbox.astype(int)
                    if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                        # 计算人脸中心到人体中心的距离（归一化）
                        pc_x, pc_y = (px1 + px2) // 2, (py1 + py2) // 2
                        p_w, p_h = max(1, px2 - px1), max(1, py2 - py1)
                        
                        # 垂直方向权重更高（人脸通常在上方）
                        dist_x = abs(fc_x - pc_x) / p_w
                        dist_y = abs(fc_y - pc_y) / p_h
                        dist_score = dist_x + dist_y  # 越小越好
                        
                        # ★★★ 新增：考虑身体特征匹配度 ★★★
                        body_sim = person_body_sims.get(p_idx, 0.5)  # 默认0.5
                        
                        # 综合分数：距离(越小越好) - 身体相似度(越大越好)
                        # 身体相似度高的人体框会获得更低的综合分数
                        combined_score = dist_score - body_sim * 0.5
                        
                        candidates.append((p_idx, combined_score, dist_score, body_sim))
                
                if candidates:
                    # 选择综合分数最小的人体
                    best_p_idx, _, best_dist, best_body = min(candidates, key=lambda x: x[1])
                    face_to_person_map[f_idx] = best_p_idx
                    
                    # 如果有多个候选且身体相似度差距大，记录警告
                    if len(candidates) > 1 and frame_count % 30 == 0:
                        sorted_cands = sorted(candidates, key=lambda x: x[1])
                        if len(sorted_cands) >= 2:
                            body_gap = sorted_cands[1][3] - sorted_cands[0][3]
                            if abs(body_gap) > 0.15:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Face[{f_idx}] 分配给 Person[{best_p_idx}] (body_sim={best_body:.2f}优于其他候选)")

            for idx, person in enumerate(persons):
                t0_reid = time.time()
                
                # 查找分配给当前person的人脸
                assigned_face = None
                assigned_face_idx = None
                for f_idx, face in enumerate(faces):
                    if face_to_person_map.get(f_idx) == idx:
                        assigned_face = face
                        assigned_face_idx = f_idx
                        break
                
                # ★★★ 核心修复：多人场景中，验证分配的人脸是否真的属于目标 ★★★
                # 只在 do_reid 时才执行验证（节省计算）
                face_rejected_by_verification = False
                if do_reid and is_multi_person_scene and assigned_face is not None and mv_recognizer.target is not None:
                    # 先计算这个人脸与目标人脸库的相似度
                    temp_face_feature = face_recognizer.extract_feature(
                        frame, assigned_face.bbox, assigned_face.keypoints
                    )
                    if temp_face_feature and temp_face_feature.embedding is not None:
                        # 与目标人脸库比较
                        target_views = mv_recognizer.target.view_features
                        max_face_sim_to_target = 0.0
                        for tv in target_views:
                            if tv.has_face and tv.face_embedding is not None:
                                sim = np.dot(temp_face_feature.embedding, tv.face_embedding)
                                if sim > max_face_sim_to_target:
                                    max_face_sim_to_target = sim
                        
                        # 如果这个人脸与目标人脸库的相似度很低，说明它不是目标的人脸
                        FACE_VERIFICATION_THRESHOLD = 0.25  # 低于此值认为"明确不是目标的人脸"
                        if max_face_sim_to_target < FACE_VERIFICATION_THRESHOLD:
                            # 这个人脸是其他人的，不要用！
                            face_rejected_by_verification = True
                            if DEBUG_VERBOSE and frame_count % 30 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ★拒绝错误人脸★ 分配的人脸与目标不匹配(sim={max_face_sim_to_target:.2f}<{FACE_VERIFICATION_THRESHOLD})")
                            assigned_face = None  # 不使用这个人脸
                
                # ★★★ 极致优化：单人场景+不需要ReID时，跳过所有特征提取 ★★★
                view = extract_view_feature(
                    frame, person.bbox, faces, face_recognizer, enhanced_reid, 
                    None if face_rejected_by_verification else assigned_face,  # 被拒绝时不传人脸
                    skip_body_reid=not do_reid,  # 不计算ReID时跳过身体特征
                    skip_face_embedding=skip_all_features  # 单人场景+不需要ReID时也跳过人脸特征
                )
                t_reid += time.time() - t0_reid
                
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
                    # 计算候选位置与last_bbox的距离
                    px1, py1, px2, py2 = person.bbox.astype(int)
                    pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                    dist_str = ""
                    if mv_recognizer.target and mv_recognizer.target.last_bbox is not None:
                        lbbox = mv_recognizer.target.last_bbox
                        lcx, lcy = (lbbox[0] + lbbox[2]) / 2, (lbbox[1] + lbbox[3]) / 2
                        dist = np.sqrt((pcx - lcx)**2 + (pcy - lcy)**2)
                        dist_str = f", dist={dist:.0f}px"
                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] match: is_match={is_match}, sim={similarity:.3f}, {face_str}, B:{body_sim:.2f}, M:{motion_score:.2f}{dist_str}")
                    if DEBUG_VERBOSE: print(f"        method={method}")
                
                # ★★★ 每帧简要日志：显示所有候选的关键指标 ★★★
                # 帮助调试转身等场景，不受30帧限制
                if len(persons) >= 1:
                    px1, py1, px2, py2 = person.bbox.astype(int)
                    pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                    dist = 0
                    if mv_recognizer.target and mv_recognizer.target.last_bbox is not None:
                        lbbox = mv_recognizer.target.last_bbox
                        lcx, lcy = (lbbox[0] + lbbox[2]) / 2, (lbbox[1] + lbbox[3]) / 2
                        dist = int(np.sqrt((pcx - lcx)**2 + (pcy - lcy)**2))
                    face_str = f"F:{face_sim:.2f}" if face_sim is not None else "F:--"
                    # 每帧简短输出，便于追踪 (仅 DEBUG_VERBOSE 时)
                    if DEBUG_VERBOSE and frame_count % 10 == 0:  # 每10帧输出一次简要信息
                        print(f"[F{frame_count}] P{idx}: {face_str} B:{body_sim:.2f} M:{motion_score:.2f} d:{dist}px bbox=[{px1},{py1},{px2},{py2}]")
                
                # 忽略 mv_recognizer 的简单判断，使用下面的分层逻辑进行详细判断
                # 只要相似度不是极低（例如 > 0.2），就进入判断流程
                if similarity > 0.2:
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
                    BODY_MOTION_THRESHOLD = 0.60    # body + motion 综合阈值 (降低以提高稳定性)
                    MULTI_PERSON_BODY_THRESHOLD = 0.65  # 多人场景下仅body匹配的阈值
                    
                    # 有效人脸的定义
                    MIN_FACE_SIZE_FOR_VALID = 30    # 有效人脸最小尺寸（测试验证30px即可准确识别）
                    MIN_FACE_CONF_FOR_VALID = 0.65  # 有效人脸最低置信度
                    MIN_FACE_SIZE_RELAXED = 30      # 放宽条件的最小尺寸
                    MIN_FACE_SIZE_SUPER_RELAXED = 20  # 超级放宽：高置信度(>=0.85)时允许更小尺寸
                    HIGH_FACE_CONF_FOR_SUPER = 0.85   # 触发超级放宽的置信度阈值
                    
                    # 检查人脸尺寸是否足够大
                    face_size_valid = False
                    face_size_valid_relaxed = False  # 放宽条件（单人+高相似度）
                    face_size_valid_super = False    # 超级放宽（高置信度）
                    current_face_size = 0
                    
                    # 使用已分配的人脸（保持一致性）
                    if assigned_face:
                        fx1, fy1, fx2, fy2 = assigned_face.bbox.astype(int)
                        face_w = fx2 - fx1
                        face_h = fy2 - fy1
                        current_face_size = min(face_w, face_h)
                        face_size_valid = current_face_size >= MIN_FACE_SIZE
                        face_size_valid_relaxed = current_face_size >= MIN_FACE_SIZE_RELAXED
                        # 超级放宽：高置信度人脸允许更小尺寸
                        face_size_valid_super = (assigned_face.confidence >= HIGH_FACE_CONF_FOR_SUPER and 
                                                  current_face_size >= MIN_FACE_SIZE_SUPER_RELAXED)
                    
                    # 计算 body + motion 综合分数
                    # ★★★ 转身容忍：动态调整权重 ★★★
                    # 当 body 很高但 motion 很低时，说明可能是转身场景
                    # 此时应该更信任 body 而不是 motion
                    if is_multi_person_scene:
                        base_motion_weight = MOTION_WEIGHT_MULTI_PERSON
                    else:
                        base_motion_weight = MOTION_WEIGHT_SINGLE_PERSON
                    
                    # 动态权重：body高时降低motion权重
                    # body >= 0.75: motion_weight 减半
                    # body >= 0.68: motion_weight 减 25%
                    weight_adjusted = False
                    if body_sim >= BODY_VERY_HIGH_THRESHOLD:
                        motion_weight = base_motion_weight * 0.5
                        weight_adjusted = True
                    elif body_sim >= BODY_HIGH_THRESHOLD:
                        motion_weight = base_motion_weight * 0.75
                        weight_adjusted = True
                    else:
                        motion_weight = base_motion_weight
                    body_weight = 1.0 - motion_weight
                    body_motion_score = body_sim * body_weight + motion_score * motion_weight
                    
                    # 转身容忍日志
                    if weight_adjusted and frame_count % 30 == 0:
                        if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] 转身容忍: body高({body_sim:.2f}) → 动态权重 B:{body_weight:.2f}/M:{motion_weight:.2f} → BM={body_motion_score:.2f}")
                    
                    # ★★★ 距离修正：当motion与距离矛盾时，修正body_motion_score ★★★
                    # 问题：motion预测方向可能错误（尤其在交汇场景）
                    # 解决：计算与last_bbox的直接距离，对距离近的候选者加分
                    distance_bonus = 0.0
                    if is_multi_person_scene and mv_recognizer.target and mv_recognizer.target.last_bbox is not None:
                        px1, py1, px2, py2 = person.bbox.astype(int)
                        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                        lbbox = mv_recognizer.target.last_bbox
                        lcx, lcy = (lbbox[0] + lbbox[2]) / 2, (lbbox[1] + lbbox[3]) / 2
                        dist = np.sqrt((pcx - lcx)**2 + (pcy - lcy)**2)
                        
                        # 距离越近，bonus越高（最大0.15）
                        # dist=0 → bonus=0.15, dist=150 → bonus=0
                        MAX_BONUS_DIST = 150  # 150像素以内给bonus
                        if dist < MAX_BONUS_DIST:
                            distance_bonus = 0.15 * (1 - dist / MAX_BONUS_DIST)
                            body_motion_score += distance_bonus
                            if DEBUG_VERBOSE and frame_count % 30 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] 距离修正: dist={dist:.0f}px → bonus={distance_bonus:.2f} → BM={body_motion_score:.2f}")
                    
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
                        if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] face_size={current_face_size}px, valid={face_size_valid}, face_matched={face_matched}{relaxed_info}")
                    
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
                    if assigned_face:
                        current_face_conf = assigned_face.confidence
                    
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
                    
                    # 超级放宽：高置信度人脸(>=0.85)允许更小尺寸(20px)
                    # 或者高相似度(>=0.55)也允许
                    face_is_valid_super = (face_in_person and
                                          face_sim is not None and
                                          current_face_size >= MIN_FACE_SIZE_SUPER_RELAXED and
                                          (current_face_conf >= HIGH_FACE_CONF_FOR_SUPER or face_sim >= 0.55))
                    
                    if frame_count % 30 == 0:
                        face_str = f"F:{face_sim:.2f}" if face_sim is not None else "F:None"
                        if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] 人脸有效性: size={current_face_size}px, conf={current_face_conf:.2f}, valid={face_is_valid}, relaxed={face_is_valid_relaxed}, super={face_is_valid_super}")
                    
                    # ★★★ 人脸验证失败处理 ★★★
                    # 人脸验证失败可能有两种情况：
                    # A) 真的是其他人的脸 → 应该拒绝
                    # B) 目标转身导致侧脸/背面，相似度低 → 应该允许（如果Motion+Body足够高）
                    # 
                    # 判断依据：如果 Motion 很高 且 Body 也不错，说明运动轨迹连续，
                    # 很可能是同一个人转身，而不是切换到其他人
                    if face_rejected_by_verification:
                        # 计算距离（用于辅助判断）
                        person_dist_for_turning = float('inf')
                        if mv_recognizer.target and mv_recognizer.target.last_bbox is not None:
                            px1, py1, px2, py2 = person.bbox.astype(int)
                            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                            lbbox = mv_recognizer.target.last_bbox
                            lcx, lcy = (lbbox[0] + lbbox[2]) / 2, (lbbox[1] + lbbox[3]) / 2
                            person_dist_for_turning = np.sqrt((pcx - lcx)**2 + (pcy - lcy)**2)
                        
                        # 转身容忍条件（多层次）
                        # 条件1：高Motion + 中等Body
                        HIGH_MOTION_FOR_TURNING = 0.55  # 降低阈值
                        MIN_BODY_FOR_TURNING = 0.45     # 降低阈值
                        # 条件2：中等Motion + 高Body + 距离近
                        MED_MOTION_FOR_TURNING = 0.40
                        MED_BODY_FOR_TURNING = 0.55
                        MAX_DIST_FOR_TURNING = 80       # 80像素以内
                        # 条件3：Body极高（独立判断）
                        VERY_HIGH_BODY_FOR_TURNING = 0.65
                        
                        turning_allowed = False
                        turning_reason = ""
                        
                        if motion_score >= HIGH_MOTION_FOR_TURNING and body_sim >= MIN_BODY_FOR_TURNING:
                            turning_allowed = True
                            turning_reason = f"M:{motion_score:.2f}>={HIGH_MOTION_FOR_TURNING} & B:{body_sim:.2f}>={MIN_BODY_FOR_TURNING}"
                        elif (motion_score >= MED_MOTION_FOR_TURNING and 
                              body_sim >= MED_BODY_FOR_TURNING and 
                              person_dist_for_turning < MAX_DIST_FOR_TURNING):
                            turning_allowed = True
                            turning_reason = f"M:{motion_score:.2f} & B:{body_sim:.2f} & dist:{person_dist_for_turning:.0f}px<{MAX_DIST_FOR_TURNING}"
                        elif body_sim >= VERY_HIGH_BODY_FOR_TURNING:
                            turning_allowed = True
                            turning_reason = f"B:{body_sim:.2f}>={VERY_HIGH_BODY_FOR_TURNING} (高body独立通过)"
                        
                        if turning_allowed:
                            # 可能是目标转身，不要因为人脸验证失败而拒绝
                            # 走 body_motion 匹配路径
                            if DEBUG_VERBOSE and frame_count % 30 == 0 or frame_count % 10 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] 人脸验证失败但转身容忍通过 ({turning_reason})")
                            # 继续进入下面的决策逻辑，但标记人脸无效
                            face_is_valid = False
                            face_is_valid_relaxed = False
                            face_is_valid_super = False
                            face_rejected_by_verification = False  # 清除标记，允许走正常流程
                        else:
                            # Motion/Body 不够高，确实可能是其他人
                            accept = False
                            match_type = ""
                            persons_rejected_by_face_mismatch += 1
                            if DEBUG_VERBOSE and frame_count % 30 == 0 or frame_count % 10 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗✗ 人脸验证失败且转身容忍未通过 (M={motion_score:.2f}, B={body_sim:.2f}, dist={person_dist_for_turning:.0f}px)")
                    elif face_is_valid:
                        # ========== 有效人脸：基于相似度分层 ==========
                        if face_sim >= FACE_HIGH_THRESHOLD:
                            # Layer 1: F >= 0.65 → 高置信度，仅靠人脸
                            # ★★★ 修复：小人脸(30-40px)需要更高置信度(>0.72)才能仅靠人脸 ★★★
                            # ★★★ 但如果F>=0.70，即使在交汇场景也应该信任（因为人脸特征很明确）★★★
                            if current_face_size < 40 and face_sim < 0.72:
                                # 降级为 Layer 2
                                # 交汇场景保护：如果Body不匹配，拒绝小人脸
                                # 但如果F>=0.70，人脸很可靠，即使Body低也接受
                                if is_crossing_scene and body_sim < 0.60 and face_sim < 0.70:
                                    accept = False
                                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                                        if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 小人脸交汇场景Body不足 (F:{face_sim:.2f}, B:{body_sim:.2f}<0.60)，拒绝")
                                elif face_sim >= 0.70:
                                    # 人脸很可靠，直接信任
                                    accept = True
                                    match_type = "face"
                                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                                        if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 小人脸高置信度F>=0.70 (F:{face_sim:.2f}, size={current_face_size}px) → 信任人脸")
                                elif motion_score >= 0.5 or body_motion_score >= BODY_MOTION_THRESHOLD:
                                    accept = True
                                    match_type = "face_motion"
                                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                                        if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 小人脸高置信度降级 (F:{face_sim:.2f}, size={current_face_size}px) → face+motion")
                                else:
                                    accept = False
                                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                                        if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 小人脸高置信度但Motion不足 (F:{face_sim:.2f}, size={current_face_size}px, M:{motion_score:.2f})")
                            else:
                                accept = True
                                match_type = "face"
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 有效人脸高置信度 (F:{face_sim:.2f}>=0.65) → face_priority")
                        elif face_sim >= FACE_MEDIUM_THRESHOLD:
                            # Layer 2: 0.45 <= F < 0.65 → 中等置信度，需要motion辅助
                            
                            # 交汇场景保护：如果Body不匹配，拒绝中等置信度人脸
                            # 稍微放宽阈值(0.55)，因为人脸置信度尚可
                            if is_crossing_scene and body_sim < 0.55:
                                accept = False
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 交汇场景Body不足 (F:{face_sim:.2f}, B:{body_sim:.2f}<0.55)，拒绝FaceMotion")
                            # 要求 motion >= 0.5 或 综合分数够高，或者Body高(>0.65)
                            # ★★★ 改进：中等人脸 + 高Body(>0.65) 也应该接受 ★★★
                            # 因为转身时 motion 可能为 0（快速移动），但 body 相似度能说明是同一人
                            elif motion_score >= 0.5 or body_motion_score >= BODY_MOTION_THRESHOLD or body_sim > 0.65:
                                accept = True
                                match_type = "face_motion"
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 有效人脸中等置信度 (F:{face_sim:.2f}, M:{motion_score:.2f}, B:{body_sim:.2f}) → face+motion")
                            else:
                                accept = False
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 有效人脸中等但motion/body不足 (F:{face_sim:.2f}, M:{motion_score:.2f}, B:{body_sim:.2f})")
                        elif face_sim >= FACE_LOW_THRESHOLD:
                            # Layer 3: 0.30 <= F < 0.45 → 低置信度，需要body+motion
                            
                            # ★★★ 交汇场景保护：如果Body不匹配，拒绝低置信度人脸 ★★★
                            if is_crossing_scene and body_sim < 0.60:
                                accept = False
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 交汇场景Body不足 (F:{face_sim:.2f}, B:{body_sim:.2f}<0.60)，拒绝Layer3")
                            elif body_motion_score >= BODY_MOTION_THRESHOLD or body_sim > 0.65:
                                accept = True
                                match_type = "body_motion"
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 有效人脸低置信度 (F:{face_sim:.2f}) + Body可信(B:{body_sim:.2f}) → body+motion")
                            else:
                                accept = False
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 有效人脸低置信度 (F:{face_sim:.2f}) 且body+motion不足")
                        else:
                            # Layer 4: F < 0.30 → 明确不匹配，拒绝！
                            # ★★★ 核心修复：即使body+motion高也拒绝 ★★★
                            # 改进：如果运动一致性极高(>0.85)，可能是侧脸/模糊脸导致低分，不要直接拒绝
                            # 进一步改进：如果Motion不错(>0.60)且Body也还行(>0.50)，也信任
                            
                            # ★★★ 交汇场景保护：如果Body不匹配，拒绝低分人脸 ★★★
                            if is_crossing_scene and body_sim < 0.60:
                                accept = False
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 交汇场景Body不足 (F:{face_sim:.2f}, B:{body_sim:.2f}<0.60)，拒绝MotionTrust")
                            elif motion_score > 0.85 or (motion_score > 0.60 and body_sim > 0.50):
                                # 信任运动，降级为 body+motion 匹配
                                if body_motion_score >= BODY_MOTION_THRESHOLD:
                                    accept = True
                                    match_type = "body_motion"
                                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                                        if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 有效人脸低分但Motion/Body可信 (F:{face_sim:.2f}, M:{motion_score:.2f}, B:{body_sim:.2f}) → 信任Motion")
                                else:
                                    accept = False
                                    persons_rejected_by_face_mismatch += 1
                                    if frame_count % 30 == 0:
                                        scene_type = "多人" if is_multi_person_scene else "单人"
                                        if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗✗ {scene_type}有效人脸不匹配且Motion不足 (F:{face_sim:.2f}, M:{motion_score:.2f}) → 拒绝")
                            else:
                                accept = False
                                persons_rejected_by_face_mismatch += 1
                                if frame_count % 30 == 0:
                                    scene_type = "多人" if is_multi_person_scene else "单人"
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗✗ {scene_type}有效人脸明确不匹配 (F:{face_sim:.2f}<0.30) → 直接拒绝")
                    
                    elif (face_is_valid_relaxed or face_is_valid_super) and face_sim is not None and face_sim >= FACE_HIGH_THRESHOLD:
                        # ========== 放宽条件：较小人脸但高相似度 ==========
                        # ★★★ 修复：小人脸(20-30px)需要更高置信度(>0.72) ★★★
                        if current_face_size < 30 and face_sim < 0.72:
                            # 降级为需要 Motion 辅助
                            if motion_score >= 0.5 or body_motion_score >= BODY_MOTION_THRESHOLD:
                                accept = True
                                match_type = "face_motion"
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 放宽小人脸降级 (F:{face_sim:.2f}, size={current_face_size}px) → face+motion")
                            else:
                                accept = False
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 放宽小人脸Motion不足 (F:{face_sim:.2f}, size={current_face_size}px, M:{motion_score:.2f})")
                        else:
                            accept = True
                            match_type = "face"
                            if DEBUG_VERBOSE and frame_count % 30 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 放宽有效人脸高置信度 (F:{face_sim:.2f}>=0.65, size={current_face_size}px)")
                    
                    elif face_is_valid_relaxed and face_sim is not None and face_sim < FACE_REJECT_THRESHOLD:
                        # ========== 放宽条件：较小人脸但明确不匹配 ==========
                        
                        # ★★★ 交汇场景特殊保护 ★★★
                        # 如果是交汇场景，且人脸明确不匹配，绝对不能信任 Motion！
                        # 因为交汇时，遮挡者可能正好在预测位置，且有人脸
                        if is_crossing_scene:
                            accept = False
                            persons_rejected_by_face_mismatch += 1
                            if DEBUG_VERBOSE and frame_count % 30 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗✗ 交汇场景人脸不匹配 (F:{face_sim:.2f})，拒绝MotionTrust")
                        
                        # 改进：同样增加 Motion Trust 保护，但要求 Body 也不太差
                        elif motion_score > 0.85 or (motion_score > 0.60 and body_sim > 0.50):
                            if body_motion_score >= BODY_MOTION_THRESHOLD:
                                accept = True
                                match_type = "body_motion"
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 较小人脸低分但Motion/Body可信 (F:{face_sim:.2f}, M:{motion_score:.2f}, B:{body_sim:.2f}) → 信任Motion")
                            else:
                                accept = False
                                persons_rejected_by_face_mismatch += 1
                                if frame_count % 30 == 0:
                                    scene_type = "多人" if is_multi_person_scene else "单人"
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗✗ {scene_type}较小人脸不匹配且Motion不足 (F:{face_sim:.2f}, M:{motion_score:.2f}) → 拒绝")
                        else:
                            accept = False
                            persons_rejected_by_face_mismatch += 1
                            if frame_count % 30 == 0:
                                scene_type = "多人" if is_multi_person_scene else "单人"
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗✗ {scene_type}较小人脸明确不匹配 (F:{face_sim:.2f}<0.30, size={current_face_size}px) → 拒绝")
                    
                    elif body_motion_matched:
                        # ========== 无有效人脸：靠 body + motion ==========
                        
                        # ★★★ 重要：小人脸（<30px）的F值不可靠，不能用于拒绝决策！★★★
                        # 只有"放宽有效"的人脸（>=30px）才能用F<0.30来判断不匹配
                        # 小人脸的低F值可能是特征提取不准，而不是真的不匹配
                        
                        if face_is_valid_relaxed and face_sim is not None and face_sim < FACE_REJECT_THRESHOLD:
                            # 放宽有效的人脸（>=30px），F<0.30 → 明确不匹配
                            
                            # ★★★ 交汇场景特殊保护 ★★★
                            if is_crossing_scene:
                                accept = False
                                persons_rejected_by_face_mismatch += 1
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗✗ 交汇场景人脸不匹配 (F:{face_sim:.2f})，拒绝MotionTrust")
                            
                            # 改进：Motion Trust 保护
                            elif motion_score > 0.85 or (motion_score > 0.60 and body_sim > 0.50):
                                accept = True # body_motion_matched 已经是 True
                                match_type = "body_motion"
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 人脸低分但Motion/Body可信 (F:{face_sim:.2f}, M:{motion_score:.2f}, B:{body_sim:.2f}) → 信任Motion")
                            else:
                                accept = False
                                persons_rejected_by_face_mismatch += 1
                                if frame_count % 30 == 0:
                                    scene_type = "多人" if is_multi_person_scene else "单人"
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗✗ {scene_type}人脸明确不匹配 (F:{face_sim:.2f}<0.30, size={current_face_size}px>=30) → 拒绝")
                        # 交汇场景特殊处理
                        elif is_crossing_scene and target_has_face:
                            # 交汇时没有有效人脸验证 → 宁可短暂丢失
                            # 但如果 Motion 极高且人脸相似度尚可，可以信任
                            # 改进：放宽条件，允许 Motion>0.80 或 Motion>0.60+Body>0.60
                            # ★★★ 进一步改进：交汇时必须有一定Body相似度，防止完全靠Motion切到遮挡者 ★★★
                            # 修复：提高Body阈值到0.60，防止Motion=1.00但Body=0.51的错误匹配
                            # ★★★ 新增：如果 face_is_valid_super (F>=0.55, size>=20px)，允许通过 ★★★
                            if face_is_valid_super and face_sim >= 0.55:
                                # 小人脸但特征明确，可以信任
                                accept = True
                                match_type = "face_motion"
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 交汇场景小人脸特征明确 (F:{face_sim:.2f}, size={current_face_size}px) → face+motion")
                            elif (motion_score > 0.80 and body_sim > 0.60) or (motion_score > 0.60 and body_sim > 0.60):
                                accept = True
                                match_type = "body_motion"
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 交汇场景Motion/Body可信 (M:{motion_score:.2f}, B:{body_sim:.2f}) → 信任Motion")
                            else:
                                if frame_count % 30 == 0:
                                    face_str = f"F:{face_sim:.2f}" if face_sim is not None else "F:None"
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ⚠️ 交汇场景无有效人脸({face_str}, size={current_face_size}px)且Body不足(B:{body_sim:.2f}), 暂停匹配")
                                accept = False
                        # ★★★ 浮动大脸保护：如果画面有不在人体框内的大脸，禁止仅motion匹配 ★★★
                        # 场景：目标靠近摄像头，人体检测失败但脸部可见，此时另一个人在画面边缘
                        # 风险：如果信任motion匹配边缘人体，会发生ID switch
                        elif floating_face_detected and not face_in_person:
                            if DEBUG_VERBOSE and frame_count % 30 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 浮动大脸场景，此人无人脸，禁止motion匹配")
                            accept = False
                        # 多人场景：body阈值提高
                        elif is_multi_person_scene and target_has_face and body_sim < MULTI_PERSON_BODY_THRESHOLD - 0.01:
                            # 改进：如果 motion 较高 (>=0.60)，说明位置预测准确，可以放宽 body 阈值
                            # 这解决了用户转身/走近镜头时，body特征变化大导致丢失的问题
                            # 但要求 body_sim 至少有一定相似度 (>0.40) 以防完全错误
                            # 进一步改进：如果 Motion 极高 (>0.90)，允许更低的 Body (>0.35)
                            if (motion_score >= 0.60 and body_sim > 0.40) or (motion_score > 0.90 and body_sim > 0.35):
                                accept = True
                                match_type = "body_motion"
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 多人场景body低但motion高 (B:{body_sim:.2f}, M:{motion_score:.2f}) → 信任motion")
                            else:
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 多人场景无有效人脸且body不足({body_sim:.2f}<{MULTI_PERSON_BODY_THRESHOLD-0.01:.2f})且motion不足")
                                accept = False
                        else:
                            accept = True
                            match_type = "body_motion"
                            if DEBUG_VERBOSE and frame_count % 30 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 无有效人脸，body+motion通过 (B:{body_sim:.2f}+M:{motion_score:.2f}={body_motion_score:.2f})")
                    else:
                        # ========== 什么都不够，但检查转身容忍 ==========
                        # ★★★ 转身保底：body 极高时独立接受 ★★★
                        # 场景：用户快速转身，motion=0，但 body=0.75+
                        # 这种情况 body 可以独立说明是同一人
                        
                        # 计算距离（用于辅助判断）
                        person_dist = float('inf')
                        if mv_recognizer.target and mv_recognizer.target.last_bbox is not None:
                            px1, py1, px2, py2 = person.bbox.astype(int)
                            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                            lbbox = mv_recognizer.target.last_bbox
                            lcx, lcy = (lbbox[0] + lbbox[2]) / 2, (lbbox[1] + lbbox[3]) / 2
                            person_dist = np.sqrt((pcx - lcx)**2 + (pcy - lcy)**2)
                        
                        # 转身保底条件
                        body_alone_accept = False
                        if is_multi_person_scene:
                            # 多人场景：body >= 0.70 且距离近(<150px)
                            if body_sim >= BODY_ALONE_ACCEPT_MULTI and person_dist < 150:
                                body_alone_accept = True
                            # 或者 body 极高(>=0.75)，无论距离
                            elif body_sim >= BODY_VERY_HIGH_THRESHOLD:
                                body_alone_accept = True
                        else:
                            # 单人场景：body >= 0.65 直接接受
                            if body_sim >= BODY_ALONE_ACCEPT_SINGLE:
                                body_alone_accept = True
                        
                        if body_alone_accept:
                            accept = True
                            match_type = "body_alone"
                            if DEBUG_VERBOSE and frame_count % 30 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✓ 转身保底: body高({body_sim:.2f})且距离近({person_dist:.0f}px) → 独立接受")
                        else:
                            if frame_count % 30 == 0:
                                face_str = f"F:{face_sim:.2f}" if face_sim is not None else "F:None"
                                if DEBUG_VERBOSE: print(f"[DEBUG] Person[{idx}] ✗ 无有效人脸且body+motion不足 ({face_str}, B:{body_sim:.2f}, M:{motion_score:.2f}, BM:{body_motion_score:.2f}, dist:{person_dist:.0f}px)")
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
                
                best_face_match = None
                if matches_by_face:
                    # 人脸匹配中，优先选 motion 高的（轨迹一致性）
                    # 排序依据: face_sim * 0.6 + motion * 0.4
                    best_face_match = max(matches_by_face, key=lambda x: (x[6] if x[6] is not None else 0) * 0.6 + x[8] * 0.4)
                
                best_body_match = None
                if matches_by_body_motion:
                    # body+motion 匹配中，多人场景强调 motion，单人场景平衡
                    if is_multi_person_scene:
                        # 多人: motion 优先（轨迹一致最重要）
                        best_body_match = max(matches_by_body_motion, key=lambda x: x[8] * 0.7 + x[7] * 0.3)
                    else:
                        # 单人: 平衡 body 和 motion
                        best_body_match = max(matches_by_body_motion, key=lambda x: x[7] * 0.5 + x[8] * 0.5)
                
                best_match = None
                
                # 冲突解决：防止低Motion的人脸匹配抢占高Motion的Body匹配
                if best_face_match and best_body_match:
                    face_motion = best_face_match[8]
                    body_motion = best_body_match[8]
                    face_sim_val = best_face_match[6] if best_face_match[6] is not None else 0.0
                    face_body_sim = best_face_match[7]  # 人脸匹配者的身体相似度
                    body_body_sim = best_body_match[7]  # body匹配者的身体相似度
                    
                    # ★★★ 新增：人脸分配错误检测 ★★★
                    # 如果人脸匹配的候选人Body相似度明显低于body匹配的候选人
                    # 说明可能是"用户的脸落入了别人的身体框"
                    # 条件：body匹配者的Body > 0.65 且比人脸匹配者高 > 0.10
                    face_assignment_suspicious = (
                        body_body_sim > 0.65 and 
                        body_body_sim - face_body_sim > 0.10 and
                        face_body_sim < 0.60
                    )
                    
                    if face_assignment_suspicious and body_motion > 0.60:
                        # 怀疑人脸分配错误，选择body匹配更好的
                        best_match = best_body_match
                        if DEBUG_VERBOSE and frame_count % 30 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] ⚠️ 疑似人脸分配错误：Face候选B:{face_body_sim:.2f} < Body候选B:{body_body_sim:.2f}，选择Body候选")
                    # 如果人脸匹配的Motion很低（<0.5），而Body匹配的Motion很高（>0.85）
                    # 且人脸相似度不是"超级高"（>0.75），则信任Motion（即原来的目标）
                    # 这防止了目标转身（Face低）时，被路人（Face偶然高）抢占
                    elif face_motion < 0.5 and body_motion > 0.85 and face_sim_val < 0.75:
                        best_match = best_body_match
                        if DEBUG_VERBOSE and frame_count % 30 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] ⚠️ 冲突解决：信任高Motion目标 (M:{body_motion:.2f}) 优于 低Motion人脸 (F:{face_sim_val:.2f}, M:{face_motion:.2f})")
                    else:
                        best_match = best_face_match
                        if DEBUG_VERBOSE and frame_count % 30 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] 选择人脸匹配 Person[{best_match[0]}] (F:{face_sim_val:.2f}, B:{face_body_sim:.2f}, M:{face_motion:.2f}, 共{len(all_person_matches)}候选)")
                elif best_face_match:
                    best_match = best_face_match
                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                        if DEBUG_VERBOSE: print(f"[DEBUG] 选择人脸匹配 Person[{best_match[0]}] (F:{best_face_match[6]:.2f}, M:{best_face_match[8]:.2f})")
                elif best_body_match:
                    best_match = best_body_match
                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                        if DEBUG_VERBOSE: print(f"[DEBUG] 选择body+motion匹配 Person[{best_match[0]}], B:{best_match[7]:.2f}, M:{best_match[8]:.2f}")
                
                if best_match:
                    # 解包: (idx, similarity, method, view, face_in_person, face_matched, face_sim, body_sim, motion_score, match_type)
                    idx, similarity, method, view, face_in_person, face_matched, match_face_sim, match_body_sim, match_motion_score, match_type = best_match
                    matched_any = True
                    target_person_idx = idx
                    lost_frames = 0
                    
                    # ★★★ 每帧简要日志 (仅 DEBUG_VERBOSE 时) ★★★
                    if DEBUG_VERBOSE and frame_count % 10 == 0:
                        px1, py1, px2, py2 = persons[idx].bbox.astype(int)
                        face_str = f"F:{match_face_sim:.2f}" if match_face_sim is not None else "F:--"
                        print(f"[F{frame_count}] ✓ P{idx}目标 {face_str} B:{match_body_sim:.2f} M:{match_motion_score:.2f} type={match_type}")
                    
                    # ★★★ 关键日志：标注最终选择的目标 (仅 DEBUG_VERBOSE 时) ★★★
                    if DEBUG_VERBOSE and frame_count % 30 == 0:
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
                        'threshold': FACE_MATCH_THRESHOLD if match_type == "face" else BODY_MOTION_THRESHOLD,
                        'face_sim': match_face_sim,
                        'body_sim': match_body_sim,
                        'motion_score': match_motion_score
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
                        if DEBUG_VERBOSE and frame_count % 60 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] 视角库已满({current_view_count})，启用替换策略")
                    
                    # 多人场景 + 没有人脸匹配 = 禁止学习
                    # 多人场景 + 人脸-人体不一致 = 禁止学习（防止关联错误导致学习污染）
                    if is_multi_person_scene and match_type != "face":
                        if DEBUG_VERBOSE and frame_count % 30 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] 多人场景无人脸匹配，禁止学习")
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
                            if DEBUG_VERBOSE and frame_count % 30 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] 多人场景人脸-人体不一致(F:{match_face_sim_check:.2f}, B:{match_body_sim_check:.2f}<{BODY_MIN_FOR_LEARN_MULTI})，禁止学习")
                            should_learn = False
                        elif face_body_gap > FACE_BODY_CONSISTENCY_GAP:
                            if DEBUG_VERBOSE and frame_count % 30 == 0:
                                if DEBUG_VERBOSE: print(f"[DEBUG] 多人场景人脸-人体差距过大(F:{match_face_sim_check:.2f}-B:{match_body_sim_check:.2f}={face_body_gap:.2f}>{FACE_BODY_CONSISTENCY_GAP})，禁止学习")
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
                        
                        # 查找分配给目标的人脸
                        target_assigned_face = None
                        for f_idx, face in enumerate(faces):
                            if face_to_person_map.get(f_idx) == idx:
                                target_assigned_face = face
                                break
                        
                        if target_assigned_face:
                            fx1, fy1, fx2, fy2 = target_assigned_face.bbox.astype(int)
                            face_w = fx2 - fx1
                            face_h = fy2 - fy1
                            current_face_size_for_learn = min(face_w, face_h)
                            face_size_ok_for_learn = current_face_size_for_learn >= MIN_FACE_SIZE_FOR_LEARN
                        
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
                                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                                        if DEBUG_VERBOSE: print(f"[DEBUG] 人脸不在人体框内，不学习body")
                            elif match_face_sim >= FACE_LEARN_THRESHOLD_LOCAL and face_size_ok_for_learn:
                                # 人脸够高 + 尺寸够大 → 直接学习当前视角
                                should_learn = True
                                learn_what = "face"
                                learn_reason = f"人脸高置信(F:{match_face_sim:.2f}, size={current_face_size_for_learn}px)"
                        
                        # Case 2: body+motion匹配通过 → 可以学习人脸
                        # 注意：方案D确保启动时一定有人脸，所以不需要"首次学习人脸"逻辑
                        elif match_type == "body_motion":
                            # ★★★ Case 2-特殊: 目标没有body视角（仅人脸启动）→ 首次学习body ★★★
                            target_has_body_view_2 = mv_recognizer.target is not None and any(v.has_body for v in mv_recognizer.target.view_features)
                            if not target_has_body_view_2 and face_in_person:
                                # 仅人脸启动的目标，即使人脸相似度低（转头），也应该学习body
                                # 条件：motion足够高 + body不太低
                                if match_motion >= 0.70 and match_body_sim >= 0.30:
                                    initial_view = mv_recognizer.target.view_features[0] if mv_recognizer.target.view_features else None
                                    if initial_view and initial_view.has_face and not initial_view.has_body:
                                        # 升级初始视角
                                        if view.part_color_hists is not None:
                                            initial_view.part_color_hists = view.part_color_hists
                                            initial_view.timestamp = time.time()
                                            print(f"[初始视角升级-motion] 仅人脸→有人体(M:{match_motion:.2f}, B:{match_body_sim:.2f})")
                                            should_learn = False
                                        else:
                                            should_learn = True
                                            learn_what = "body"
                                            learn_reason = f"motion首次学习body(M:{match_motion:.2f}, B:{match_body_sim:.2f})"
                                    else:
                                        should_learn = True
                                        learn_what = "body"
                                        learn_reason = f"motion首次学习body(M:{match_motion:.2f}, B:{match_body_sim:.2f})"
                            
                            # Case 2a: 人脸相似度够高 → 学习/更新人脸
                            elif face_in_person and match_face_sim >= FACE_MIN_FOR_BODY_LEARN and face_size_ok_for_learn:
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
                                
                                # ★★★ 严格限制：如果是靠"Motion Trust" (低Body高Motion) 匹配的，禁止学习 ★★★
                                # 防止在转身/遮挡等不稳定状态下学习错误的Body特征
                                if match_body_sim < 0.60 and match_motion > 0.80:
                                    should_learn = False
                                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                                        if DEBUG_VERBOSE: print(f"[DEBUG] 靠MotionTrust匹配(B:{match_body_sim:.2f}, M:{match_motion:.2f})，禁止学习Body")
                                else:
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
                                if DEBUG_VERBOSE and frame_count % 60 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] 当前质量({current_quality:.2f})<0.75，不替换")
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
                    # 修复：使用 confidence 而不是 score
                    face_conf = face.confidence if hasattr(face, 'confidence') else (float(face.score) if hasattr(face, 'score') else 0.5)
                    
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
                        if DEBUG_VERBOSE and frame_count % 30 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] Face[{face_idx}] 在不匹配的人体框内(多人场景)，跳过")
                        continue
                    elif num_persons > 1 and face_in_any_person and all_rejected_by_face_mismatch:
                        if DEBUG_VERBOSE and frame_count % 30 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] Face[{face_idx}] 所有人体因人脸不匹配被拒绝，尝试仅人脸匹配")
                    
                    # 远处人脸使用更高阈值
                    # ★★★ 浮动大脸例外：如果当前人脸就是浮动大脸，不应用远处惩罚 ★★★
                    # 浮动大脸场景：目标靠近摄像头，人体检测失败但人脸清晰可见
                    # 此时人脸不在任何人体框内，但它是目标的唯一身份标识
                    is_floating_face = (floating_face_detected and face_idx == floating_face_idx)
                    is_distant_face = num_persons > 0 and not face_in_any_person and not is_floating_face
                    
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
                                    # 改进：稳定人脸的候选阈值降低，允许进入后续的 motion 验证逻辑
                                    # 原来是 FACE_ONLY_THRESHOLD (0.70)，现在降为 FACE_ONLY_THRESHOLD_UNSTABLE (0.50)
                                    current_threshold = FACE_ONLY_THRESHOLD_UNSTABLE + multi_face_penalty
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
                                
                                if DEBUG_VERBOSE and frame_count % 30 == 0:
                                    if DEBUG_VERBOSE: print(f"[DEBUG] Face[{face_idx}] vs View[{vi}]: sim={sim:.3f}, conf={face_conf:.2f}, size={face_size}px, quality={face_quality}, threshold={current_threshold:.2f}")
                                
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
                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                        if DEBUG_VERBOSE: print(f"[DEBUG] 稳定人脸匹配成功! face_idx={best_face_idx}, sim={best_face_sim:.3f}")
                        
                elif (best_face_quality == 'unstable' or best_face_quality == 'stable') and best_face_sim >= 0.35:
                    # 不稳定人脸(或低分稳定人脸): 需要motion辅助验证
                    # 改进：将最低人脸阈值从 0.50 降至 0.35，以支持侧脸/坐下时的低分情况
                    # 只要 Motion 足够高，综合分数就能过
                    
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
                    
                    # 特殊情况：如果 Motion 极高 (>0.90)，允许更低的综合分
                    threshold = 0.50
                    if motion_score > 0.90:
                        threshold = 0.45
                        
                    if combined_score >= threshold:  # 综合分数阈值
                        face_match_success = True
                        if frame_count % 30 == 0:
                            quality_str = "稳定" if best_face_quality == 'stable' else "不稳定"
                            if DEBUG_VERBOSE: print(f"[DEBUG] {quality_str}人脸+motion匹配成功! face={best_face_sim:.2f}, motion={motion_score:.2f}, combined={combined_score:.2f}>={threshold}")
                    else:
                        if frame_count % 30 == 0:
                            quality_str = "稳定" if best_face_quality == 'stable' else "不稳定"
                            if DEBUG_VERBOSE: print(f"[DEBUG] {quality_str}人脸+motion不足 (face={best_face_sim:.2f}, motion={motion_score:.2f}, combined={combined_score:.2f}<{threshold})")
                
                # ★★★ 单人场景转头优化：纯运动预测信任 ★★★
                # 当人脸相似度极低（大幅转头/侧脸）但 motion 高时，信任运动连续性
                elif is_single_person_scene and len(faces) == 1:
                    # 计算motion分数
                    motion_score = 0.0
                    last_bbox = mv_recognizer.target.last_bbox if mv_recognizer.target else None
                    if last_bbox is not None:
                        face_bbox = faces[0].bbox
                        pred_bbox = last_bbox
                        
                        fc_x = (face_bbox[0] + face_bbox[2]) / 2
                        fc_y = (face_bbox[1] + face_bbox[3]) / 2
                        pc_x = (pred_bbox[0] + pred_bbox[2]) / 2
                        pc_y = (pred_bbox[1] + pred_bbox[3]) / 2
                        
                        frame_h, frame_w = frame.shape[:2]
                        dist = np.sqrt((fc_x - pc_x)**2 + (fc_y - pc_y)**2)
                        max_dist = np.sqrt(frame_w**2 + frame_h**2) * 0.3
                        motion_score = max(0, 1.0 - dist / max_dist)
                    
                    # 单人场景分级信任：
                    # - motion >= 0.80 且 lost <= 10: 完全信任
                    # - motion >= 0.70 且 lost <= 5: 部分信任
                    if motion_score >= 0.80 and lost_frames <= 10:
                        face_match_success = True
                        best_face_idx = 0
                        best_face_sim = 0.0  # 不是基于人脸匹配
                        if DEBUG_VERBOSE and frame_count % 30 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] 单人场景纯motion匹配! motion={motion_score:.2f}>=0.80, lost={lost_frames}帧")
                    elif motion_score >= 0.70 and lost_frames <= 5:
                        face_match_success = True
                        best_face_idx = 0
                        best_face_sim = 0.0
                        if DEBUG_VERBOSE and frame_count % 30 == 0:
                            if DEBUG_VERBOSE: print(f"[DEBUG] 单人场景motion辅助匹配! motion={motion_score:.2f}>=0.70, lost={lost_frames}帧")
                    elif frame_count % 30 == 0:
                        # 始终打印失败原因
                        if DEBUG_VERBOSE: print(f"[DEBUG] 单人场景motion不足: motion={motion_score:.2f}, lost={lost_frames}帧, last_bbox={last_bbox}")
                
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
                    
                    # ★★★ 仅人脸匹配时：用人脸框扩展成伪人体框再更新tracking ★★★
                    # 避免用小的人脸框覆盖last_bbox，导致后续motion计算混乱
                    face_bbox = faces[best_face_idx].bbox
                    fx1, fy1, fx2, fy2 = face_bbox.astype(int)
                    face_w, face_h = fx2 - fx1, fy2 - fy1
                    pseudo_body_bbox = np.array([
                        max(0, fx1 - face_w * 0.5),
                        fy1,
                        min(w, fx2 + face_w * 0.5),
                        min(h, fy2 + face_h * 5)
                    ])
                    mv_recognizer.update_tracking(pseudo_body_bbox)
                    
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
                            if DEBUG_VERBOSE: print(f"[DEBUG] 不稳定人脸不学习")
                        elif not is_single_person_scene:
                            if DEBUG_VERBOSE: print(f"[DEBUG] 仅人脸匹配不学习: 多人场景")
                        else:
                            if DEBUG_VERBOSE: print(f"[DEBUG] 仅人脸匹配不学习: 相似度不足({best_face_sim:.2f}<0.80)")
                            
                elif DEBUG_VERBOSE and frame_count % 30 == 0 and best_face_sim > 0:
                    if DEBUG_VERBOSE: print(f"[DEBUG] 人脸匹配失败: sim={best_face_sim:.3f}, quality={best_face_quality}")
            
            if not matched_any:
                lost_frames += 1
                # 清空匹配历史，防止误匹配
                mv_recognizer.clear_match_history()
                # ★★★ 每帧显示未匹配原因 (仅 DEBUG_VERBOSE 时) ★★★
                if DEBUG_VERBOSE and frame_count % 10 == 0:
                    print(f"[F{frame_count}] ❌ 未匹配 lost={lost_frames}/{max_lost_frames} persons={num_persons} faces={num_faces}")
                if DEBUG_VERBOSE and frame_count % 30 == 0:
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
                
                if DEBUG_VERBOSE and frame_count % 30 == 0:
                    if DEBUG_VERBOSE: print(f"[DEBUG] 重新锁定候选: Person[{idx}], sim={similarity:.2f}, 连续帧={relock_confirm_count}/{RELOCK_CONFIRM_FRAMES}")
                
                # 达到连续帧要求，确认重新锁定
                if relock_confirm_count >= RELOCK_CONFIRM_FRAMES:
                    state_machine.state = SystemState.TRACKING
                    target_person_idx = idx
                    lost_frames = 0
                    relock_confirm_count = 0
                    relock_candidate_idx = -1
                    mv_recognizer.update_tracking(persons[idx].bbox)
                    
                    # 更新 current_match_info 以便正确绘制
                    current_match_info = {
                        'type': 'person',
                        'similarity': similarity,
                        'method': method,
                        'match_type': 'relock',
                        'threshold': RELOCK_FUSED_THRESHOLD if has_face else RELOCK_BODY_THRESHOLD,
                        'face_sim': face_sim if has_face else None
                    }
                    
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
                    if DEBUG_VERBOSE and frame_count % 30 == 0:
                        if DEBUG_VERBOSE: print(f"[DEBUG] 重新锁定候选丢失，重置计数")
            
            # 禁用仅人脸重新锁定 - 太容易误识别远处的相似人脸
            # 只有当人脸在人体框内时才能通过人体+人脸联合匹配来锁定
            # 原因：仅人脸匹配缺少位置、身体特征等关联信息，容易误匹配
            # if not matched_any and faces and mv_recognizer.target and mv_recognizer.target.has_face_view:
            #     for face_idx, face in enumerate(faces):
            #         ...
        
        # 匹配逻辑结束，记录时间
        t_match += time.time() - t_section_start
        t_section_start = time.time()
        
        # ============== 云台控制更新 ==============
        # 优先让人脸居中，没有人脸时让人体上半部分居中
        # 目标位置：画面上1/3处（更符合构图习惯）
        pan_tilt_state = None
        
        # 计算循环耗时 (用于自适应舵机时间)
        loop_end = time.time()
        loop_dt = loop_end - loop_start
        
        # 更新平均循环时间 (指数移动平均)
        _avg_loop_dt = _avg_loop_dt * 0.9 + loop_dt * 0.1
        
        # ========== 预先计算目标人体对应的人脸索引 ==========
        # 这个变量在云台控制和绘制阶段都会用到，所以提前计算
        target_person_assigned_face_idx = -1
        if target_person_idx >= 0 and target_person_idx < len(persons):
            # 方式1: 使用匹配阶段建立的 face_to_person_map
            for f_idx, p_idx in face_to_person_map.items():
                if p_idx == target_person_idx:
                    target_person_assigned_face_idx = f_idx
                    break
            
            # 方式2: 如果没有分配的人脸，fallback 到框内最大人脸
            if target_person_assigned_face_idx < 0 and faces:
                px1, py1, px2, py2 = persons[target_person_idx].bbox.astype(int)
                max_area = 0
                for f_idx, f in enumerate(faces):
                    fx1, fy1, fx2, fy2 = f.bbox.astype(int)
                    fc_x, fc_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                        area = (fx2 - fx1) * (fy2 - fy1)
                        if area > max_area:
                            max_area = area
                            target_person_assigned_face_idx = f_idx
        
        if pan_tilt_controller and pan_tilt_active:
            if state_machine.state == SystemState.TRACKING and target_person_idx >= 0:
                target_person = persons[target_person_idx]
                px1, py1, px2, py2 = target_person.bbox
                
                # ========== 确定追踪点 ==========
                # 优先级1: 目标人体内的人脸 → 人脸中心
                # 优先级2: 无人脸 → 人体框上1/3处（胸部位置）
                
                track_point = None
                track_source = "body"
                
                # 查找目标人体内的人脸
                if faces and target_person_assigned_face_idx >= 0:
                    target_face = faces[target_person_assigned_face_idx]
                    fx1, fy1, fx2, fy2 = target_face.bbox.astype(int)
                    # 人脸中心
                    track_point = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
                    track_source = "face"
                
                # 没有人脸时，使用人体上1/3处
                if track_point is None:
                    body_cx = (px1 + px2) / 2
                    body_top_third = py1 + (py2 - py1) * 0.25  # 上1/4处（胸部位置）
                    track_point = (body_cx, body_top_third)
                    track_source = "body_upper"
                
                # ========== 目标位置偏移 ==========
                # 人脸应该在画面上1/3处，而不是正中央
                # 通过给云台控制器一个偏移的目标点来实现
                target_cx, target_cy = track_point
                
                # 将目标点向下偏移，使人脸最终位于画面上1/3处
                # 偏移量 = 画面高度 * (0.5 - 0.33) = 画面高度 * 0.17
                if track_source == "face":
                    # 人脸时，目标位置在画面上1/3处
                    target_cy_offset = h * 0.12  # 向下偏移12%，使人脸在上1/3
                    target_cy += target_cy_offset
                
                # 更新云台 (PD控制自动计算角度增量，传入循环耗时用于自适应)
                pan_tilt_state = pan_tilt_controller.update(
                    target_center=(target_cx, target_cy),
                    loop_dt=_avg_loop_dt
                )
                
                # 调试日志
                if frame_count % 30 == 0:
                    offset_x, offset_y = pan_tilt_controller.get_center_offset(
                        target_center=(target_cx, target_cy)
                    )
                    print(f"[云台] 追踪={track_source}, 偏移=({offset_x:+.0f}, {offset_y:+.0f}), "
                          f"Pan={pan_tilt_state.pan:+.1f}°, Tilt={pan_tilt_state.tilt:+.1f}°, "
                          f"模式={pan_tilt_state.mode}")
            
            elif state_machine.state == SystemState.LOST_TARGET:
                # 目标丢失时，保持当前位置不动
                pass
        
        # 云台控制结束，记录时间
        t_other3 += time.time() - t_section_start
        t_other += time.time() - t_section_start
        t_section_start = time.time()
        
        # ============== 绘制 ==============
        # ★★★ 绘制前日志：明确 target_person_idx 的值 (仅 DEBUG_VERBOSE 时) ★★★
        if DEBUG_VERBOSE and frame_count % 30 == 0:
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
        
        # 注意：target_person_assigned_face_idx 已在云台控制阶段提前计算

        # 绘制人脸框 - 使用之前匹配过程中的结果，避免重复计算
        # target_face_idx 是仅人脸匹配时确定的目标人脸
        # 对于有人体匹配的情况，只有 face_in_person=True 且通过 face_priority 验证的才标记为目标
        for face_idx, face in enumerate(faces):
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            
            # 判断是否为目标人脸
            # ★★★ 核心修复：使用 face_to_person_map 而不是几何位置 ★★★
            # 这样可以正确识别哪个人脸属于目标人体
            is_target_face = False
            
            if target_person_idx >= 0:
                # 使用匹配阶段确定的人脸分配
                if face_idx == target_person_assigned_face_idx:
                    if current_match_info:
                        # 检查人脸相似度是否足够高（避免标记陌生人脸）
                        face_sim = current_match_info.get('face_sim')
                        body_sim = current_match_info.get('body_sim', 0)
                        motion_score = current_match_info.get('motion_score', 0)
                        match_type = current_match_info.get('match_type', '')
                        
                        # 单人场景：只要人脸在目标框内且没有被明确拒绝，就显示绿框
                        # 多人场景：需要更严格的验证
                        is_single_person = len(persons) == 1 and len(faces) == 1
                        
                        if is_single_person:
                            # ★★★ 单人场景运动预测信任 ★★★
                            # 如果 body+motion 匹配成功，说明运动连续，即使转头（人脸相似度低）
                            # 也应该信任这个人脸是目标人脸
                            # 条件：M >= 0.70（运动连续）且 B >= 0.50（身体特征基本匹配）
                            motion_trusted = (motion_score >= 0.70 and body_sim >= 0.50)
                            
                            if motion_trusted:
                                # 运动连续性高，信任当前人脸（支持转头场景）
                                is_target_face = True
                            elif face_sim is None or face_sim >= 0.20:
                                # 人脸相似度足够高
                                is_target_face = True
                        else:
                            # 多人场景：需要更高的相似度（F >= 0.45）或 face 类型匹配
                            if face_sim is not None and face_sim >= 0.45:
                                is_target_face = True
                            elif match_type in ('face', 'face_motion'):
                                is_target_face = True
                    # 注意：不再使用 "只有一个人脸在框内就认为是目标" 的逻辑
                    # 因为在遮挡场景下，遮挡者的人脸可能正好在目标人体框内
                    # 这会导致错误的绿框
            
            # 判断最终是否绘制绿框
            draw_green = False
            draw_reason = ""
            
            if face_idx == target_face_idx and target_person_idx < 0:
                # 仅人脸匹配的目标
                draw_green = True
                draw_reason = "仅人脸匹配"
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(frame, "TARGET(Face)", (fx1, fy1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif is_target_face:
                # 目标人体内的人脸 - 用绿色高亮
                draw_green = True
                draw_reason = "人体内人脸"
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            elif state_machine.state == SystemState.IDLE:
                # 空闲状态显示所有人脸
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)
            else:
                # 跟踪状态显示非目标人脸（淡色）
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (128, 128, 128), 1)
            
            # 人脸框绘制日志 (仅 DEBUG_VERBOSE 时)
            if DEBUG_VERBOSE and frame_count % 30 == 0:
                if draw_green:
                    print(f"       Face[{face_idx}]: ✓ 绿框 ({draw_reason})")
                else:
                    print(f"       Face[{face_idx}]: 非目标 (target_face={target_face_idx}, target_person={target_person_idx})")
        
        # ★★★ 绘制移动预测框 ★★★
        # 显示 motion 预测的位置（虚线框），帮助调试运动连续性判断
        if state_machine.state == SystemState.TRACKING and mv_recognizer.target is not None:
            last_bbox = mv_recognizer.target.last_bbox
            if last_bbox is not None:
                lx1, ly1, lx2, ly2 = last_bbox.astype(int)
                # 用虚线黄色框表示预测位置
                # OpenCV 没有直接的虚线，用短线段模拟
                color = (0, 255, 255)  # 黄色
                thickness = 1
                dash_length = 10
                gap_length = 5
                
                # 上边
                for x in range(lx1, lx2, dash_length + gap_length):
                    cv2.line(frame, (x, ly1), (min(x + dash_length, lx2), ly1), color, thickness)
                # 下边
                for x in range(lx1, lx2, dash_length + gap_length):
                    cv2.line(frame, (x, ly2), (min(x + dash_length, lx2), ly2), color, thickness)
                # 左边
                for y in range(ly1, ly2, dash_length + gap_length):
                    cv2.line(frame, (lx1, y), (lx1, min(y + dash_length, ly2)), color, thickness)
                # 右边
                for y in range(ly1, ly2, dash_length + gap_length):
                    cv2.line(frame, (lx2, y), (lx2, min(y + dash_length, ly2)), color, thickness)
                
                # 标注 "Pred"
                cv2.putText(frame, "Pred", (lx1, ly1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 绘制画面中心十字线 (云台控制目标)
        if pan_tilt_controller and pan_tilt_active and state_machine.state == SystemState.TRACKING:
            cx, cy = w // 2, h // 2
            cross_size = 30
            cross_color = (0, 255, 255) if pan_tilt_state and pan_tilt_state.mode != "HOLDING" else (0, 255, 0)
            cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), cross_color, 2)
            cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), cross_color, 2)
            cv2.circle(frame, (cx, cy), 5, cross_color, -1)
            
            # 绘制目标到中心的连线
            if target_person_idx >= 0:
                target_person = persons[target_person_idx]
                tx = int((target_person.bbox[0] + target_person.bbox[2]) / 2)
                ty = int((target_person.bbox[1] + target_person.bbox[3]) / 2)
                cv2.line(frame, (cx, cy), (tx, ty), (255, 128, 0), 1)
        
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
        
        # 云台信息
        pan_tilt_info = ""
        if pan_tilt_controller and pan_tilt_active:
            if pan_tilt_state:
                pan_tilt_info = f"Gimbal: Pan={pan_tilt_state.pan:+.1f} Tilt={pan_tilt_state.tilt:+.1f} [{pan_tilt_state.mode}]"
            else:
                pan_tilt_info = f"Gimbal: READY"
        elif pan_tilt_controller:
            pan_tilt_info = "Gimbal: DISABLED (press 'p')"
        
        info_lines = [
            f"FPS: {fps:.1f}",
            f"State: {state_machine.state.value}",
            f"Persons: {len(persons)}, Faces: {len(faces)}",
            f"Target: {target_info}",
            match_info,
            f"Gesture: {gesture.gesture_type.value}" + (f" ({hold_progress*100:.0f}%)" if hold_progress > 0 else ""),
            pan_tilt_info
        ]
        
        for i, line in enumerate(info_lines):
            if line:
                # 不同信息用不同颜色
                if "Match:" in line:
                    color = (0, 255, 255)  # 黄色
                elif "Gimbal:" in line:
                    color = (255, 128, 0)  # 橙色
                else:
                    color = (0, 255, 0)    # 绿色
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
        
        # 绘制结束，记录时间
        t_draw += time.time() - t_section_start
        
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
            if pan_tilt_controller and pan_tilt_active:
                pan_tilt_controller.stop_tracking()
            print("[手动清除] 目标已清除")
        elif key == ord('m'):
            mv_config.auto_learn = not mv_config.auto_learn
            print(f"[自动学习] {'开启' if mv_config.auto_learn else '关闭'}")
        elif key == ord('p'):
            # 切换云台控制
            if pan_tilt_controller:
                pan_tilt_active = not pan_tilt_active
                if pan_tilt_active and state_machine.state == SystemState.TRACKING:
                    pan_tilt_controller.start_tracking(
                        pan_tilt_controller.state.pan,
                        pan_tilt_controller.state.tilt
                    )
                elif not pan_tilt_active:
                    pan_tilt_controller.stop_tracking()
                print(f"[云台控制] {'开启' if pan_tilt_active else '关闭'}")
            else:
                print("[云台控制] 未初始化，无法切换")
    
    # 清理资源
    if pan_tilt_controller and pan_tilt_controller.servo.is_connected():
        pan_tilt_controller.stop_tracking()
        pan_tilt_controller.servo.disconnect()
    
    gesture_detector.release()
    cap.release()
    cv2.destroyAllWindows()
    
    # 关闭并行检测线程池
    _detection_executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
