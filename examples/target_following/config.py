"""
目标跟随系统配置参数
Target Following System Configuration
"""

from dataclasses import dataclass, field
from typing import Tuple, List
from enum import Enum
import os

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# detection_recognition 模型目录
DET_RECOG_DIR = os.path.join(os.path.dirname(BASE_DIR), "detection_recognition")
DET_RECOG_MODELS_DIR = os.path.join(DET_RECOG_DIR, "model")


class SystemState(Enum):
    """系统状态枚举"""
    IDLE = "idle"                  # 空闲状态
    TRACKING = "tracking"          # 跟随中
    LOST_TARGET = "lost_target"    # 目标丢失


class GestureType(Enum):
    """手势类型枚举"""
    NONE = "none"                  # 无手势
    OPEN_PALM = "open_palm"        # 张开手掌 - 启动
    CLOSED_FIST = "closed_fist"    # 握拳 - 停止
    VICTORY = "victory"            # 剪刀手 - 暂停/恢复
    THUMB_UP = "thumb_up"          # 竖起大拇指
    THUMB_DOWN = "thumb_down"      # 大拇指向下


class TargetSelectionMode(Enum):
    """目标选择模式"""
    NEAREST_CENTER = "nearest_center"    # 选择离画面中心最近的
    HIGHEST_CONFIDENCE = "highest_confidence"  # 选择置信度最高的


class FaceDetectorType(Enum):
    """人脸检测器类型"""
    SCRFD = "scrfd"        # SCRFD (ONNX, 更稳定)
    YUNET = "yunet"        # YuNet (PyTorch, 更快)


class PersonDetectorType(Enum):
    """人体检测器类型
    
    推荐选择:
    - YOLOV5_NANO: 多人检测 (3.9MB, ~15ms), 精度好, 推荐
    - MEDIAPIPE: 单人姿态 (复用 MediaPipe), 无需额外安装
    - MOVENET: 超轻量 (2.5MB, ~10ms), 需要 TensorFlow
    - YOLOV8: 高精度姿态 (12.9MB, ~150ms), 慢
    """
    YOLOV5_NANO = "yolov5_nano"  # YOLOv5-Nano (ONNX, 3.9MB, 推荐多人检测)
    MEDIAPIPE = "mediapipe"       # MediaPipe Pose (单人姿态, 已安装)
    MOVENET = "movenet"           # MoveNet Lightning (TFLite, 2.5MB)
    YOLOV8 = "yolov8"             # YOLOv8n-Pose (ONNX, 12.9MB, 慢)


class FaceRecognizerType(Enum):
    """人脸识别器类型
    
    推荐选择:
    - 边缘设备/实时应用: MOBILEFACENET (4MB, 快速)
    - 高精度需求: ARCFACE (166MB, 最准确)
    
    注: MagFace 仅有 iResNet50/100 版本 (~50-100MB)，
        不适合边缘部署，故使用 MobileFaceNet 替代。
    """
    MOBILEFACENET = "mobilefacenet"    # MobileFaceNet (ONNX, 4MB, 推荐边缘部署)
    ARCFACE = "arcface"                # ArcFace (ONNX, 166MB, 高精度)
    SHUFFLEFACENET = "shufflefacenet"  # ShuffleFaceNet (PyTorch, 已废弃)

@dataclass
class CameraConfig:
    """摄像头配置"""
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class GestureConfig:
    """手势识别配置"""
    # MediaPipe 配置 (优化性能)
    max_num_hands: int = 1          # 只检测1只手
    min_detection_confidence: float = 0.6  # 降低检测阈值 [0.7→0.6]
    min_tracking_confidence: float = 0.4   # 降低跟踪阈值 [0.5→0.4]
    model_complexity: int = 0       # 模型复杂度 [0=Lite最快, 1=Full]
    
    # 手势确认帧数 (防止误触发) - 旧参数，保留兼容
    gesture_confirm_frames: int = 5
    
    # 手势持续时间触发 (秒) - 需要持续检测到手势这么久才触发
    gesture_hold_duration: float = 3.0  # 默认3秒


@dataclass
class FaceDetectorConfig:
    """人脸检测配置 (SCRFD - 保留兼容)"""
    model_path: str = os.path.join(MODELS_DIR, "scrfd_500m_bnkps.onnx")
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4


@dataclass
class YuNetDetectorConfig:
    """YuNet 人脸检测配置"""
    model_path: str = os.path.join(DET_RECOG_MODELS_DIR, "yunet_final.pth")
    confidence_threshold: float = 0.7
    nms_threshold: float = 0.3
    top_k: int = 5000
    keep_top_k: int = 10


@dataclass
class FaceRecognizerConfig:
    """人脸识别配置 (ArcFace - 保留兼容)"""
    model_path: str = os.path.join(MODELS_DIR, "w600k_r50.onnx")
    input_size: Tuple[int, int] = (112, 112)
    # 特征相似度阈值
    # 0.4-0.5: 严格匹配（推荐）
    similarity_threshold: float = 0.45


@dataclass
class ShuffleFaceNetConfig:
    """ShuffleFaceNet 人脸识别配置"""
    model_path: str = os.path.join(DET_RECOG_MODELS_DIR, "300.pth")
    vector_path: str = os.path.join(DET_RECOG_MODELS_DIR, "vector.npy")
    similarity_threshold: float = 0.5
    input_size: Tuple[int, int] = (112, 112)


@dataclass
class MobileFaceNetConfig:
    """MobileFaceNet 人脸识别配置 (轻量级)"""
    model_path: str = os.path.join(MODELS_DIR, "mobilefacenet.onnx")
    input_size: Tuple[int, int] = (112, 112)
    # 特征相似度阈值
    # 0.4-0.5: 严格匹配（推荐，减少误识别）
    # 0.3-0.4: 宽松匹配（可能误识别）
    similarity_threshold: float = 0.45


@dataclass
class PersonDetectorConfig:
    """人体检测配置 (YOLOv8)"""
    model_path: str = os.path.join(MODELS_DIR, "yolov8n-pose.onnx")
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    # 只检测 person 类别 (COCO class 0)
    target_class: int = 0


@dataclass 
class MediaPipePoseConfig:
    """MediaPipe Pose 配置 (轻量级人体检测)
    
    使用 MediaPipe Pose:
    - 无需额外安装 (复用已有 MediaPipe)
    - model_complexity: 0=Lite, 1=Full, 2=Heavy
    - Lite 模式: 最快, 30+ FPS
    """
    model_complexity: int = 0  # 0=Lite (推荐), 1=Full, 2=Heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False


@dataclass
class MoveNetConfig:
    """MoveNet 配置 (轻量级人体检测)
    
    MoveNet Lightning:
    - 输入: 192x192 (vs YOLOv8 640x640)
    - 模型: ~2.5 MB (vs 12.9 MB)
    - 速度: ~10ms (vs ~150ms)
    """
    model_path: str = os.path.join(MODELS_DIR, "movenet_lightning_int8.tflite")
    input_size: int = 192
    confidence_threshold: float = 0.3


@dataclass
class YOLOv5PersonConfig:
    """YOLOv5-Nano 人体检测配置 (多人检测)
    
    优点:
    - 支持多人检测 (vs MediaPipe 单人)
    - 精度好, 速度快 (~15ms)
    - 模型轻量 (~3.9MB)
    """
    model_path: str = os.path.join(MODELS_DIR, "yolov5n.onnx")
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    person_class_id: int = 0  # COCO 中 person 类别


@dataclass
class PersonReIDConfig:
    """人体识别 (ReID) 配置
    
    MobileNetV2-ReID:
    - 256D embedding
    - 输入: 256x128 (Market1501 标准)
    - 用于目标锁定和背身恢复
    """
    model_path: str = os.path.join(MODELS_DIR, "mobilenetv2_reid.onnx")
    input_size: Tuple[int, int] = (256, 128)  # (height, width)
    embedding_dim: int = 256
    # 相似度阈值 (余弦相似度)
    # 0.65-0.75: 严格 (推荐，减少误识别)
    # 0.5-0.65: 中等
    # 0.4-0.5: 宽松
    similarity_threshold: float = 0.65
    # 0.4-0.5: 中等
    similarity_threshold: float = 0.5


@dataclass
class TrackerConfig:
    """跟踪器配置"""
    # IoU 匹配阈值 (降低以提高匹配率)
    iou_threshold: float = 0.15
    # 特征匹配权重
    feature_weight: float = 0.6
    iou_weight: float = 0.4
    # 目标丢失超时 (检测帧数，不是实际帧数)
    # 实际丢失时间 ≈ lost_timeout_frames × face_detect_interval / fps
    # 例如: 12 × 8 / 30 ≈ 3.2秒
    lost_timeout_frames: int = 12  # 约 3 秒 (增加间隔后减少帧数)
    # 人体特征匹配阈值 (用于背身恢复目标)
    person_feature_threshold: float = 0.7
    # 追踪优先级: "face_first" = 有人脸优先人脸, "person_first" = 优先人体
    priority: str = "face_first"


@dataclass
class VisualizerConfig:
    """可视化配置"""
    # 颜色配置 (BGR)
    face_box_color: Tuple[int, int, int] = (0, 255, 0)       # 绿色
    person_box_color: Tuple[int, int, int] = (255, 0, 0)     # 蓝色
    target_box_color: Tuple[int, int, int] = (0, 0, 255)     # 红色
    gesture_color: Tuple[int, int, int] = (255, 255, 0)      # 青色
    
    # 线条粗细
    box_thickness: int = 2
    keypoint_radius: int = 3
    
    # 字体
    font_scale: float = 0.6
    
    # 显示选项
    show_fps: bool = True
    show_state: bool = True
    show_gesture: bool = True
    
    window_name: str = "Target Following System"


@dataclass
class AppConfig:
    """应用配置"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    face_detector: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    face_recognizer: FaceRecognizerConfig = field(default_factory=FaceRecognizerConfig)
    person_detector: PersonDetectorConfig = field(default_factory=PersonDetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    
    # 新模型配置
    yunet_detector: YuNetDetectorConfig = field(default_factory=YuNetDetectorConfig)
    shufflefacenet: ShuffleFaceNetConfig = field(default_factory=ShuffleFaceNetConfig)
    mobilefacenet: MobileFaceNetConfig = field(default_factory=MobileFaceNetConfig)
    movenet: MoveNetConfig = field(default_factory=MoveNetConfig)
    mediapipe_pose: MediaPipePoseConfig = field(default_factory=MediaPipePoseConfig)
    yolov5_person: YOLOv5PersonConfig = field(default_factory=YOLOv5PersonConfig)
    person_reid: PersonReIDConfig = field(default_factory=PersonReIDConfig)
    
    # 模型选择 (可切换)
    face_detector_type: FaceDetectorType = FaceDetectorType.SCRFD  # 推荐 SCRFD
    face_recognizer_type: FaceRecognizerType = FaceRecognizerType.MOBILEFACENET  # 默认轻量级
    person_detector_type: PersonDetectorType = PersonDetectorType.YOLOV5_NANO  # 推荐 YOLOv5-Nano (多人)
    
    # 功能开关
    use_person_reid: bool = True  # 是否使用人体 ReID (目标锁定/背身恢复)
    
    # 兼容性: use_new_models 映射到模型类型
    # True = YuNet + ShuffleFaceNet, False = SCRFD + ArcFace
    use_new_models: bool = False  # 默认使用 SCRFD + ArcFace (更准确)
    
    # 功能开关
    use_face_recognition: bool = True
    use_person_detection: bool = True
    
    # 目标选择模式
    target_selection_mode: TargetSelectionMode = TargetSelectionMode.NEAREST_CENTER
    
    # 性能优化: 检测间隔 (跳帧)
    # MediaPipe Pose 快 (~15ms), 可以用较小间隔
    gesture_detect_interval: int = 4   # 手势每4帧检测一次 (~15ms)
    face_detect_interval: int = 8      # 人脸每8帧检测一次 (~30ms)
    person_detect_interval: int = 6    # 人体每6帧检测一次 (MediaPipe: ~15ms)
    feature_extract_interval: int = 8  # 特征提取每8帧一次 (跟人脸检测同步)
    
    # 截图保存目录
    screenshot_dir: str = os.path.join(BASE_DIR, "screenshots")
    
    def __post_init__(self):
        # 确保目录存在
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)


# 默认配置实例
DEFAULT_CONFIG = AppConfig()
