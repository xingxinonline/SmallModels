"""
人脸检测配置参数
Face Detection Configuration
"""

from dataclasses import dataclass
from typing import Tuple
import os

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


@dataclass
class CameraConfig:
    """摄像头配置"""
    device_id: int = 0              # 摄像头设备 ID
    width: int = 640                # 采集宽度
    height: int = 480               # 采集高度
    fps: int = 30                   # 帧率


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型文件路径
    model_path: str = os.path.join(MODELS_DIR, "scrfd_500m_bnkps.onnx")
    
    # 模型输入尺寸
    input_size: Tuple[int, int] = (640, 640)
    
    # 推理设备 ('cpu', 'cuda', 'dml')
    device: str = "cpu"
    
    # 置信度阈值
    confidence_threshold: float = 0.5
    
    # NMS IoU 阈值
    nms_threshold: float = 0.4
    
    # 是否检测关键点
    detect_keypoints: bool = True


@dataclass
class VisualizerConfig:
    """可视化配置"""
    # 边框颜色 (BGR)
    box_color: Tuple[int, int, int] = (0, 255, 0)
    
    # 边框粗细
    box_thickness: int = 2
    
    # 关键点颜色
    keypoint_color: Tuple[int, int, int] = (0, 0, 255)
    
    # 关键点半径
    keypoint_radius: int = 3
    
    # 字体缩放
    font_scale: float = 0.6
    
    # 显示 FPS
    show_fps: bool = True
    
    # 显示置信度
    show_confidence: bool = True
    
    # 窗口名称
    window_name: str = "Face Detection - SCRFD"


@dataclass
class AppConfig:
    """应用配置"""
    camera: CameraConfig = None
    model: ModelConfig = None
    visualizer: VisualizerConfig = None
    
    # 截图保存目录
    screenshot_dir: str = os.path.join(BASE_DIR, "screenshots")
    
    def __post_init__(self):
        if self.camera is None:
            self.camera = CameraConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.visualizer is None:
            self.visualizer = VisualizerConfig()
        
        # 确保截图目录存在
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)


# 默认配置实例
DEFAULT_CONFIG = AppConfig()
