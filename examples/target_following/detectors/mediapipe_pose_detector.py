"""
MediaPipe Pose 人体姿态检测器 - 轻量版
MediaPipe Pose Person Detector - Lightweight

特点:
  - 复用 MediaPipe (已安装)
  - 实时性能: 30+ FPS
  - 33 关键点 (比 COCO 17 更多)
  - 无需额外安装
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import mediapipe as mp


@dataclass
class MediaPipePoseConfig:
    """MediaPipe Pose 配置"""
    model_complexity: int = 0  # 0=Lite, 1=Full, 2=Heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False


@dataclass
class PersonDetection:
    """人体检测结果 (与 YOLOv8 接口兼容)"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    keypoints: Optional[np.ndarray] = None  # [33, 3] (x, y, visibility)


class MediaPipePoseDetector:
    """MediaPipe Pose 人体姿态检测器"""
    
    # MediaPipe Pose 33 关键点名称
    KEYPOINT_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    def __init__(self, config: MediaPipePoseConfig = None):
        self.config = config or MediaPipePoseConfig()
        self.mp_pose = mp.solutions.pose
        self.pose = None
        self._is_loaded = False
    
    def load(self) -> bool:
        """初始化 MediaPipe Pose"""
        try:
            self.pose = self.mp_pose.Pose(
                model_complexity=self.config.model_complexity,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                enable_segmentation=self.config.enable_segmentation,
                static_image_mode=False
            )
            self._is_loaded = True
            
            complexity_name = ["Lite", "Full", "Heavy"][self.config.model_complexity]
            print(f"[INFO] MediaPipe Pose 已加载 (模式: {complexity_name})")
            return True
            
        except Exception as e:
            print(f"[ERROR] MediaPipe Pose 加载失败: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        检测人体姿态
        
        Args:
            frame: BGR 图像
            
        Returns:
            检测结果列表 (MediaPipe Pose 只返回单人)
        """
        if not self._is_loaded:
            return []
        
        h, w = frame.shape[:2]
        
        # MediaPipe 需要 RGB 输入
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 推理
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return []
        
        landmarks = results.pose_landmarks.landmark
        
        # 提取关键点 [33, 3] (x, y, visibility)
        keypoints = np.zeros((33, 3), dtype=np.float32)
        x_coords = []
        y_coords = []
        
        for i, lm in enumerate(landmarks):
            keypoints[i] = [lm.x * w, lm.y * h, lm.visibility]
            if lm.visibility > 0.5:
                x_coords.append(lm.x * w)
                y_coords.append(lm.y * h)
        
        if not x_coords:
            return []
        
        # 计算边界框
        x1 = max(0, min(x_coords) - 20)
        y1 = max(0, min(y_coords) - 20)
        x2 = min(w, max(x_coords) + 20)
        y2 = min(h, max(y_coords) + 20)
        
        # 计算平均置信度
        visible_conf = [lm.visibility for lm in landmarks if lm.visibility > 0.5]
        avg_conf = np.mean(visible_conf) if visible_conf else 0.0
        
        detection = PersonDetection(
            bbox=np.array([x1, y1, x2, y2]),
            confidence=float(avg_conf),
            keypoints=keypoints
        )
        
        return [detection]
    
    def compute_person_feature(self, frame: np.ndarray, detection: PersonDetection) -> Optional[np.ndarray]:
        """
        计算人体特征向量 (基于关键点几何特征)
        
        Args:
            frame: BGR 图像
            detection: 检测结果
            
        Returns:
            特征向量 (归一化)
        """
        if detection.keypoints is None:
            return None
        
        keypoints = detection.keypoints
        
        # 使用关键点构建简单特征
        # 1. 关键点相对位置 (归一化到边界框)
        bbox = detection.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        if w < 1 or h < 1:
            return None
        
        # 归一化关键点坐标
        norm_kpts = keypoints.copy()
        norm_kpts[:, 0] = (norm_kpts[:, 0] - bbox[0]) / w
        norm_kpts[:, 1] = (norm_kpts[:, 1] - bbox[1]) / h
        
        # 2. 计算骨架长度比例 (躯干、手臂、腿)
        # 肩膀距离
        shoulder_dist = np.linalg.norm(keypoints[11, :2] - keypoints[12, :2])
        # 躯干高度 (肩膀到臀部)
        torso_height = (np.linalg.norm(keypoints[11, :2] - keypoints[23, :2]) +
                        np.linalg.norm(keypoints[12, :2] - keypoints[24, :2])) / 2
        
        # 特征向量: 归一化关键点 + 比例特征
        feature = np.concatenate([
            norm_kpts[:, :2].flatten(),  # 33*2 = 66
            [shoulder_dist / (w + 1e-6)],
            [torso_height / (h + 1e-6)]
        ])
        
        # 归一化
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        
        return feature.astype(np.float32)
    
    def draw_pose(self, frame: np.ndarray, detection: PersonDetection,
                  color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        绘制姿态骨架
        
        Args:
            frame: BGR 图像
            detection: 检测结果
            color: 绘制颜色
            
        Returns:
            绘制后的图像
        """
        if detection.keypoints is None:
            return frame
        
        output = frame.copy()
        keypoints = detection.keypoints
        
        # 绘制关键点
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:
                cv2.circle(output, (int(x), int(y)), 3, color, -1)
        
        # 绘制骨架连接
        connections = [
            # 躯干
            (11, 12), (11, 23), (12, 24), (23, 24),
            # 左臂
            (11, 13), (13, 15),
            # 右臂
            (12, 14), (14, 16),
            # 左腿
            (23, 25), (25, 27),
            # 右腿
            (24, 26), (26, 28),
            # 面部
            (0, 7), (0, 8)  # 鼻子到耳朵
        ]
        
        for i, j in connections:
            if keypoints[i, 2] > 0.5 and keypoints[j, 2] > 0.5:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(output, pt1, pt2, color, 2)
        
        return output
    
    def release(self) -> None:
        """释放资源"""
        if self.pose:
            self.pose.close()
            self.pose = None
        self._is_loaded = False


# 便于导入
__all__ = ["MediaPipePoseConfig", "MediaPipePoseDetector", "PersonDetection"]
