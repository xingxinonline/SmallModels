"""
可视化模块
Visualization Module
"""

import cv2
import numpy as np
import time
from typing import List, Optional
from detector import FaceDetection
from config import VisualizerConfig


class Visualizer:
    """可视化器"""
    
    def __init__(self, config: VisualizerConfig):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config
        
        # FPS 计算
        self._frame_times: List[float] = []
        self._fps = 0.0
    
    def draw(
        self, 
        image: np.ndarray, 
        detections: List[FaceDetection],
        inference_time: Optional[float] = None
    ) -> np.ndarray:
        """
        绘制检测结果
        
        Args:
            image: BGR 图像
            detections: 检测结果列表
            inference_time: 推理时间 (秒)
            
        Returns:
            绘制后的图像
        """
        output = image.copy()
        
        # 绘制每个检测结果
        for det in detections:
            self._draw_detection(output, det)
        
        # 更新 FPS
        self._update_fps()
        
        # 绘制信息面板
        self._draw_info_panel(output, len(detections), inference_time)
        
        return output
    
    def _draw_detection(self, image: np.ndarray, detection: FaceDetection):
        """绘制单个检测结果"""
        bbox = detection.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # 绘制边框
        cv2.rectangle(
            image, 
            (x1, y1), (x2, y2), 
            self.config.box_color, 
            self.config.box_thickness
        )
        
        # 绘制置信度
        if self.config.show_confidence:
            label = f"{detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.font_scale, 
                1
            )
            
            # 背景框
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                self.config.box_color,
                -1
            )
            
            # 文字
            cv2.putText(
                image,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        
        # 绘制关键点
        if detection.keypoints is not None:
            for kp in detection.keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(
                    image, 
                    (x, y), 
                    self.config.keypoint_radius, 
                    self.config.keypoint_color, 
                    -1
                )
    
    def _draw_info_panel(
        self, 
        image: np.ndarray, 
        num_faces: int,
        inference_time: Optional[float]
    ):
        """绘制信息面板"""
        h, w = image.shape[:2]
        
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (200, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # FPS
        if self.config.show_fps:
            fps_text = f"FPS: {self._fps:.1f}"
            cv2.putText(
                image, fps_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        # 人脸数量
        face_text = f"Faces: {num_faces}"
        cv2.putText(
            image, face_text, (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        
        # 推理时间
        if inference_time is not None:
            time_text = f"Infer: {inference_time*1000:.1f}ms"
            cv2.putText(
                image, time_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
    
    def _update_fps(self):
        """更新 FPS 计算"""
        current_time = time.time()
        self._frame_times.append(current_time)
        
        # 只保留最近 30 帧
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        
        # 计算 FPS
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed
    
    def show(self, image: np.ndarray) -> int:
        """
        显示图像
        
        Args:
            image: 要显示的图像
            
        Returns:
            按键值
        """
        cv2.imshow(self.config.window_name, image)
        return cv2.waitKey(1) & 0xFF
    
    def close(self):
        """关闭所有窗口"""
        cv2.destroyAllWindows()
