"""
摄像头采集模块
Camera Capture Module
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CameraConfig


class CameraCapture:
    """摄像头采集器"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
    
    def open(self) -> bool:
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.config.device_id)
        
        if not self.cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 {self.config.device_id}")
            return False
        
        # 设置参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # 减少缓冲区大小以降低延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[INFO] 摄像头已打开:")
        print(f"       设备 ID: {self.config.device_id}")
        print(f"       分辨率: {actual_width}x{actual_height}")
        print(f"       帧率: {actual_fps:.1f} FPS")
        
        self._is_opened = True
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧图像 (丢弃旧帧以降低延迟)"""
        if not self._is_opened or self.cap is None:
            return False, None
        
        # 丢弃缓冲区中的旧帧，只取最新帧
        # 这可以减少延迟，但会丢失一些帧
        self.cap.grab()  # 丢弃一帧
        ret, frame = self.cap.retrieve()
        if not ret:
            ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        return True, frame
    
    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False
            print("[INFO] 摄像头已释放")
    
    @property
    def is_opened(self) -> bool:
        return self._is_opened
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
