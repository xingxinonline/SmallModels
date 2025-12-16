"""
摄像头采集模块
Camera Capture Module
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from config import CameraConfig


class CameraCapture:
    """摄像头采集器"""
    
    def __init__(self, config: CameraConfig):
        """
        初始化摄像头
        
        Args:
            config: 摄像头配置
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
    
    def open(self) -> bool:
        """
        打开摄像头
        
        Returns:
            是否成功打开
        """
        self.cap = cv2.VideoCapture(self.config.device_id)
        
        if not self.cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 {self.config.device_id}")
            return False
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # 验证实际设置
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
        """
        读取一帧图像
        
        Returns:
            (成功标志, BGR图像)
        """
        if not self._is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        
        return True, frame
    
    def release(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False
            print("[INFO] 摄像头已释放")
    
    @property
    def is_opened(self) -> bool:
        """摄像头是否已打开"""
        return self._is_opened
    
    def __enter__(self):
        """上下文管理器入口"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.release()
        return False
