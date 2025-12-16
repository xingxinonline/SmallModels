"""
可视化模块
Visualizer Module
"""

import cv2
import numpy as np
import time
from typing import List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VisualizerConfig, SystemState, GestureType


class Visualizer:
    """可视化器"""
    
    # 状态显示颜色
    STATE_COLORS = {
        SystemState.IDLE: (128, 128, 128),       # 灰色
        SystemState.TRACKING: (0, 255, 0),       # 绿色
        SystemState.LOST_TARGET: (0, 165, 255),  # 橙色
    }
    
    # 手势显示文字
    GESTURE_NAMES = {
        GestureType.NONE: "None",
        GestureType.OPEN_PALM: "Open Palm (Start)",
        GestureType.CLOSED_FIST: "Fist (Stop)",
        GestureType.VICTORY: "Victory",
        GestureType.THUMB_UP: "Thumb Up",
        GestureType.THUMB_DOWN: "Thumb Down",
    }
    
    def __init__(self, config: VisualizerConfig):
        self.config = config
        self._frame_times: List[float] = []
        self._fps = 0.0
    
    def draw(
        self,
        image: np.ndarray,
        state: SystemState,
        gesture_result=None,
        face_detections: List = None,
        person_detections: List = None,
        target_bbox: Optional[np.ndarray] = None,
        is_target_found: bool = False,
        inference_times: dict = None,
        fps: float = None,
        show_non_target_hint: bool = False,
        gesture_progress: float = 0.0,
        selection_mode: str = None,
        recognizer_type: str = None
    ) -> np.ndarray:
        """
        绘制可视化结果
        
        Args:
            image: BGR 图像
            state: 系统状态
            gesture_result: 手势识别结果
            face_detections: 人脸检测结果
            person_detections: 人体检测结果
            target_bbox: 跟踪目标边界框
            is_target_found: 目标是否找到
            inference_times: 各模块推理时间
            fps: 外部计算的 FPS (如果提供则使用)
            show_non_target_hint: 是否在人脸框上显示"非目标"提示
            gesture_progress: 手势持续进度 (0.0 - 1.0)
            selection_mode: 目标选择模式 (用于IDLE状态显示)
            recognizer_type: 人脸识别器类型 (ArcFace/ShuffleFaceNet)
            
        Returns:
            绘制后的图像
        """
        output = image.copy()
        
        # 绘制人体检测框
        if person_detections:
            for det in person_detections:
                self._draw_person(output, det, is_target=False)
        
        # 绘制人脸检测框
        if face_detections:
            for det in face_detections:
                self._draw_face(output, det, show_non_target_hint=show_non_target_hint)
        
        # 绘制目标跟踪框 (高亮)
        if target_bbox is not None:
            self._draw_target(output, target_bbox, is_target_found)
        
        # 绘制手势
        if gesture_result and gesture_result.hand_bbox is not None:
            self._draw_gesture(output, gesture_result)
        
        # 绘制手势进度条
        if gesture_progress > 0:
            self._draw_gesture_progress(output, gesture_progress)
        
        # 更新 FPS (优先使用外部传入的)
        if fps is not None:
            self._fps = fps
        else:
            self._update_fps()
        
        # 绘制信息面板
        self._draw_info_panel(
            output, state, 
            gesture_result.gesture_type if gesture_result else GestureType.NONE,
            inference_times,
            recognizer_type
        )
        
        return output
    
    def _draw_face(self, image: np.ndarray, detection, show_non_target_hint: bool = False):
        """绘制人脸检测框"""
        bbox = detection.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # 如果是非目标提示模式，使用灰色框并显示提示
        if show_non_target_hint:
            # 灰色框
            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (128, 128, 128),  # 灰色
                self.config.box_thickness
            )
            # 显示"非目标"提示
            label = "Not Target"
            font_scale = 0.5
            thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            # 背景框
            cv2.rectangle(
                image, 
                (x1, y1 - label_h - 5), 
                (x1 + label_w + 5, y1),
                (128, 128, 128), 
                -1
            )
            # 文字
            cv2.putText(
                image, label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )
        else:
            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                self.config.face_box_color,
                self.config.box_thickness
            )
        
        # 绘制关键点 (5个人脸关键点)
        if detection.keypoints is not None and len(detection.keypoints) > 0:
            # 关键点颜色: 眼睛蓝色, 鼻子绿色, 嘴角红色
            kp_colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)]
            for i, kp in enumerate(detection.keypoints):
                x, y = int(kp[0]), int(kp[1])
                color = kp_colors[i] if i < len(kp_colors) else (0, 255, 255)
                cv2.circle(image, (x, y), 3, color, -1)  # 增大到3像素
    
    def _draw_person(self, image: np.ndarray, detection, is_target: bool = False):
        """绘制人体检测框"""
        bbox = detection.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        color = self.config.target_box_color if is_target else self.config.person_box_color
        
        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            color,
            self.config.box_thickness
        )
        
        # 绘制姿态关键点
        if detection.keypoints is not None:
            self._draw_skeleton(image, detection.keypoints)
    
    def _draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray):
        """绘制骨架"""
        # COCO 骨架连接
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
        ]
        
        for i, j in skeleton:
            if keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(image, pt1, pt2, (0, 255, 255), 1)
        
        # 绘制关键点
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.3:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    
    def _draw_target(
        self, 
        image: np.ndarray, 
        bbox: np.ndarray, 
        is_found: bool
    ):
        """绘制跟踪目标框"""
        bbox = bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # 颜色: 找到=绿色, 丢失=红色
        color = (0, 255, 0) if is_found else (0, 0, 255)
        thickness = 3
        
        # 绘制角标记
        corner_len = 20
        
        # 左上角
        cv2.line(image, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(image, (x1, y1), (x1, y1 + corner_len), color, thickness)
        
        # 右上角
        cv2.line(image, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(image, (x2, y1), (x2, y1 + corner_len), color, thickness)
        
        # 左下角
        cv2.line(image, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(image, (x1, y2), (x1, y2 - corner_len), color, thickness)
        
        # 右下角
        cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(image, (x2, y2), (x2, y2 - corner_len), color, thickness)
        
        # 中心点
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(image, (cx, cy), 5, color, -1)
        
        # 标签
        label = "TARGET" if is_found else "LOST"
        cv2.putText(
            image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    
    def _draw_gesture(self, image: np.ndarray, gesture_result):
        """绘制手势"""
        bbox = gesture_result.hand_bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # 绘制边框
        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            self.config.gesture_color,
            self.config.box_thickness
        )
        
        # 绘制手部关键点
        if gesture_result.hand_landmarks is not None:
            landmarks = gesture_result.hand_landmarks.landmarks
            for pt in landmarks:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(image, (x, y), 3, self.config.gesture_color, -1)
    
    def _draw_info_panel(
        self,
        image: np.ndarray,
        state: SystemState,
        gesture: GestureType,
        inference_times: dict = None,
        recognizer_type: str = None
    ):
        """绘制信息面板"""
        h, w = image.shape[:2]
        
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (280, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        y_offset = 30
        
        # 系统状态
        state_color = self.STATE_COLORS.get(state, (255, 255, 255))
        cv2.putText(
            image, f"State: {state.value.upper()}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2
        )
        y_offset += 25
        
        # FPS
        cv2.putText(
            image, f"FPS: {self._fps:.1f}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        y_offset += 20
        
        # 识别器类型
        if recognizer_type:
            cv2.putText(
                image, f"Recog: {recognizer_type}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1
            )
            y_offset += 20
        
        # 手势
        gesture_name = self.GESTURE_NAMES.get(gesture, "Unknown")
        gesture_color = (0, 255, 255) if gesture != GestureType.NONE else (128, 128, 128)
        cv2.putText(
            image, f"Gesture: {gesture_name}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 1
        )
        y_offset += 20
        
        # 推理时间
        if inference_times:
            total_time = sum(inference_times.values())
            cv2.putText(
                image, f"Infer: {total_time*1000:.1f}ms",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
        
        # 操作提示 (右下角)
        tips = [
            "Palm 3s: Toggle",
            "M: Mode | F: Recog",
            "Q: Quit"
        ]
        for i, tip in enumerate(tips):
            cv2.putText(
                image, tip,
                (w - 140, h - 60 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
            )
    
    def _draw_gesture_progress(self, image: np.ndarray, progress: float):
        """
        绘制手势持续进度条
        
        Args:
            image: 图像
            progress: 进度 (0.0 - 1.0)
        """
        h, w = image.shape[:2]
        
        # 进度条位置 (画面底部中央)
        bar_width = 200
        bar_height = 20
        bar_x = (w - bar_width) // 2
        bar_y = h - 50
        
        # 背景
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
        
        # 进度
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            # 颜色: 绿色渐变到红色
            color = (0, int(255 * (1 - progress)), int(255 * progress))
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # 文字
        text = f"Hold: {progress*100:.0f}%"
        cv2.putText(image, text, (bar_x + bar_width // 2 - 30, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _update_fps(self):
        """更新 FPS"""
        current_time = time.time()
        self._frame_times.append(current_time)
        
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed
    
    def show(self, image: np.ndarray) -> int:
        """显示图像"""
        cv2.imshow(self.config.window_name, image)
        return cv2.waitKey(1) & 0xFF
    
    def close(self):
        """关闭窗口"""
        cv2.destroyAllWindows()
