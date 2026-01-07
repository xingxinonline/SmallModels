"""
手势检测器 - 使用 MediaPipe Hands
Gesture Detector using MediaPipe Hands
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GestureConfig, GestureType


@dataclass
class HandLandmarks:
    """手部关键点"""
    landmarks: np.ndarray  # [21, 3] - 21个关键点的 (x, y, z)
    handedness: str        # "Left" or "Right"
    confidence: float


@dataclass
class GestureResult:
    """手势识别结果"""
    gesture_type: GestureType
    confidence: float
    hand_landmarks: Optional[HandLandmarks] = None
    hand_bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2]


class GestureDetector:
    """手势检测器"""
    
    # MediaPipe 手部关键点索引
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    
    def __init__(self, config: GestureConfig):
        self.config = config
        self.mp_hands = None
        self.hands = None
        self._is_loaded = False
        
        # 手势确认计数器
        self._gesture_counter = {}
        self._last_confirmed_gesture = GestureType.NONE
    
    def load(self) -> bool:
        """加载 MediaPipe Hands 模型"""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            
            # 获取 model_complexity，默认为0 (Lite模型，最快)
            model_complexity = getattr(self.config, 'model_complexity', 0)
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config.max_num_hands,
                model_complexity=model_complexity,  # 0=Lite, 1=Full
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
            self._is_loaded = True
            complexity_name = "Lite" if model_complexity == 0 else "Full"
            print(f"[INFO] 手势检测器已加载 (MediaPipe Hands, {complexity_name}模型)")
            return True
        except Exception as e:
            print(f"[ERROR] 手势检测器加载失败: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> GestureResult:
        """
        检测手势
        
        Args:
            image: BGR 图像
            
        Returns:
            手势识别结果
        """
        if not self._is_loaded:
            return GestureResult(GestureType.NONE, 0.0)
        
        # BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe 推理
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            self._reset_gesture_counter()
            return GestureResult(GestureType.NONE, 0.0)
        
        h, w = image.shape[:2]
        
        # ============================================
        # 选择最佳的手：优先选择做出有效手势的手
        # 而不是固定取第一只检测到的手
        # ============================================
        best_hand_idx = 0
        best_gesture = GestureType.NONE
        best_hand_size = 0
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 提取关键点
            landmarks = np.array([
                [lm.x * w, lm.y * h, lm.z] 
                for lm in hand_landmarks.landmark
            ])
            
            # 计算手的大小（用于优先选择大的手/近的手）
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            hand_size = (x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min())
            
            # 识别这只手的手势
            gesture_type = self._classify_gesture(landmarks)
            
            # 选择策略：
            # 1. 优先选择做出有效手势（张开手掌/握拳）的手
            # 2. 如果多只手都有有效手势，选择更大的手（更近的手）
            # 3. 如果没有有效手势，选择最大的手
            is_valid_gesture = gesture_type in (GestureType.OPEN_PALM, GestureType.CLOSED_FIST, 
                                                 GestureType.VICTORY, GestureType.THUMB_UP)
            best_is_valid = best_gesture in (GestureType.OPEN_PALM, GestureType.CLOSED_FIST,
                                              GestureType.VICTORY, GestureType.THUMB_UP)
            
            should_update = False
            if is_valid_gesture and not best_is_valid:
                # 当前手有有效手势，之前的没有 → 选当前
                should_update = True
            elif is_valid_gesture and best_is_valid:
                # 都有有效手势 → 选更大的
                if hand_size > best_hand_size:
                    should_update = True
            elif not is_valid_gesture and not best_is_valid:
                # 都没有有效手势 → 选更大的
                if hand_size > best_hand_size:
                    should_update = True
            
            if should_update:
                best_hand_idx = hand_idx
                best_gesture = gesture_type
                best_hand_size = hand_size
        
        # 处理选中的手
        hand_landmarks = results.multi_hand_landmarks[best_hand_idx]
        handedness = results.multi_handedness[best_hand_idx].classification[0]
        
        # 提取关键点
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z] 
            for lm in hand_landmarks.landmark
        ])
        
        # 计算边界框
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        bbox = np.array([
            x_coords.min(), y_coords.min(),
            x_coords.max(), y_coords.max()
        ])
        
        # 识别手势
        gesture_type = self._classify_gesture(landmarks)
        
        # 手势确认 (防止误触发)
        confirmed_gesture = self._confirm_gesture(gesture_type)
        
        hand_lm = HandLandmarks(
            landmarks=landmarks,
            handedness=handedness.label,
            confidence=handedness.score
        )
        
        return GestureResult(
            gesture_type=confirmed_gesture,
            confidence=handedness.score,
            hand_landmarks=hand_lm,
            hand_bbox=bbox
        )
    
    def _classify_gesture(self, landmarks: np.ndarray) -> GestureType:
        """
        根据关键点分类手势
        
        使用手指伸展状态判断:
        - 张开手掌: 5根手指都伸展
        - 握拳: 所有手指都弯曲
        - 剪刀手: 只有食指和中指伸展
        - 竖大拇指: 只有大拇指伸展
        """
        # 判断每根手指是否伸展
        fingers_extended = self._get_fingers_state(landmarks)
        
        # 统计伸展的手指数
        num_extended = sum(fingers_extended)
        
        # 分类手势
        if num_extended == 5:
            return GestureType.OPEN_PALM
        elif num_extended == 0:
            return GestureType.CLOSED_FIST
        elif fingers_extended == [False, True, True, False, False]:
            return GestureType.VICTORY
        elif fingers_extended == [True, False, False, False, False]:
            return GestureType.THUMB_UP
        else:
            return GestureType.NONE
    
    def _get_fingers_state(self, landmarks: np.ndarray) -> List[bool]:
        """
        获取每根手指的伸展状态
        
        Returns:
            [拇指, 食指, 中指, 无名指, 小指] 的伸展状态
        """
        fingers = []
        
        # 拇指: 比较 TIP 和 IP 的 x 坐标 (考虑左右手)
        thumb_extended = landmarks[self.THUMB_TIP][0] > landmarks[self.THUMB_IP][0]
        fingers.append(thumb_extended)
        
        # 其他四指: 比较 TIP 和 PIP 的 y 坐标
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        finger_pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        
        for tip, pip in zip(finger_tips, finger_pips):
            # y 坐标越小越高 (图像坐标系)
            extended = landmarks[tip][1] < landmarks[pip][1]
            fingers.append(extended)
        
        return fingers
    
    def _confirm_gesture(self, gesture_type: GestureType) -> GestureType:
        """
        确认手势 (需要连续多帧检测到相同手势)
        """
        if gesture_type == GestureType.NONE:
            self._reset_gesture_counter()
            return GestureType.NONE
        
        # 增加计数
        if gesture_type not in self._gesture_counter:
            self._gesture_counter = {gesture_type: 1}
        else:
            self._gesture_counter[gesture_type] += 1
        
        # 检查是否达到确认阈值
        if self._gesture_counter[gesture_type] >= self.config.gesture_confirm_frames:
            self._last_confirmed_gesture = gesture_type
            return gesture_type
        
        return GestureType.NONE
    
    def _reset_gesture_counter(self):
        """重置手势计数器"""
        self._gesture_counter = {}
    
    def release(self):
        """释放资源"""
        if self.hands:
            self.hands.close()
            self._is_loaded = False
