"""
场景感知状态机 (Scene-Aware State Machine)
根据目标距离、朝向和环境条件自动切换跟踪策略

状态流转:
  Idle → FaceMode → FusionMode → BodyMode → BackMode → SearchMode
       ↑                                                  ↓
       └──────────────────────────────────────────────────┘

设计理念:
  1. 每个状态有明确的进入/退出/保持条件
  2. 状态切换使用面积比例+置信度双条件 (适配不同摄像头)
  3. 加入运动一致性和步态周期特征
  4. 控制层加入加速度限制避免过冲
"""

from enum import Enum, auto
from typing import Optional, Callable, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SystemState, GestureType


class SceneState(Enum):
    """场景状态枚举 (更细粒度)"""
    IDLE = "idle"                      # 空闲状态
    FACE_MODE = "face_mode"            # 人脸模式 (近距、正面)
    FUSION_MODE = "fusion_mode"        # 融合模式 (人脸+人体)
    BODY_MODE = "body_mode"            # 人体模式 (远距)
    BACK_MODE = "back_mode"            # 背对模式 (纯人体)
    SEARCH_MODE = "search_mode"        # 搜索模式 (目标丢失)


@dataclass
class SceneSwitchConfig:
    """场景切换条件配置
    
    使用面积比例 + 置信度双条件，适配不同分辨率/焦距摄像头
    """
    # ===== 面积比例阈值 (bbox面积 / 画面面积) =====
    # 大于此值 → 切换到人脸模式 (近距)
    face_mode_area_ratio: float = 0.02        # 2% 画面面积
    # 小于此值 → 切换到人体模式 (远距)
    body_mode_area_ratio: float = 0.005       # 0.5% 画面面积
    
    # ===== 置信度阈值 =====
    # 人脸检测置信度高于此值 → 可进入人脸模式
    face_confidence_threshold: float = 0.7
    # 人体检测置信度高于此值 → 可进入人体模式
    body_confidence_threshold: float = 0.5
    # 人脸识别相似度高于此值 → 确认是目标
    face_similarity_threshold: float = 0.55
    # 人体相似度高于此值 → 确认是目标  
    body_similarity_threshold: float = 0.60
    
    # ===== 运动一致性阈值 =====
    motion_consistency_weight: float = 0.15
    max_velocity_deviation: float = 100.0     # 最大速度偏差 (像素/帧)
    max_direction_deviation: float = 45.0     # 最大方向偏差 (度)
    
    # ===== 状态保持帧数 (防抖) =====
    min_frames_in_state: int = 5              # 最少在当前状态保持帧数
    
    # ===== 搜索模式 =====
    lost_to_search_frames: int = 15           # 丢失多少帧进入搜索
    search_timeout_seconds: float = 10.0      # 搜索超时秒数
    

@dataclass
class MotionState:
    """运动状态 (用于一致性判断)"""
    position: np.ndarray = None               # 当前位置 (cx, cy)
    velocity: np.ndarray = None               # 速度 (vx, vy)
    direction: float = 0.0                    # 运动方向 (弧度)
    
    # 历史轨迹
    history_positions: List[np.ndarray] = field(default_factory=list)
    history_timestamps: List[float] = field(default_factory=list)
    max_history: int = 30
    
    # 步态周期 (简化版)
    gait_phase: float = 0.0                   # 当前步态相位
    gait_frequency: float = 0.0               # 步态频率
    
    def update(self, bbox: np.ndarray, timestamp: float):
        """更新运动状态"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        new_pos = np.array([cx, cy])
        
        if self.position is not None:
            dt = timestamp - self.history_timestamps[-1] if self.history_timestamps else 1/30
            dt = max(dt, 1e-6)
            
            self.velocity = (new_pos - self.position) / dt
            if np.linalg.norm(self.velocity) > 1e-3:
                self.direction = np.arctan2(self.velocity[1], self.velocity[0])
        
        self.position = new_pos
        
        # 更新历史
        self.history_positions.append(new_pos.copy())
        self.history_timestamps.append(timestamp)
        
        if len(self.history_positions) > self.max_history:
            self.history_positions.pop(0)
            self.history_timestamps.pop(0)
        
        # 简化步态估计 (通过 y 坐标周期性变化)
        self._estimate_gait()
    
    def _estimate_gait(self):
        """估计步态周期 (简化版: 通过 y 坐标的周期性波动)"""
        if len(self.history_positions) < 10:
            return
        
        # 提取 y 坐标序列
        y_coords = np.array([p[1] for p in self.history_positions[-20:]])
        
        # 去趋势
        y_detrend = y_coords - np.linspace(y_coords[0], y_coords[-1], len(y_coords))
        
        # 简单的过零点检测估计频率
        zero_crossings = np.where(np.diff(np.sign(y_detrend)))[0]
        if len(zero_crossings) >= 2:
            # 估计周期
            periods = np.diff(zero_crossings)
            if len(periods) > 0:
                avg_period = np.mean(periods) * 2  # 半周期 -> 全周期
                dt = (self.history_timestamps[-1] - self.history_timestamps[-len(y_coords)]) / len(y_coords)
                if dt > 0:
                    self.gait_frequency = 1.0 / (avg_period * dt + 1e-6)
    
    def predict_position(self, dt: float = 1/30) -> np.ndarray:
        """预测未来位置"""
        if self.position is None:
            return None
        if self.velocity is None:
            return self.position.copy()
        return self.position + self.velocity * dt
    
    def compute_consistency(self, candidate_bbox: np.ndarray) -> float:
        """计算候选框与运动状态的一致性 [0, 1]"""
        if self.position is None or self.velocity is None:
            return 1.0  # 无历史信息，不做限制
        
        # 预测位置
        predicted = self.predict_position()
        
        # 候选位置
        cx = (candidate_bbox[0] + candidate_bbox[2]) / 2
        cy = (candidate_bbox[1] + candidate_bbox[3]) / 2
        candidate_pos = np.array([cx, cy])
        
        # 位置偏差
        dist = np.linalg.norm(candidate_pos - predicted)
        
        # 速度大小
        speed = np.linalg.norm(self.velocity)
        
        # 允许的偏差随速度增加
        allowed_deviation = max(50, speed * 0.5)
        
        # 一致性分数
        consistency = max(0, 1 - dist / (allowed_deviation + 1e-6))
        
        return float(consistency)


@dataclass
class TargetState:
    """目标状态"""
    # 特征
    face_feature: Optional[Any] = None
    body_feature: Optional[Any] = None
    
    # 位置
    last_bbox: Optional[np.ndarray] = None
    last_face_bbox: Optional[np.ndarray] = None
    last_body_bbox: Optional[np.ndarray] = None
    
    # 运动
    motion: MotionState = field(default_factory=MotionState)
    
    # 统计
    lost_frame_count: int = 0
    frames_in_current_state: int = 0
    lock_time: float = 0.0
    
    # 置信度
    face_confidence: float = 0.0
    body_confidence: float = 0.0
    match_confidence: float = 0.0


class SceneStateMachine:
    """场景感知状态机
    
    功能:
    1. 根据目标距离和朝向自动切换跟踪模式
    2. 使用面积比例+置信度双条件适配不同摄像头
    3. 集成运动一致性和步态特征
    """
    
    def __init__(
        self,
        switch_config: SceneSwitchConfig = None,
        gesture_hold_duration: float = 3.0,
        gesture_cooldown_seconds: float = 3.0
    ):
        self.config = switch_config or SceneSwitchConfig()
        
        # 状态
        self.scene_state = SceneState.IDLE
        self.target = TargetState()
        
        # 手势相关
        self._gesture_hold_duration = gesture_hold_duration
        self._cooldown_seconds = gesture_cooldown_seconds
        self._cooldown_end_time = 0.0
        self._gesture_start_time = None
        self._current_holding_gesture = GestureType.NONE
        
        # 回调
        self._on_state_change: Optional[Callable] = None
        
        # 场景切换表
        self._scene_transitions: Dict[SceneState, Dict[str, SceneState]] = {
            SceneState.IDLE: {
                "start": SceneState.FACE_MODE,      # 默认从人脸模式开始
                "start_body": SceneState.BODY_MODE, # 如果没检测到人脸
            },
            SceneState.FACE_MODE: {
                "distance_far": SceneState.FUSION_MODE,
                "face_lost": SceneState.BODY_MODE,
                "stop": SceneState.IDLE,
            },
            SceneState.FUSION_MODE: {
                "distance_near": SceneState.FACE_MODE,
                "distance_far": SceneState.BODY_MODE,
                "face_lost": SceneState.BODY_MODE,
                "stop": SceneState.IDLE,
            },
            SceneState.BODY_MODE: {
                "distance_near": SceneState.FUSION_MODE,
                "face_found": SceneState.FUSION_MODE,
                "turn_back": SceneState.BACK_MODE,
                "target_lost": SceneState.SEARCH_MODE,
                "stop": SceneState.IDLE,
            },
            SceneState.BACK_MODE: {
                "turn_front": SceneState.BODY_MODE,
                "face_found": SceneState.FUSION_MODE,
                "target_lost": SceneState.SEARCH_MODE,
                "stop": SceneState.IDLE,
            },
            SceneState.SEARCH_MODE: {
                "target_found": SceneState.BODY_MODE,
                "target_found_face": SceneState.FACE_MODE,
                "timeout": SceneState.IDLE,
                "stop": SceneState.IDLE,
            },
        }
        
        # 搜索模式计时
        self._search_start_time = 0.0
        
        # 画面尺寸 (用于计算面积比例)
        self._frame_area = 640 * 480  # 默认值
    
    def set_frame_size(self, width: int, height: int):
        """设置画面尺寸"""
        self._frame_area = width * height
    
    def compute_area_ratio(self, bbox: np.ndarray) -> float:
        """计算边界框面积占画面比例"""
        if bbox is None:
            return 0.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return (w * h) / (self._frame_area + 1e-6)
    
    def evaluate_scene_conditions(
        self,
        face_bbox: Optional[np.ndarray],
        face_confidence: float,
        body_bbox: Optional[np.ndarray],
        body_confidence: float,
        face_similarity: float = 0.0,
        body_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """评估场景切换条件
        
        Returns:
            包含各种条件评估结果的字典
        """
        conditions = {
            "has_face": face_bbox is not None,
            "has_body": body_bbox is not None,
            "face_area_ratio": self.compute_area_ratio(face_bbox) if face_bbox is not None else 0.0,
            "body_area_ratio": self.compute_area_ratio(body_bbox) if body_bbox is not None else 0.0,
            "face_confidence": face_confidence,
            "body_confidence": body_confidence,
            "face_similarity": face_similarity,
            "body_similarity": body_similarity,
        }
        
        # 双条件判断
        conditions["can_face_mode"] = (
            conditions["has_face"] and
            conditions["face_area_ratio"] >= self.config.face_mode_area_ratio and
            conditions["face_confidence"] >= self.config.face_confidence_threshold
        )
        
        conditions["should_body_mode"] = (
            conditions["has_body"] and
            (not conditions["has_face"] or 
             conditions["face_area_ratio"] < self.config.body_mode_area_ratio)
        )
        
        conditions["is_back_view"] = (
            conditions["has_body"] and
            not conditions["has_face"] and
            conditions["body_confidence"] >= self.config.body_confidence_threshold and
            conditions["body_similarity"] >= self.config.body_similarity_threshold
        )
        
        # 运动一致性
        if body_bbox is not None:
            conditions["motion_consistency"] = self.target.motion.compute_consistency(body_bbox)
        else:
            conditions["motion_consistency"] = 1.0
        
        return conditions
    
    def update_scene(
        self,
        face_bbox: Optional[np.ndarray],
        face_confidence: float,
        body_bbox: Optional[np.ndarray],
        body_confidence: float,
        face_similarity: float = 0.0,
        body_similarity: float = 0.0,
        timestamp: float = None
    ) -> Tuple[bool, str]:
        """更新场景状态
        
        Returns:
            (是否切换状态, 切换原因)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 更新运动状态
        primary_bbox = face_bbox if face_bbox is not None else body_bbox
        if primary_bbox is not None:
            self.target.motion.update(primary_bbox, timestamp)
            self.target.last_bbox = primary_bbox
            self.target.lost_frame_count = 0
        else:
            self.target.lost_frame_count += 1
        
        # 更新置信度
        self.target.face_confidence = face_confidence
        self.target.body_confidence = body_confidence
        
        # 更新状态保持帧数
        self.target.frames_in_current_state += 1
        
        # 评估条件
        conditions = self.evaluate_scene_conditions(
            face_bbox, face_confidence,
            body_bbox, body_confidence,
            face_similarity, body_similarity
        )
        
        # 根据当前状态决定转换
        return self._process_scene_transition(conditions, timestamp)
    
    def _process_scene_transition(
        self,
        conditions: Dict[str, Any],
        timestamp: float
    ) -> Tuple[bool, str]:
        """处理场景状态转换"""
        
        # 防抖: 至少保持 N 帧
        if self.target.frames_in_current_state < self.config.min_frames_in_state:
            return False, "debounce"
        
        # 根据当前状态处理
        if self.scene_state == SceneState.IDLE:
            return False, "idle"  # IDLE 只能通过手势启动
        
        elif self.scene_state == SceneState.FACE_MODE:
            if not conditions["has_face"]:
                return self._trigger_scene("face_lost")
            if conditions["face_area_ratio"] < self.config.body_mode_area_ratio:
                return self._trigger_scene("distance_far")
        
        elif self.scene_state == SceneState.FUSION_MODE:
            if conditions["can_face_mode"]:
                return self._trigger_scene("distance_near")
            if conditions["should_body_mode"]:
                return self._trigger_scene("distance_far")
            if not conditions["has_face"]:
                return self._trigger_scene("face_lost")
        
        elif self.scene_state == SceneState.BODY_MODE:
            if conditions["can_face_mode"]:
                return self._trigger_scene("face_found")
            if conditions["face_area_ratio"] > 0 and conditions["face_area_ratio"] >= self.config.face_mode_area_ratio:
                return self._trigger_scene("distance_near")
            if conditions["is_back_view"]:
                return self._trigger_scene("turn_back")
            if self.target.lost_frame_count > self.config.lost_to_search_frames:
                return self._trigger_scene("target_lost")
        
        elif self.scene_state == SceneState.BACK_MODE:
            if conditions["has_face"] and conditions["face_similarity"] > self.config.face_similarity_threshold:
                return self._trigger_scene("face_found")
            if conditions["has_face"] and not conditions["is_back_view"]:
                return self._trigger_scene("turn_front")
            if self.target.lost_frame_count > self.config.lost_to_search_frames:
                return self._trigger_scene("target_lost")
        
        elif self.scene_state == SceneState.SEARCH_MODE:
            # 检查超时
            if timestamp - self._search_start_time > self.config.search_timeout_seconds:
                return self._trigger_scene("timeout")
            # 检查是否重新找到目标
            if conditions["has_face"] and conditions["face_similarity"] > self.config.face_similarity_threshold:
                return self._trigger_scene("target_found_face")
            if conditions["has_body"] and conditions["body_similarity"] > self.config.body_similarity_threshold:
                return self._trigger_scene("target_found")
        
        return False, "no_change"
    
    def _trigger_scene(self, action: str) -> Tuple[bool, str]:
        """触发场景状态转换"""
        if action not in self._scene_transitions.get(self.scene_state, {}):
            return False, f"invalid_action:{action}"
        
        old_state = self.scene_state
        new_state = self._scene_transitions[self.scene_state][action]
        self.scene_state = new_state
        
        # 重置状态帧计数
        self.target.frames_in_current_state = 0
        
        # 特殊处理
        if new_state == SceneState.SEARCH_MODE:
            self._search_start_time = time.time()
        if new_state == SceneState.IDLE:
            self.clear_target()
        
        print(f"[SCENE] {old_state.value} --({action})--> {new_state.value}")
        
        # 回调
        if self._on_state_change:
            self._on_state_change(old_state, new_state, action)
        
        return True, action
    
    def process_gesture(
        self,
        gesture: GestureType,
        has_face: bool = False,
        current_time: float = None,
        debug: bool = False
    ) -> bool:
        """处理手势输入"""
        if current_time is None:
            current_time = time.time()
        
        # 检查冷却
        if current_time < self._cooldown_end_time:
            if debug:
                print(f"[SCENE] 冷却中: 剩余 {self._cooldown_end_time - current_time:.1f}s")
            return False
        
        if gesture == GestureType.NONE:
            self._gesture_start_time = None
            self._current_holding_gesture = GestureType.NONE
            return False
        
        if gesture != GestureType.OPEN_PALM:
            self._gesture_start_time = None
            self._current_holding_gesture = GestureType.NONE
            return False
        
        # OPEN_PALM 处理
        if self._current_holding_gesture != GestureType.OPEN_PALM:
            self._current_holding_gesture = GestureType.OPEN_PALM
            self._gesture_start_time = current_time
            if debug:
                print(f"[SCENE] 检测到 OPEN_PALM，开始计时")
            return False
        
        hold_time = current_time - self._gesture_start_time
        if debug:
            print(f"[SCENE] OPEN_PALM 持续 {hold_time:.2f}s / {self._gesture_hold_duration}s")
        
        if hold_time >= self._gesture_hold_duration:
            # 触发
            self._gesture_start_time = None
            self._current_holding_gesture = GestureType.NONE
            self._cooldown_end_time = current_time + self._cooldown_seconds
            
            if self.scene_state == SceneState.IDLE:
                # 启动跟踪
                if has_face:
                    return self._trigger_scene("start")[0]
                else:
                    return self._trigger_scene("start_body")[0]
            else:
                # 停止跟踪
                return self._trigger_scene("stop")[0]
        
        return False
    
    def get_gesture_hold_progress(self) -> float:
        """获取手势持续进度"""
        if self._gesture_start_time is None:
            return 0.0
        hold_time = time.time() - self._gesture_start_time
        return min(1.0, hold_time / self._gesture_hold_duration)
    
    def get_current_holding_gesture(self) -> GestureType:
        """获取当前持续的手势"""
        return self._current_holding_gesture
    
    def lock_target(self, face_feature=None, body_feature=None):
        """锁定目标"""
        self.target.face_feature = face_feature
        self.target.body_feature = body_feature
        self.target.lost_frame_count = 0
        self.target.lock_time = time.time()
    
    def clear_target(self):
        """清除目标"""
        self.target = TargetState()
    
    def set_state_change_callback(self, callback: Callable):
        """设置状态变更回调"""
        self._on_state_change = callback
    
    @property
    def is_tracking(self) -> bool:
        """是否在跟踪状态"""
        return self.scene_state not in [SceneState.IDLE]
    
    @property
    def has_target(self) -> bool:
        """是否有目标"""
        return self.target.face_feature is not None or self.target.body_feature is not None
    
    # 兼容旧接口
    @property
    def state(self) -> SystemState:
        """映射到旧的 SystemState"""
        if self.scene_state == SceneState.IDLE:
            return SystemState.IDLE
        elif self.scene_state == SceneState.SEARCH_MODE:
            return SystemState.LOST_TARGET
        else:
            return SystemState.TRACKING
    
    def get_tracking_mode(self) -> str:
        """获取当前跟踪模式描述"""
        mode_desc = {
            SceneState.IDLE: "空闲",
            SceneState.FACE_MODE: "人脸模式",
            SceneState.FUSION_MODE: "融合模式",
            SceneState.BODY_MODE: "人体模式",
            SceneState.BACK_MODE: "背对模式",
            SceneState.SEARCH_MODE: "搜索中",
        }
        return mode_desc.get(self.scene_state, "未知")


# 测试代码
if __name__ == "__main__":
    print("=== 场景状态机测试 ===\n")
    
    sm = SceneStateMachine()
    sm.set_frame_size(640, 480)
    
    # 模拟场景
    print("1. 初始状态:", sm.scene_state.value)
    
    # 模拟手势启动
    print("\n2. 模拟手势启动 (持续3秒)...")
    start = time.time()
    while time.time() - start < 3.5:
        result = sm.process_gesture(GestureType.OPEN_PALM, has_face=True, debug=True)
        if result:
            print("   >> 状态变更!")
            break
        time.sleep(0.5)
    
    print("   当前状态:", sm.scene_state.value)
    
    # 模拟场景切换
    print("\n3. 模拟距离变化...")
    
    # 人脸较大 (近距)
    face_bbox = np.array([200, 100, 400, 350])  # 200x250 = 50000 / 307200 = 16%
    sm.update_scene(face_bbox, 0.95, None, 0.0, 0.8, 0.0)
    print(f"   近距人脸: {sm.scene_state.value}")
    
    # 人脸变小 (中距)
    for i in range(6):
        face_bbox = np.array([250, 150, 350, 280])  # 100x130 = 13000 / 307200 = 4%
        sm.update_scene(face_bbox, 0.85, face_bbox, 0.7, 0.75, 0.65)
    print(f"   中距融合: {sm.scene_state.value}")
    
    # 人脸很小 (远距)
    for i in range(6):
        face_bbox = np.array([290, 180, 330, 220])  # 40x40 = 1600 / 307200 = 0.5%
        sm.update_scene(face_bbox, 0.6, None, 0.0, 0.7, 0.0)
    print(f"   远距人体: {sm.scene_state.value}")
    
    print("\n=== 测试完成 ===")
