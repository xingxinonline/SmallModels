"""
状态机控制器
State Machine Controller
"""

from enum import Enum
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass, field
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SystemState, GestureType


@dataclass
class TargetInfo:
    """目标信息"""
    face_feature: Optional[any] = None      # 人脸特征向量
    person_feature: Optional[any] = None    # 人体特征向量
    last_bbox: Optional[any] = None         # 最后边界框
    lost_frame_count: int = 0               # 丢失帧数
    lock_time: float = 0.0                  # 锁定时间戳


class StateMachine:
    """状态机控制器"""
    
    def __init__(self, lost_timeout_frames: int = 60, gesture_hold_duration: float = 3.0, gesture_cooldown_seconds: float = 3.0):
        self.state = SystemState.IDLE
        self.target = TargetInfo()
        self.lost_timeout_frames = lost_timeout_frames
        
        # 状态变更回调
        self._on_state_change: Optional[Callable] = None
        
        # 手势防抖: 防止同一手势连续触发 (使用时间戳，不受帧率影响)
        self._last_gesture = GestureType.NONE
        self._cooldown_seconds = gesture_cooldown_seconds  # 冷却时间 (秒)
        self._cooldown_end_time = 0.0  # 冷却结束时间戳
        
        # 状态转换规则
        self._transitions: Dict[SystemState, Dict[str, SystemState]] = {
            SystemState.IDLE: {
                "start": SystemState.TRACKING,
            },
            SystemState.TRACKING: {
                "stop": SystemState.IDLE,
                "lost": SystemState.LOST_TARGET,
            },
            SystemState.LOST_TARGET: {
                "found": SystemState.TRACKING,
                "stop": SystemState.IDLE,
                "timeout": SystemState.IDLE,
            }
        }
        
        # 手势持续时间检测
        self._gesture_hold_duration = gesture_hold_duration  # 需要持续多少秒
        self._gesture_start_time = None  # 手势开始时间
        self._current_holding_gesture = GestureType.NONE
    
    def process_gesture(self, gesture: GestureType, current_time: float = None, debug: bool = False) -> bool:
        """
        处理手势输入 (手掌 Toggle 模式 + 持续时间检测)
        
        Args:
            gesture: 检测到的手势
            current_time: 当前时间戳 (秒)
            debug: 是否打印调试日志
            
        Returns:
            是否触发状态变更
        """
        if current_time is None:
            current_time = time.time()
        
        # 检查冷却 (使用时间戳)
        if current_time < self._cooldown_end_time:
            if debug:
                remaining = self._cooldown_end_time - current_time
                print(f"[SM] 冷却中: 剩余 {remaining:.1f}s")
            return False
        
        if gesture == GestureType.NONE:
            # 手势消失，重置持续时间计数
            if debug and self._current_holding_gesture != GestureType.NONE:
                print(f"[SM] 手势消失，重置计时 (之前: {self._current_holding_gesture.value})")
            self._gesture_start_time = None
            self._current_holding_gesture = GestureType.NONE
            self._last_gesture = GestureType.NONE
            return False
        
        # 手掌 Toggle 模式: 张开手掌切换启动/停止
        if gesture != GestureType.OPEN_PALM:
            # 收到其他手势（如握拳），重置计时
            if self._current_holding_gesture == GestureType.OPEN_PALM:
                if debug:
                    print(f"[SM] 手势变为 {gesture.value}，重置计时")
                self._gesture_start_time = None
                self._current_holding_gesture = GestureType.NONE
            return False
        
        # 以下是 gesture == GestureType.OPEN_PALM 的处理
        if gesture == GestureType.OPEN_PALM:
            # 检查是否是新的手势开始
            if self._current_holding_gesture != GestureType.OPEN_PALM:
                self._current_holding_gesture = GestureType.OPEN_PALM
                self._gesture_start_time = current_time
                if debug:
                    print(f"[SM] 检测到新手势 OPEN_PALM，开始计时 t={current_time:.2f}")
                return False
            
            # 检查持续时间是否足够
            if self._gesture_start_time is not None:
                hold_time = current_time - self._gesture_start_time
                
                if debug:
                    print(f"[SM] OPEN_PALM 持续 {hold_time:.2f}s / {self._gesture_hold_duration}s")
                
                if hold_time >= self._gesture_hold_duration:
                    # 持续时间足够，触发动作
                    if debug:
                        print(f"[SM] 持续时间足够! 触发 Toggle")
                    self._gesture_start_time = None
                    self._current_holding_gesture = GestureType.NONE
                    self._cooldown_end_time = current_time + self._cooldown_seconds  # 设置冷却结束时间
                    
                    # Toggle: 空闲->启动, 跟踪->停止
                    # 注意：LOST_TARGET 状态不能直接停止，必须先找回目标或超时
                    if self.state == SystemState.IDLE:
                        return self._trigger("start")
                    elif self.state == SystemState.TRACKING:
                        # 只有在正常跟踪状态才能停止
                        return self._trigger("stop")
        
        elif gesture == GestureType.VICTORY:
            # 暂停/恢复 (可扩展)
            pass
        
        return False
    
    def get_gesture_hold_progress(self) -> float:
        """
        获取当前手势持续进度 (0.0 - 1.0)
        用于UI显示进度条
        """
        if self._gesture_start_time is None or self._current_holding_gesture == GestureType.NONE:
            return 0.0
        
        hold_time = time.time() - self._gesture_start_time
        progress = min(1.0, hold_time / self._gesture_hold_duration)
        return progress
    
    def get_current_holding_gesture(self) -> GestureType:
        """获取当前正在持续的手势"""
        return self._current_holding_gesture
    
    def update_tracking(self, target_found: bool) -> bool:
        """
        更新跟踪状态
        
        Args:
            target_found: 是否找到目标
            
        Returns:
            是否触发状态变更
        """
        if self.state == SystemState.TRACKING:
            if not target_found:
                self.target.lost_frame_count += 1
                # 使用配置的丢失超时帧数
                if self.target.lost_frame_count > self.lost_timeout_frames:
                    return self._trigger("lost")
            else:
                self.target.lost_frame_count = 0
        
        elif self.state == SystemState.LOST_TARGET:
            if target_found:
                self.target.lost_frame_count = 0
                return self._trigger("found")
            # 移除 timeout: 永远等待目标重新出现，只能通过手势停止
        
        return False
    
    def lock_target(
        self, 
        face_feature=None, 
        person_feature=None
    ):
        """锁定目标"""
        self.target.face_feature = face_feature
        self.target.person_feature = person_feature
        self.target.lost_frame_count = 0
        self.target.lock_time = time.time()
    
    def update_position(self, bbox):
        """更新目标位置"""
        if bbox is not None:
            self.target.last_bbox = bbox
    
    def clear_target(self):
        """清除目标"""
        self.target = TargetInfo()
    
    def _trigger(self, action: str) -> bool:
        """触发状态转换"""
        if action not in self._transitions.get(self.state, {}):
            return False
        
        old_state = self.state
        new_state = self._transitions[self.state][action]
        self.state = new_state
        
        print(f"[STATE] {old_state.value} --({action})--> {new_state.value}")
        
        # 状态进入处理
        if new_state == SystemState.IDLE:
            self.clear_target()
        
        # 回调
        if self._on_state_change:
            self._on_state_change(old_state, new_state, action)
        
        return True
    
    def set_state_change_callback(self, callback: Callable):
        """设置状态变更回调"""
        self._on_state_change = callback
    
    @property
    def is_tracking(self) -> bool:
        return self.state in [SystemState.TRACKING, SystemState.LOST_TARGET]
    
    @property
    def has_target(self) -> bool:
        return self.target.face_feature is not None or self.target.person_feature is not None
