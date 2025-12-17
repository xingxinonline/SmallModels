"""
控制层 - PID 控制器 + 安全限制
Controller Layer with PID Control and Safety Limits

功能:
  1. PID 控制器计算目标位置误差
  2. 加速度限制避免电机过冲
  3. 二阶滤波平滑输出
  4. 死区处理避免抖动
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
import time


@dataclass
class PIDConfig:
    """PID 控制器配置"""
    # PID 增益
    kp: float = 0.5           # 比例增益
    ki: float = 0.02          # 积分增益
    kd: float = 0.1           # 微分增益
    
    # 积分限制 (防止积分饱和)
    integral_max: float = 100.0
    
    # 输出限制 [-output_max, output_max]
    output_max: float = 1.0


@dataclass
class SafetyConfig:
    """安全控制配置"""
    # ===== 加速度限制 =====
    max_acceleration: float = 0.3         # 最大加速度 (单位/帧)
    max_velocity: float = 0.8             # 最大速度 (单位/帧)
    
    # ===== 二阶滤波 (低通) =====
    use_second_order_filter: bool = True
    filter_omega: float = 10.0            # 自然频率 (rad/s)
    filter_zeta: float = 0.7              # 阻尼比 (0.7 = 临界阻尼)
    
    # ===== 死区 =====
    deadzone: float = 0.02                # 误差小于此值不输出
    
    # ===== 平滑 =====
    smoothing_factor: float = 0.3         # 指数平滑因子 (0-1)


@dataclass
class ControlOutput:
    """控制输出"""
    pan: float = 0.0          # 水平控制量 [-1, 1]
    tilt: float = 0.0         # 垂直控制量 [-1, 1]
    raw_pan: float = 0.0      # 原始 PID 输出
    raw_tilt: float = 0.0
    velocity_pan: float = 0.0
    velocity_tilt: float = 0.0
    timestamp: float = 0.0


class PIDController:
    """单轴 PID 控制器"""
    
    def __init__(self, config: PIDConfig = None):
        self.config = config or PIDConfig()
        
        # 状态
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
    
    def reset(self):
        """重置控制器状态"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
    
    def compute(self, error: float, dt: float = None) -> float:
        """计算 PID 输出
        
        Args:
            error: 当前误差 (目标 - 当前)
            dt: 时间间隔 (秒)，None 则使用实际时间
            
        Returns:
            控制输出 [-output_max, output_max]
        """
        current_time = time.time()
        
        if dt is None:
            if self._prev_time is None:
                dt = 1/30  # 默认 30 FPS
            else:
                dt = current_time - self._prev_time
        
        dt = max(dt, 1e-6)  # 防止除零
        
        # P 项
        p_term = self.config.kp * error
        
        # I 项 (带限制)
        self._integral += error * dt
        self._integral = np.clip(
            self._integral,
            -self.config.integral_max,
            self.config.integral_max
        )
        i_term = self.config.ki * self._integral
        
        # D 项
        derivative = (error - self._prev_error) / dt
        d_term = self.config.kd * derivative
        
        # 总输出
        output = p_term + i_term + d_term
        output = np.clip(output, -self.config.output_max, self.config.output_max)
        
        # 更新状态
        self._prev_error = error
        self._prev_time = current_time
        
        return float(output)


class SecondOrderFilter:
    """二阶低通滤波器 (用于平滑控制输出)
    
    传递函数: H(s) = ω² / (s² + 2ζωs + ω²)
    
    使用双线性变换离散化
    """
    
    def __init__(self, omega: float = 10.0, zeta: float = 0.7, dt: float = 1/30):
        """
        Args:
            omega: 自然频率 (rad/s)
            zeta: 阻尼比 (0.7 = 临界阻尼附近，响应快且无超调)
            dt: 采样周期 (秒)
        """
        self.omega = omega
        self.zeta = zeta
        self.dt = dt
        
        # 状态变量
        self._y1 = 0.0  # y[n-1]
        self._y2 = 0.0  # y[n-2]
        self._x1 = 0.0  # x[n-1]
        self._x2 = 0.0  # x[n-2]
        
        # 计算滤波器系数
        self._update_coefficients()
    
    def _update_coefficients(self):
        """更新滤波器系数 (双线性变换)"""
        w = self.omega
        z = self.zeta
        T = self.dt
        
        # 双线性变换: s = 2/T * (z-1)/(z+1)
        # 展开后得到差分方程系数
        
        # 分母多项式: s² + 2ζωs + ω²
        # 分子多项式: ω²
        
        # 预计算
        wT = w * T
        wT2 = wT * wT
        
        # 简化的二阶低通滤波器系数
        alpha = wT2 / (wT2 + 2 * z * wT + 1)
        beta1 = 2 * (wT2 - 1) / (wT2 + 2 * z * wT + 1)
        beta2 = (wT2 - 2 * z * wT + 1) / (wT2 + 2 * z * wT + 1)
        
        # 输入系数
        self._b0 = alpha
        self._b1 = 2 * alpha
        self._b2 = alpha
        
        # 输出反馈系数 (负号已包含)
        self._a1 = beta1
        self._a2 = beta2
    
    def reset(self):
        """重置滤波器状态"""
        self._y1 = 0.0
        self._y2 = 0.0
        self._x1 = 0.0
        self._x2 = 0.0
    
    def filter(self, x: float) -> float:
        """滤波
        
        Args:
            x: 输入值
            
        Returns:
            滤波后的输出值
        """
        # 差分方程: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        y = (self._b0 * x + 
             self._b1 * self._x1 + 
             self._b2 * self._x2 - 
             self._a1 * self._y1 - 
             self._a2 * self._y2)
        
        # 更新状态
        self._x2 = self._x1
        self._x1 = x
        self._y2 = self._y1
        self._y1 = y
        
        return float(y)


class TargetController:
    """目标跟随控制器
    
    功能:
    1. 双轴 PID 控制 (pan/tilt)
    2. 加速度限制
    3. 二阶滤波平滑
    4. 死区处理
    """
    
    def __init__(
        self,
        pid_config: PIDConfig = None,
        safety_config: SafetyConfig = None,
        frame_size: Tuple[int, int] = (640, 480)
    ):
        self.pid_config = pid_config or PIDConfig()
        self.safety_config = safety_config or SafetyConfig()
        
        self._frame_width, self._frame_height = frame_size
        
        # 双轴 PID
        self._pan_pid = PIDController(self.pid_config)
        self._tilt_pid = PIDController(self.pid_config)
        
        # 二阶滤波器
        self._pan_filter = SecondOrderFilter(
            omega=self.safety_config.filter_omega,
            zeta=self.safety_config.filter_zeta
        )
        self._tilt_filter = SecondOrderFilter(
            omega=self.safety_config.filter_omega,
            zeta=self.safety_config.filter_zeta
        )
        
        # 状态
        self._prev_output = ControlOutput()
        self._velocity_pan = 0.0
        self._velocity_tilt = 0.0
    
    def set_frame_size(self, width: int, height: int):
        """设置画面尺寸"""
        self._frame_width = width
        self._frame_height = height
    
    def reset(self):
        """重置控制器"""
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._pan_filter.reset()
        self._tilt_filter.reset()
        self._prev_output = ControlOutput()
        self._velocity_pan = 0.0
        self._velocity_tilt = 0.0
    
    def compute(
        self,
        target_bbox: np.ndarray,
        frame_center: Tuple[float, float] = None
    ) -> ControlOutput:
        """计算控制输出
        
        Args:
            target_bbox: 目标边界框 [x1, y1, x2, y2]
            frame_center: 画面中心，默认为画面中心点
            
        Returns:
            ControlOutput
        """
        current_time = time.time()
        
        if frame_center is None:
            frame_center = (self._frame_width / 2, self._frame_height / 2)
        
        # 计算目标中心
        target_cx = (target_bbox[0] + target_bbox[2]) / 2
        target_cy = (target_bbox[1] + target_bbox[3]) / 2
        
        # 计算归一化误差 [-1, 1]
        error_x = (target_cx - frame_center[0]) / (self._frame_width / 2)
        error_y = (target_cy - frame_center[1]) / (self._frame_height / 2)
        
        # 死区处理
        if abs(error_x) < self.safety_config.deadzone:
            error_x = 0.0
        if abs(error_y) < self.safety_config.deadzone:
            error_y = 0.0
        
        # PID 计算
        raw_pan = self._pan_pid.compute(error_x)
        raw_tilt = self._tilt_pid.compute(error_y)
        
        # 加速度限制
        pan, self._velocity_pan = self._limit_acceleration(
            raw_pan, 
            self._prev_output.pan,
            self._velocity_pan
        )
        tilt, self._velocity_tilt = self._limit_acceleration(
            raw_tilt,
            self._prev_output.tilt,
            self._velocity_tilt
        )
        
        # 二阶滤波
        if self.safety_config.use_second_order_filter:
            pan = self._pan_filter.filter(pan)
            tilt = self._tilt_filter.filter(tilt)
        
        # 限幅
        pan = np.clip(pan, -1.0, 1.0)
        tilt = np.clip(tilt, -1.0, 1.0)
        
        # 构建输出
        output = ControlOutput(
            pan=float(pan),
            tilt=float(tilt),
            raw_pan=float(raw_pan),
            raw_tilt=float(raw_tilt),
            velocity_pan=float(self._velocity_pan),
            velocity_tilt=float(self._velocity_tilt),
            timestamp=current_time
        )
        
        self._prev_output = output
        return output
    
    def _limit_acceleration(
        self,
        target: float,
        current: float,
        velocity: float
    ) -> Tuple[float, float]:
        """限制加速度
        
        Args:
            target: 目标值
            current: 当前值
            velocity: 当前速度
            
        Returns:
            (新值, 新速度)
        """
        # 目标速度
        target_velocity = target - current
        
        # 加速度限制
        acceleration = target_velocity - velocity
        acceleration = np.clip(
            acceleration,
            -self.safety_config.max_acceleration,
            self.safety_config.max_acceleration
        )
        
        # 更新速度
        new_velocity = velocity + acceleration
        new_velocity = np.clip(
            new_velocity,
            -self.safety_config.max_velocity,
            self.safety_config.max_velocity
        )
        
        # 更新位置
        new_value = current + new_velocity
        
        return float(new_value), float(new_velocity)


# 测试代码
if __name__ == "__main__":
    print("=== 控制器测试 ===\n")
    
    controller = TargetController(frame_size=(640, 480))
    
    # 模拟目标从右侧移动到中心
    print("模拟目标从右侧 (500, 240) 移动...")
    
    for i in range(30):
        # 目标逐渐移向中心
        x = 500 - i * 10  # 500 -> 210
        target_bbox = np.array([x - 50, 190, x + 50, 290])  # 100x100 的框
        
        output = controller.compute(target_bbox)
        
        if i % 5 == 0:
            print(f"  帧 {i:2d}: 目标 x={x:3d}, "
                  f"raw_pan={output.raw_pan:+.3f}, "
                  f"pan={output.pan:+.3f}, "
                  f"vel={output.velocity_pan:+.3f}")
    
    print("\n模拟突然变化 (测试加速度限制)...")
    controller.reset()
    
    # 突然的大变化
    target_bbox = np.array([550, 190, 650, 290])  # 右侧
    output = controller.compute(target_bbox)
    print(f"  突变: raw_pan={output.raw_pan:+.3f}, pan={output.pan:+.3f}")
    
    # 连续几帧
    for i in range(10):
        output = controller.compute(target_bbox)
        print(f"  帧 {i+1}: raw_pan={output.raw_pan:+.3f}, "
              f"pan={output.pan:+.3f}, vel={output.velocity_pan:+.3f}")
    
    print("\n=== 测试完成 ===")
