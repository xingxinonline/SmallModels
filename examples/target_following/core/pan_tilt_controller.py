"""
云台舵机控制器 (Pan-Tilt Servo Controller)

参考: https://github.com/xingxinonline/6dof-visual-arm-control

功能:
  1. 串口通信控制总线舵机
  2. PD 控制器保持目标居中
  3. 动态增益调度 (快速追踪 / 精密微调)
  4. 死区处理避免抖动
  
硬件配置 (6DOF 机械臂):
  - ID 6: 底座 (Pan) - 水平旋转
  - ID 3: 大臂 (Tilt) - 俯仰控制
  
协议:
  - 波特率: 115200
  - 帧头: 0x55 0x55
  - 格式: [ID] [Length] [Cmd] [Params...] [Checksum]
"""

import serial
import serial.tools.list_ports
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum


class ServoID(Enum):
    """舵机 ID 定义"""
    CLAW = 1      # 机械爪
    ROLL = 2      # 云台横滚
    TILT = 3      # 大臂 (俯仰)
    JOINT3 = 4    # 中臂
    JOINT2 = 5    # 小臂
    PAN = 6       # 底座 (水平)


@dataclass
class ServoConfig:
    """单个舵机配置"""
    id: int
    name: str
    min_angle: float      # 用户角度最小值
    max_angle: float      # 用户角度最大值
    phys_min: float       # 物理角度最小值 (0-240 映射)
    phys_max: float       # 物理角度最大值
    reset_angle: float    # 复位角度
    
    def user_to_physical(self, user_angle: float) -> float:
        """用户角度转物理角度"""
        user_angle = np.clip(user_angle, self.min_angle, self.max_angle)
        ratio = (user_angle - self.min_angle) / (self.max_angle - self.min_angle)
        return self.phys_min + ratio * (self.phys_max - self.phys_min)


# 默认舵机配置 (来自 gui_control.py)
DEFAULT_SERVO_CONFIGS = {
    ServoID.CLAW: ServoConfig(1, "机械爪", 0, 90, 30, 120, 24),
    ServoID.ROLL: ServoConfig(2, "云台旋转", -90, 90, 30, 210, 0),
    ServoID.TILT: ServoConfig(3, "大臂", -90, 90, 30, 210, -40),
    ServoID.JOINT3: ServoConfig(4, "中臂", -90, 90, 30, 210, -50),
    ServoID.JOINT2: ServoConfig(5, "小臂", 0, 180, 30, 210, 68),
    ServoID.PAN: ServoConfig(6, "底座", -90, 90, 30, 210, 0),
}


@dataclass
class PDConfig:
    """PD 控制器配置 - 优化版 (防过冲)
    
    过冲问题分析:
    1. D项增益不足 → 无法有效阻尼
    2. 精确模式增益仍偏高 → 小误差时仍有明显调整
    3. 缺少速度衰减 → 接近目标时应更慢
    
    解决方案:
    1. 大幅增加 Kd (阻尼)
    2. 降低 precision_mode_boost
    3. 降低 smoothing_alpha (增加平滑性)
    4. 增大死区避免微调抖动
    """
    # 基础增益
    kp: float = 5.0           # 比例增益 [6.0→5.0 降低以减少过冲]
    kd: float = 8.0           # 微分增益 [4.0→8.0 大幅增加阻尼]
    
    # 动态增益调度
    fast_mode_threshold: float = 0.20      # 误差 > 20% 开启快速模式 [0.25→0.20]
    precision_mode_threshold: float = 0.08  # 误差 < 8% 开启精密模式 [0.10→0.08]
    fast_mode_boost: float = 1.5           # 快速模式增益倍数 [1.8→1.5 降低]
    precision_mode_boost: float = 0.4      # 精密模式增益倍数 [0.8→0.4 大幅降低]
    
    # 死区 - 增大以避免微调抖动
    deadzone: float = 0.03    # 归一化误差死区 (3%) [2%→3%]
    angle_deadzone: float = 0.5  # 角度死区 (度) [0.3→0.5]
    
    # 滤波 - 降低以增加平滑性 (alpha越小越平滑)
    smoothing_alpha: float = 0.6  # 低通滤波系数 [0.85→0.6 更平滑]
    
    # 角度限制
    pan_min: float = -90.0
    pan_max: float = 90.0
    tilt_min: float = -90.0
    tilt_max: float = 90.0


@dataclass
class PanTiltState:
    """云台状态"""
    pan: float = 0.0          # 当前水平角度
    tilt: float = 0.0         # 当前俯仰角度
    target_pan: float = 0.0   # 目标水平角度
    target_tilt: float = 0.0  # 目标俯仰角度
    error_x: float = 0.0      # 水平误差
    error_y: float = 0.0      # 垂直误差
    mode: str = "IDLE"        # 模式: IDLE, TRACKING, PRECISION, FAST


class SerialServoController:
    """串口舵机控制器"""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self.lock = threading.Lock()
        
        # 发送节流 - 与帧率同步
        self.last_send_time = {}
        self.min_send_interval = 0.015  # 15ms (~66Hz) - 匹配高帧率
        
        # 调试模式
        self.debug = False
        
        # 舵机配置
        self.servo_configs = DEFAULT_SERVO_CONFIGS.copy()
    
    def list_ports(self) -> List[str]:
        """列出可用串口"""
        return [p.device for p in serial.tools.list_ports.comports()]
    
    def connect(self, port: Optional[str] = None) -> bool:
        """连接串口"""
        if port:
            self.port = port
        
        if not self.port:
            ports = self.list_ports()
            if not ports:
                print("[ERROR] 未找到可用串口")
                return False
            # 优先选择 COM4 或第一个
            self.port = "COM3" if "COM3" in ports else ports[0]
        
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=2)
            print(f"[INFO] 串口已连接: {self.port}")
            return True
        except Exception as e:
            print(f"[ERROR] 串口连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开串口"""
        with self.lock:
            if self.serial and self.serial.is_open:
                self.serial.close()
                print("[INFO] 串口已断开")
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.serial is not None and self.serial.is_open
    
    def _append_checksum(self, command: bytes) -> bytes:
        """添加校验和"""
        checksum = 0
        for b in command:
            checksum += b
        checksum = (~checksum) & 0xFF
        return b'\x55\x55' + command + bytes([checksum])
    
    def move_servo(self, servo_id: int, physical_angle: float, time_ms: int = 100):
        """移动舵机到指定物理角度
        
        Args:
            servo_id: 舵机 ID (1-6)
            physical_angle: 物理角度 (0-240)
            time_ms: 运动时间 (毫秒)
        """
        if not self.is_connected():
            return
        
        # 节流检查
        current_time = time.time()
        if servo_id in self.last_send_time:
            if current_time - self.last_send_time[servo_id] < self.min_send_interval:
                return
        self.last_send_time[servo_id] = current_time
        
        # 尝试获取锁 (非阻塞)
        if not self.lock.acquire(blocking=False):
            return
        
        try:
            if not self.is_connected():
                return
            
            # 构建指令 (移除 reset_input_buffer 以提升性能)
            cmd = bytes([servo_id])
            cmd += b'\x07\x01'  # Length=7, Cmd=1 (MOVE_TIME_WRITE)
            
            # 角度转换: 0-240度 → 0-1000
            position = int((physical_angle / 240.0) * 1000)
            position = max(0, min(1000, position))
            
            cmd += position.to_bytes(2, byteorder="little")
            cmd += time_ms.to_bytes(2, byteorder="little")
            
            cmd = self._append_checksum(cmd)
            self.serial.write(cmd)
            
            if self.debug:
                print(f"[SERVO] ID={servo_id} -> {physical_angle:.1f}° (pos={position}, time={time_ms}ms)")
            
        except Exception as e:
            print(f"[ERROR] 舵机控制异常: {e}")
        finally:
            self.lock.release()
    
    def move_servo_angle(self, servo_id: ServoID, user_angle: float, time_ms: int = 100):
        """移动舵机到用户角度"""
        config = self.servo_configs.get(servo_id)
        if config:
            physical_angle = config.user_to_physical(user_angle)
            self.move_servo(config.id, physical_angle, time_ms)
    
    def servo_on(self, servo_id: int):
        """启用舵机扭矩"""
        if not self.is_connected():
            return
        
        with self.lock:
            try:
                cmd = bytes([servo_id]) + b'\x04\x1F\x01'
                self.serial.write(self._append_checksum(cmd))
            except Exception as e:
                print(f"[ERROR] 舵机启用失败: {e}")
    
    def servo_off(self, servo_id: int):
        """关闭舵机扭矩"""
        if not self.is_connected():
            return
        
        with self.lock:
            try:
                cmd = bytes([servo_id]) + b'\x04\x1F\x00'
                self.serial.write(self._append_checksum(cmd))
            except Exception as e:
                print(f"[ERROR] 舵机关闭失败: {e}")
    
    def enable_all(self):
        """启用所有舵机"""
        for i in range(1, 7):
            self.servo_on(i)
            time.sleep(0.05)
    
    def disable_all(self):
        """关闭所有舵机"""
        for i in range(1, 7):
            self.servo_off(i)
            time.sleep(0.05)


class PanTiltController:
    """云台 PD 控制器
    
    功能:
    1. 计算目标与画面中心的误差
    2. PD 控制器输出角度增量
    3. 动态增益调度 (快速/精密模式)
    4. 低通滤波平滑输出
    """
    
    def __init__(
        self,
        config: Optional[PDConfig] = None,
        frame_size: Tuple[int, int] = (640, 480),
        servo_controller: Optional[SerialServoController] = None
    ):
        self.config = config or PDConfig()
        self.frame_width, self.frame_height = frame_size
        
        # 舵机控制器
        self.servo = servo_controller or SerialServoController()
        
        # 状态
        self.state = PanTiltState()
        
        # PD 控制状态
        self._last_error_x = 0.0
        self._last_error_y = 0.0
        
        # 平滑输出
        self._smooth_pan = 0.0
        self._smooth_tilt = 0.0
        
        # 上次发送角度 (用于角度死区判断)
        self._last_sent_pan = None
        self._last_sent_tilt = None
        
        # 追踪状态
        self._tracking_active = False
        self._lost_counter = 0
        self._max_lost_frames = 10
    
    def set_frame_size(self, width: int, height: int):
        """设置画面尺寸"""
        self.frame_width = width
        self.frame_height = height
    
    def reset(self):
        """重置控制器"""
        self.state = PanTiltState()
        self._last_error_x = 0.0
        self._last_error_y = 0.0
        self._smooth_pan = self.state.pan
        self._smooth_tilt = self.state.tilt
        self._last_sent_pan = None
        self._last_sent_tilt = None
        self._lost_counter = 0
    
    def start_tracking(self, initial_pan: float = 0.0, initial_tilt: float = -40.0):
        """启动追踪
        
        Args:
            initial_pan: 初始水平角度
            initial_tilt: 初始俯仰角度
        """
        self._tracking_active = True
        self.state.pan = initial_pan
        self.state.tilt = initial_tilt
        self._smooth_pan = initial_pan
        self._smooth_tilt = initial_tilt
        self._lost_counter = 0
        
        # 移动到初始位置
        if self.servo.is_connected():
            self.servo.move_servo_angle(ServoID.PAN, initial_pan, 500)
            self.servo.move_servo_angle(ServoID.TILT, initial_tilt, 500)
        
        print(f"[INFO] 云台追踪已启动: Pan={initial_pan}°, Tilt={initial_tilt}°")
    
    def stop_tracking(self):
        """停止追踪"""
        self._tracking_active = False
        self.state.mode = "IDLE"
        print("[INFO] 云台追踪已停止")
    
    def update(
        self,
        target_bbox: Optional[np.ndarray] = None,
        target_center: Optional[Tuple[float, float]] = None,
        speed: float = 1.0,
        loop_dt: float = 0.040
    ) -> PanTiltState:
        """更新控制器
        
        Args:
            target_bbox: 目标边界框 [x1, y1, x2, y2]
            target_center: 目标中心点 (x, y)，优先使用
            speed: 速度系数 (0.1 - 2.0)
            loop_dt: 循环耗时 (秒)，用于自适应舵机时间
            
        Returns:
            PanTiltState: 当前状态
        """
        self._loop_dt = loop_dt  # 保存以便 _send_servo_commands 使用
        if not self._tracking_active:
            self.state.mode = "IDLE"
            return self.state
        
        # 计算目标中心
        if target_center is not None:
            target_cx, target_cy = target_center
        elif target_bbox is not None:
            target_cx = (target_bbox[0] + target_bbox[2]) / 2
            target_cy = (target_bbox[1] + target_bbox[3]) / 2
        else:
            # 目标丢失
            self._lost_counter += 1
            if self._lost_counter > self._max_lost_frames:
                self.state.mode = "LOST"
            return self.state
        
        # 重置丢失计数
        self._lost_counter = 0
        
        # 画面中心
        center_x = self.frame_width / 2
        center_y = self.frame_height / 2
        
        # 计算归一化误差 [-1, 1]
        # 注意: X 轴镜像 (摄像头画面通常是镜像的)
        error_x = (target_cx - center_x) / center_x
        error_y = (target_cy - center_y) / center_y
        
        self.state.error_x = error_x
        self.state.error_y = error_y
        
        # 死区处理
        if abs(error_x) < self.config.deadzone:
            error_x = 0.0
        if abs(error_y) < self.config.deadzone:
            error_y = 0.0
        
        # 如果都在死区内，保持当前位置
        if error_x == 0.0 and error_y == 0.0:
            self.state.mode = "HOLDING"
            self._last_error_x = 0.0
            self._last_error_y = 0.0
            return self.state
        
        # 动态增益调度 (带平滑过渡)
        error_magnitude = (error_x ** 2 + error_y ** 2) ** 0.5
        
        if error_magnitude > self.config.fast_mode_threshold:
            dynamic_boost = self.config.fast_mode_boost
            self.state.mode = "FAST"
        elif error_magnitude < self.config.precision_mode_threshold:
            # 精密模式：误差越小，增益越低 (渐进衰减)
            # 当误差为0时，boost接近0.2；当误差为threshold时，boost为precision_mode_boost
            decay_ratio = error_magnitude / self.config.precision_mode_threshold
            dynamic_boost = 0.2 + (self.config.precision_mode_boost - 0.2) * decay_ratio
            self.state.mode = "PRECISION"
        else:
            # 中间模式：线性插值
            range_size = self.config.fast_mode_threshold - self.config.precision_mode_threshold
            ratio = (error_magnitude - self.config.precision_mode_threshold) / range_size
            dynamic_boost = self.config.precision_mode_boost + (1.0 - self.config.precision_mode_boost) * ratio
            self.state.mode = "TRACKING"
        
        # PD 控制
        kp = self.config.kp * speed * dynamic_boost
        # D项增益在精密模式下保持较高 (抑制过冲更重要)
        kd_boost = max(dynamic_boost, 0.6)  # D项增益最低为60%
        kd = self.config.kd * speed * kd_boost
        
        # 误差变化率 (微分)
        d_error_x = error_x - self._last_error_x
        d_error_y = error_y - self._last_error_y
        
        self._last_error_x = error_x
        self._last_error_y = error_y
        
        # 计算角度增量
        # Pan: 人在右边 (error_x > 0) → 云台向右转 (角度减小，因为镜像)
        delta_pan = -(error_x * kp + d_error_x * kd)
        # Tilt: 人在下面 (error_y > 0) → 云台向下看 (角度减小)
        delta_tilt = -(error_y * kp + d_error_y * kd)
        
        # 目标角度
        target_pan = self.state.pan + delta_pan
        target_tilt = self.state.tilt + delta_tilt
        
        # 低通滤波平滑
        alpha = self.config.smoothing_alpha
        self._smooth_pan = self._smooth_pan * (1 - alpha) + target_pan * alpha
        self._smooth_tilt = self._smooth_tilt * (1 - alpha) + target_tilt * alpha
        
        # 更新当前角度
        self.state.pan = self._smooth_pan
        self.state.tilt = self._smooth_tilt
        
        # 限制角度范围
        self.state.pan = np.clip(self.state.pan, self.config.pan_min, self.config.pan_max)
        self.state.tilt = np.clip(self.state.tilt, self.config.tilt_min, self.config.tilt_max)
        
        self.state.target_pan = target_pan
        self.state.target_tilt = target_tilt
        
        # 发送舵机指令 (带角度死区和自适应时间)
        self._send_servo_commands(self._loop_dt)
        
        return self.state
    
    def _send_servo_commands(self, loop_dt: float = 0.040):
        """发送舵机指令
        
        Args:
            loop_dt: 当前循环耗时 (秒)，用于自适应运动时间
        """
        if not self.servo.is_connected():
            return
        
        # 检查是否需要发送
        send_pan = (
            self._last_sent_pan is None or
            abs(self.state.pan - self._last_sent_pan) > self.config.angle_deadzone
        )
        send_tilt = (
            self._last_sent_tilt is None or
            abs(self.state.tilt - self._last_sent_tilt) > self.config.angle_deadzone
        )
        
        # 自适应运动时间 (参考 6dof-visual-arm-control)
        # 基础时间 = 循环耗时 * 1000ms，确保舵机运动与视觉处理同步
        base_time = int(loop_dt * 1000)
        base_time = max(20, min(60, base_time))  # 限制在 20-60ms
        
        if send_pan:
            diff_pan = abs(self.state.pan - (self._last_sent_pan or self.state.pan))
            # 运动时间 = 基础时间 + 角度差 * 系数 (角度越大时间越长)
            move_time = int(base_time + diff_pan * 1.0)  # 减小系数以加快响应
            move_time = max(20, min(100, move_time))   # 限制在 20-100ms
            
            if self.servo.debug:
                print(f"[PAN] {self._last_sent_pan or 0:.1f}° -> {self.state.pan:.1f}° (diff={diff_pan:.1f}°, time={move_time}ms)")
            
            self.servo.move_servo_angle(ServoID.PAN, self.state.pan, move_time)
            self._last_sent_pan = self.state.pan
        
        if send_tilt:
            diff_tilt = abs(self.state.tilt - (self._last_sent_tilt or self.state.tilt))
            move_time = int(base_time + diff_tilt * 1.0)
            move_time = max(20, min(100, move_time))
            
            if self.servo.debug:
                print(f"[TILT] {self._last_sent_tilt or 0:.1f}° -> {self.state.tilt:.1f}° (diff={diff_tilt:.1f}°, time={move_time}ms)")
            
            self.servo.move_servo_angle(ServoID.TILT, self.state.tilt, move_time)
            self._last_sent_tilt = self.state.tilt
    
    def get_center_offset(
        self,
        target_bbox: Optional[np.ndarray] = None,
        target_center: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """计算目标与画面中心的偏移量 (像素)
        
        Returns:
            (offset_x, offset_y): 偏移像素，正值表示目标在右/下方
        """
        if target_center is not None:
            target_cx, target_cy = target_center
        elif target_bbox is not None:
            target_cx = (target_bbox[0] + target_bbox[2]) / 2
            target_cy = (target_bbox[1] + target_bbox[3]) / 2
        else:
            return (0.0, 0.0)
        
        center_x = self.frame_width / 2
        center_y = self.frame_height / 2
        
        return (target_cx - center_x, target_cy - center_y)


# ============================================
# 虚拟舵机控制器 (无硬件时模拟)
# ============================================

class VirtualServoController(SerialServoController):
    """虚拟舵机控制器 (用于无硬件时测试)"""
    
    def __init__(self):
        super().__init__()
        self._connected = False
        self._positions = {}  # 记录各舵机位置
    
    def connect(self, port: Optional[str] = None) -> bool:
        self._connected = True
        print("[INFO] 虚拟舵机已连接")
        return True
    
    def disconnect(self):
        self._connected = False
        print("[INFO] 虚拟舵机已断开")
    
    def is_connected(self) -> bool:
        return self._connected
    
    def move_servo(self, servo_id: int, physical_angle: float, time_ms: int = 100):
        if self._connected:
            self._positions[servo_id] = physical_angle
            # print(f"[VIRTUAL] Servo {servo_id} -> {physical_angle:.1f}° ({time_ms}ms)")
    
    def servo_on(self, servo_id: int):
        if self._connected:
            print(f"[VIRTUAL] Servo {servo_id} ON")
    
    def servo_off(self, servo_id: int):
        if self._connected:
            print(f"[VIRTUAL] Servo {servo_id} OFF")


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("    云台控制器测试")
    print("=" * 60)
    
    # 使用虚拟舵机
    virtual_servo = VirtualServoController()
    virtual_servo.connect()
    
    # 创建控制器
    controller = PanTiltController(
        servo_controller=virtual_servo,
        frame_size=(640, 480)
    )
    
    # 启动追踪
    controller.start_tracking(initial_pan=0, initial_tilt=-40)
    
    print("\n--- 模拟目标从右侧移动到中心 ---")
    for i in range(20):
        # 目标从右侧 (500, 300) 移动到中心 (320, 240)
        progress = i / 19
        target_x = 500 - progress * 180
        target_y = 300 - progress * 60
        
        state = controller.update(target_center=(target_x, target_y))
        
        if i % 5 == 0:
            offset_x, offset_y = controller.get_center_offset(target_center=(target_x, target_y))
            print(f"  帧 {i:2d}: 目标=({target_x:.0f}, {target_y:.0f}), "
                  f"偏移=({offset_x:+.0f}, {offset_y:+.0f}), "
                  f"Pan={state.pan:+.1f}°, Tilt={state.tilt:+.1f}°, "
                  f"模式={state.mode}")
    
    print("\n--- 模拟目标在死区内抖动 ---")
    for i in range(10):
        # 在中心附近小幅抖动
        target_x = 320 + np.random.uniform(-10, 10)
        target_y = 240 + np.random.uniform(-10, 10)
        
        state = controller.update(target_center=(target_x, target_y))
        print(f"  帧 {i}: 目标=({target_x:.0f}, {target_y:.0f}), 模式={state.mode}")
    
    print("\n--- 模拟目标突然移动 (快速模式) ---")
    state = controller.update(target_center=(100, 100))  # 左上角
    print(f"  突变: Pan={state.pan:+.1f}°, Tilt={state.tilt:+.1f}°, 模式={state.mode}")
    
    controller.stop_tracking()
    virtual_servo.disconnect()
    
    print("\n=== 测试完成 ===")
