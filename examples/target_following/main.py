"""
目标跟随系统 - 主程序
Target Following System - Main Program

功能:
1. 手势识别控制启动/停止
2. 人脸+人体检测与识别
3. 目标跟随与状态管理

控制说明:
- 张开手掌: 启动跟随 (锁定当前目标)
- 握拳: 停止跟随
- 按 'q': 退出程序
- 按 'r': 重置系统
"""

import sys
import os
import time
import cv2
import numpy as np

from config import (
    AppConfig, SystemState, GestureType,
    CameraConfig, GestureConfig, FaceDetectorConfig,
    FaceRecognizerConfig, PersonDetectorConfig,
    TrackerConfig, VisualizerConfig
)
from core.camera import CameraCapture
from core.state_machine import StateMachine
from detectors.gesture_detector import GestureDetector, GestureResult
from detectors.face_detector import FaceDetector
from detectors.face_recognizer import FaceRecognizer
from detectors.person_detector import PersonDetector
from trackers.target_tracker import TargetTracker
from utils.visualizer import Visualizer


class TargetFollowingSystem:
    """目标跟随系统"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        
        # 组件
        self.camera = CameraCapture(self.config.camera)
        self.gesture_detector = GestureDetector(self.config.gesture)
        self.face_detector = FaceDetector(self.config.face_detector)
        self.face_recognizer = FaceRecognizer(self.config.face_recognizer)
        self.person_detector = PersonDetector(self.config.person_detector)
        self.tracker = TargetTracker(self.config.tracker)
        self.state_machine = StateMachine()
        self.visualizer = Visualizer(self.config.visualizer)
        
        # 状态
        self.is_running = False
    
    def load(self) -> bool:
        """加载所有模型"""
        print("=" * 60)
        print("    目标跟随系统 - 加载模型")
        print("=" * 60)
        print()
        
        # 手势检测 (MediaPipe，不需要额外模型文件)
        print("[1/4] 加载手势检测器...")
        if not self.gesture_detector.load():
            print("[ERROR] 手势检测器加载失败")
            return False
        print("      ✓ 手势检测器就绪 (MediaPipe)")
        
        # 人脸检测
        print("[2/4] 加载人脸检测器...")
        if not self.face_detector.load():
            print("[ERROR] 人脸检测器加载失败")
            return False
        print("      ✓ 人脸检测器就绪 (SCRFD)")
        
        # 人脸识别
        print("[3/4] 加载人脸识别器...")
        if not self.face_recognizer.load():
            print("[WARNING] 人脸识别器加载失败，将使用位置匹配")
            self.config.use_face_recognition = False
        else:
            print("      ✓ 人脸识别器就绪 (ArcFace)")
        
        # 人体检测
        print("[4/4] 加载人体检测器...")
        if not self.person_detector.load():
            print("[WARNING] 人体检测器加载失败，将只使用人脸跟踪")
            self.config.use_person_detection = False
        else:
            print("      ✓ 人体检测器就绪 (YOLOv8-Pose)")
        
        print()
        print("[INFO] 模型加载完成!")
        return True
    
    def run(self):
        """运行主循环"""
        if not self.camera.open():
            print("[ERROR] 摄像头打开失败")
            return
        
        print()
        print("-" * 60)
        print("  系统启动! 控制说明:")
        print("  - 张开手掌: 启动/停止跟随 (Toggle)")
        print("  - 按 'q': 退出")
        print("  - 按 'r': 重置")
        print("-" * 60)
        print()
        
        self.is_running = True
        frame_count = 0
        start_time = time.time()
        fps_update_time = time.time()
        fps = 0.0
        
        # 缓存检测结果 (用于跳帧优化)
        cached_face_detections = []
        cached_person_detections = []
        cached_face_features = {}
        cached_gesture_result = GestureResult(GestureType.NONE, 0.0)
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                frame_count += 1
                current_state = self.state_machine.state
                
                # ===== 优化策略: 根据状态决定检测内容 =====
                
                # 1. 手势检测 (跳帧)
                if frame_count % self.config.gesture_detect_interval == 0:
                    cached_gesture_result = self.gesture_detector.detect(frame)
                gesture_result = cached_gesture_result
                
                # 2. 处理手势，更新状态机
                prev_state = current_state
                self.state_machine.process_gesture(gesture_result.gesture_type)
                current_state = self.state_machine.state
                
                # 状态切换时清空缓存
                if prev_state != current_state:
                    if current_state == SystemState.IDLE:
                        # 回到空闲，清空所有缓存
                        cached_face_detections = []
                        cached_person_detections = []
                        cached_face_features = {}
                
                # 3. 按状态决定检测内容
                if current_state == SystemState.IDLE:
                    # IDLE 状态: 只检测人脸 (用于显示)，不做人体检测和特征提取
                    if frame_count % self.config.face_detect_interval == 0:
                        cached_face_detections = self.face_detector.detect(frame)
                    # 确保人体检测缓存清空
                    cached_person_detections = []
                    cached_face_features = {}
                    
                elif current_state in [SystemState.TRACKING, SystemState.LOST_TARGET]:
                    # 跟踪状态: 需要完整检测
                    
                    # 人脸检测
                    is_face_detect_frame = (frame_count % self.config.face_detect_interval == 0)
                    if is_face_detect_frame:
                        cached_face_detections = self.face_detector.detect(frame)
                        # 在检测帧立即提取特征 (确保同步)
                        if self.config.use_face_recognition and cached_face_detections:
                            cached_face_features = {}
                            for i, face in enumerate(cached_face_detections):
                                feature = self.face_recognizer.extract_feature(
                                    frame, face.bbox, face.keypoints
                                )
                                if feature is not None:
                                    cached_face_features[i] = feature
                        else:
                            cached_face_features = {}
                    
                    # 人体检测 (只在需要时 - 人脸追踪失败后才启用)
                    need_person = (
                        self.config.use_person_detection and
                        frame_count % self.config.person_detect_interval == 0 and
                        (len(cached_face_detections) == 0 or current_state == SystemState.LOST_TARGET)
                    )
                    if need_person:
                        cached_person_detections = self.person_detector.detect(frame)
                    elif len(cached_face_detections) > 0:
                        # 有人脸时不需要显示人体检测结果
                        cached_person_detections = []
                
                # ===== 状态处理 =====
                target_bbox = None
                
                # 是否为检测帧 (只在检测帧更新跟踪状态)
                is_detect_frame = (frame_count % self.config.face_detect_interval == 0)
                
                if current_state == SystemState.IDLE:
                    # 空闲状态，等待启动手势
                    pass
                
                elif current_state == SystemState.TRACKING:
                    # 跟踪状态
                    if self.state_machine.target.face_feature is None:
                        # 刚启动，需要锁定目标
                        if cached_face_detections:
                            target_face = cached_face_detections[0]
                            
                            # 立即提取人脸特征
                            target_feature = self.face_recognizer.extract_feature(
                                frame, target_face.bbox, target_face.keypoints
                            ) if self.config.use_face_recognition else None
                            
                            # 立即检测人体
                            if self.config.use_person_detection:
                                cached_person_detections = self.person_detector.detect(frame)
                            
                            # 找到对应的人体
                            target_person = None
                            person_feature = None
                            if cached_person_detections:
                                target_person = self._find_person_for_face(
                                    target_face, cached_person_detections
                                )
                                if target_person:
                                    person_feature = self.person_detector.compute_person_feature(
                                        target_person.keypoints
                                    )
                            
                            # 锁定目标
                            self.state_machine.lock_target(target_feature, person_feature)
                            
                            # 设置跟踪器
                            target_bbox = target_person.bbox if target_person else target_face.bbox
                            self.tracker.set_target(target_bbox, target_feature, person_feature)
                            
                            cached_face_features[0] = target_feature
                            print("[INFO] 目标已锁定!")
                    else:
                        # 已锁定，进行跟踪
                        # 只在检测帧调用 tracker.track()，非检测帧使用缓存位置
                        if is_detect_frame:
                            track_result = self.tracker.track(
                                cached_face_detections, cached_face_features,
                                cached_person_detections, self.face_recognizer
                            )
                            
                            if track_result is not None and track_result.found:
                                target_bbox = track_result.bbox
                                self.state_machine.update_tracking(True)
                                # 调试输出: 显示匹配类型
                                if frame_count % 30 == 0:  # 每30帧输出一次
                                    print(f"[TRACK] 匹配类型: {track_result.match_type}, 置信度: {track_result.confidence:.2f}")
                            else:
                                # 目标丢失 - 只在检测帧判定
                                self.state_machine.update_tracking(False)
                                if self.state_machine.state == SystemState.LOST_TARGET:
                                    target_bbox = self.tracker.target_last_bbox
                                    print("[WARNING] 目标丢失，等待目标重新出现...")
                                else:
                                    # 还在容忍范围内，使用上一次的位置
                                    target_bbox = self.tracker.target_last_bbox
                        else:
                            # 非检测帧: 使用缓存的目标位置
                            target_bbox = self.tracker.target_last_bbox
                
                elif current_state == SystemState.LOST_TARGET:
                    # 丢失状态，继续尝试寻找
                    # 只在检测帧尝试重新匹配，必须使用特征匹配（防止匹配到错误的人脸）
                    if is_detect_frame:
                        track_result = self.tracker.track(
                            cached_face_detections, cached_face_features,
                            cached_person_detections, self.face_recognizer,
                            require_feature_match=True  # 必须特征匹配才能恢复
                        )
                        
                        if track_result is not None and track_result.found:
                            target_bbox = track_result.bbox
                            self.state_machine.update_tracking(True)
                            print(f"[INFO] 目标已恢复! 匹配类型: {track_result.match_type}, 置信度: {track_result.confidence:.2f}")
                        else:
                            target_bbox = self.tracker.target_last_bbox
                    else:
                        # 非检测帧: 使用缓存的目标位置
                        target_bbox = self.tracker.target_last_bbox
                
                # 计算实时 FPS
                loop_time = time.time() - loop_start
                if loop_time > 0:
                    instant_fps = 1.0 / loop_time
                    fps = 0.9 * fps + 0.1 * instant_fps  # 平滑
                
                # 可视化 - 根据状态决定显示内容
                if current_state == SystemState.IDLE:
                    # 空闲状态: 显示所有检测到的人脸
                    output = self.visualizer.draw(
                        frame,
                        state=current_state,
                        gesture_result=gesture_result,
                        face_detections=cached_face_detections,
                        person_detections=[],  # 不显示人体
                        target_bbox=None,
                        is_target_found=False,
                        fps=fps
                    )
                elif current_state == SystemState.TRACKING:
                    # 跟踪状态: 只显示目标
                    output = self.visualizer.draw(
                        frame,
                        state=current_state,
                        gesture_result=gesture_result,
                        face_detections=[],  # 不显示其他人脸
                        person_detections=[],
                        target_bbox=target_bbox,
                        is_target_found=True,
                        fps=fps
                    )
                else:
                    # LOST_TARGET状态: 显示最后位置 + 检测到的非目标人脸（标记为非目标）
                    output = self.visualizer.draw(
                        frame,
                        state=current_state,
                        gesture_result=gesture_result,
                        face_detections=cached_face_detections,  # 显示检测到的人脸(会被标记为非目标)
                        person_detections=[],
                        target_bbox=target_bbox,  # 显示最后已知位置
                        is_target_found=False,
                        fps=fps,
                        show_non_target_hint=True  # 新参数: 显示非目标提示
                    )
                
                cv2.imshow("Target Following System", output)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                elif key == ord('r'):
                    self.reset()
                    cached_face_detections = []
                    cached_person_detections = []
                    cached_face_features = {}
                    print("[INFO] 系统已重置")
        
        finally:
            self.cleanup()
    
    def _find_person_for_face(self, face, persons):
        """找到包含人脸的人体"""
        face_center = np.array([
            (face.bbox[0] + face.bbox[2]) / 2,
            (face.bbox[1] + face.bbox[3]) / 2
        ])
        
        best_person = None
        best_score = -1
        
        for person in persons:
            # 检查人脸中心是否在人体框内
            if (person.bbox[0] <= face_center[0] <= person.bbox[2] and
                person.bbox[1] <= face_center[1] <= person.bbox[3]):
                # 计算人脸占人体的比例作为得分
                person_height = person.bbox[3] - person.bbox[1]
                face_height = face.bbox[3] - face.bbox[1]
                score = face_height / person_height if person_height > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_person = person
        
        return best_person
    
    def reset(self):
        """重置系统"""
        self.state_machine.reset()
        self.tracker.reset()
    
    def cleanup(self):
        """清理资源"""
        self.camera.release()
        self.gesture_detector.release()
        cv2.destroyAllWindows()


def main():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║         目标跟随系统 (Target Following System)          ║")
    print("║       手势控制 + 人脸识别 + 人体检测 + 目标跟踪          ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    # 创建系统
    system = TargetFollowingSystem()
    
    # 加载模型
    if not system.load():
        print("[ERROR] 系统初始化失败")
        return 1
    
    # 运行
    system.run()
    
    print("\n[INFO] 系统已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
