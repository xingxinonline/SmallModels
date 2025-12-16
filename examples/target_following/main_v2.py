"""
目标跟随系统 - 主程序 v2
Target Following System - Main Program v2

功能:
1. 手势识别控制启动/停止 (需持续3秒)
2. 可切换人脸检测器: SCRFD / YuNet
3. 可切换人脸识别器: ArcFace / ShuffleFaceNet
4. YOLOv8-Pose 人体检测
5. 目标跟随与状态管理

控制说明:
- 张开手掌持续3秒: 启动/停止跟随 (Toggle)
- 按 'q': 退出程序
- 按 'r': 重置系统
- 按 'm': 切换目标选择模式 (中心/置信度)
- 按 'f': 切换人脸识别器 (ArcFace/ShuffleFaceNet)
"""

import sys
import os
import time
import cv2
import numpy as np

from config import (
    AppConfig, SystemState, GestureType, TargetSelectionMode,
    FaceDetectorType, FaceRecognizerType, PersonDetectorType,
    CameraConfig, GestureConfig, 
    FaceDetectorConfig, FaceRecognizerConfig,
    YuNetDetectorConfig, ShuffleFaceNetConfig, MoveNetConfig, MediaPipePoseConfig,
    PersonDetectorConfig, TrackerConfig, VisualizerConfig
)
from core.camera import CameraCapture
from core.state_machine import StateMachine
from detectors.gesture_detector import GestureDetector, GestureResult
from trackers.target_tracker import TargetTracker
from utils.visualizer import Visualizer


class TargetFollowingSystem:
    """目标跟随系统 v2"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        
        # 组件
        self.camera = CameraCapture(self.config.camera)
        self.gesture_detector = GestureDetector(self.config.gesture)
        
        # 人脸检测器和识别器 (根据配置选择)
        self.face_detector = None
        self.face_recognizer = None
        self.face_detector_type = self.config.face_detector_type
        self.face_recognizer_type = self.config.face_recognizer_type
        
        # 人体检测器 (根据配置选择)
        self.person_detector = None
        self.person_detector_type = self.config.person_detector_type
        
        self.tracker = TargetTracker(self.config.tracker)
        
        # 状态机 (传入手势持续时间和丢失超时)
        self.state_machine = StateMachine(
            lost_timeout_frames=self.config.tracker.lost_timeout_frames,
            gesture_hold_duration=self.config.gesture.gesture_hold_duration
        )
        
        self.visualizer = Visualizer(self.config.visualizer)
        
        # 状态
        self.is_running = False
        self.target_selection_mode = self.config.target_selection_mode
    
    def _get_recognizer_name(self, recognizer_type: FaceRecognizerType) -> str:
        """获取识别器显示名称"""
        names = {
            FaceRecognizerType.ARCFACE: "ArcFace",
            FaceRecognizerType.MOBILEFACENET: "MobileFaceNet",
            FaceRecognizerType.SHUFFLEFACENET: "ShuffleFaceNet"
        }
        return names.get(recognizer_type, "Unknown")
    
    def _get_person_detector_name(self, detector_type: PersonDetectorType) -> str:
        """获取人体检测器显示名称"""
        names = {
            PersonDetectorType.MEDIAPIPE: "MediaPipe Pose",
            PersonDetectorType.MOVENET: "MoveNet",
            PersonDetectorType.YOLOV8: "YOLOv8-Pose"
        }
        return names.get(detector_type, "Unknown")
    
    def _create_person_detector(self, detector_type: PersonDetectorType):
        """创建人体检测器"""
        if detector_type == PersonDetectorType.MEDIAPIPE:
            from detectors.mediapipe_pose_detector import MediaPipePoseDetector, MediaPipePoseConfig
            config = MediaPipePoseConfig(
                model_complexity=self.config.mediapipe_pose.model_complexity,
                min_detection_confidence=self.config.mediapipe_pose.min_detection_confidence,
                min_tracking_confidence=self.config.mediapipe_pose.min_tracking_confidence
            )
            return MediaPipePoseDetector(config)
        elif detector_type == PersonDetectorType.MOVENET:
            from detectors.movenet_detector import MoveNetDetector, MoveNetConfig
            config = MoveNetConfig(
                model_path=self.config.movenet.model_path,
                input_size=self.config.movenet.input_size,
                confidence_threshold=self.config.movenet.confidence_threshold
            )
            return MoveNetDetector(config)
        else:  # YOLOv8
            from detectors.person_detector import PersonDetector
            return PersonDetector(self.config.person_detector)
    
    def _create_face_detector(self, detector_type: FaceDetectorType):
        """创建人脸检测器"""
        if detector_type == FaceDetectorType.YUNET:
            from detectors.yunet_detector import YuNetDetector, YuNetConfig
            cfg = YuNetConfig(
                model_path=self.config.yunet_detector.model_path,
                confidence_threshold=self.config.yunet_detector.confidence_threshold,
                nms_threshold=self.config.yunet_detector.nms_threshold,
                top_k=self.config.yunet_detector.top_k,
                keep_top_k=self.config.yunet_detector.keep_top_k
            )
            return YuNetDetector(cfg)
        else:  # SCRFD
            from detectors.face_detector import FaceDetector
            return FaceDetector(self.config.face_detector)
    
    def _create_face_recognizer(self, recognizer_type: FaceRecognizerType):
        """创建人脸识别器"""
        if recognizer_type == FaceRecognizerType.SHUFFLEFACENET:
            from detectors.shufflefacenet_recognizer import ShuffleFaceNetRecognizer, ShuffleFaceNetConfig as SFConfig
            cfg = SFConfig(
                model_path=self.config.shufflefacenet.model_path,
                vector_path=self.config.shufflefacenet.vector_path,
                similarity_threshold=self.config.shufflefacenet.similarity_threshold,
                input_size=self.config.shufflefacenet.input_size
            )
            return ShuffleFaceNetRecognizer(cfg)
        elif recognizer_type == FaceRecognizerType.MOBILEFACENET:
            from detectors.mobilefacenet_recognizer import MobileFaceNetRecognizer
            return MobileFaceNetRecognizer(self.config.mobilefacenet)
        else:  # ArcFace (默认)
            from detectors.face_recognizer import FaceRecognizer
            return FaceRecognizer(self.config.face_recognizer)
    
    def load(self) -> bool:
        """加载所有模型"""
        print("=" * 60)
        print("    目标跟随系统 v2 - 加载模型")
        print("=" * 60)
        print()
        
        detector_name = "YuNet" if self.face_detector_type == FaceDetectorType.YUNET else "SCRFD"
        recognizer_name = self._get_recognizer_name(self.face_recognizer_type)
        person_det_name = self._get_person_detector_name(self.person_detector_type)
        print(f"[配置] 人脸检测器: {detector_name}")
        print(f"[配置] 人脸识别器: {recognizer_name}")
        print(f"[配置] 人体检测器: {person_det_name}")
        print()
        
        # 手势检测 (MediaPipe)
        print("[1/4] 加载手势检测器...")
        if not self.gesture_detector.load():
            print("[ERROR] 手势检测器加载失败")
            return False
        print("      ✓ 手势检测器就绪 (MediaPipe)")
        
        # 人脸检测器 (根据配置选择)
        print("[2/4] 加载人脸检测器...")
        self.face_detector = self._create_face_detector(self.face_detector_type)
        if not self.face_detector.load():
            print("[ERROR] 人脸检测器加载失败")
            return False
        detector_name = "YuNet" if self.face_detector_type == FaceDetectorType.YUNET else "SCRFD"
        print(f"      ✓ 人脸检测器就绪 ({detector_name})")
        
        # 人脸识别器 (根据配置选择)
        print("[3/4] 加载人脸识别器...")
        self.face_recognizer = self._create_face_recognizer(self.face_recognizer_type)
        if not self.face_recognizer.load():
            print("[WARNING] 人脸识别器加载失败，将使用位置匹配")
            self.config.use_face_recognition = False
        else:
            recognizer_name = self._get_recognizer_name(self.face_recognizer_type)
            print(f"      ✓ 人脸识别器就绪 ({recognizer_name})")
        
        # 人体检测器 (根据配置选择)
        print("[4/4] 加载人体检测器...")
        self.person_detector = self._create_person_detector(self.person_detector_type)
        if not self.person_detector.load():
            print("[WARNING] 人体检测器加载失败，将只使用人脸跟踪")
            self.config.use_person_detection = False
        else:
            person_det_name = self._get_person_detector_name(self.person_detector_type)
            print(f"      ✓ 人体检测器就绪 ({person_det_name})")
        
        print()
        print("[INFO] 模型加载完成!")
        print(f"[INFO] 手势触发需持续: {self.config.gesture.gesture_hold_duration:.1f} 秒")
        print(f"[INFO] 目标选择模式: {self.target_selection_mode.value}")
        return True
    
    def select_target_face(self, face_detections, frame_shape):
        """
        从多个人脸中选择目标
        
        Args:
            face_detections: 人脸检测结果列表
            frame_shape: 画面尺寸 (height, width)
            
        Returns:
            选中的人脸 (或 None)
        """
        if not face_detections:
            return None
        
        if len(face_detections) == 1:
            return face_detections[0]
        
        frame_h, frame_w = frame_shape[:2]
        frame_center = np.array([frame_w / 2, frame_h / 2])
        
        if self.target_selection_mode == TargetSelectionMode.NEAREST_CENTER:
            # 选择离画面中心最近的人脸
            best_face = None
            best_dist = float('inf')
            
            for face in face_detections:
                face_center = np.array([
                    (face.bbox[0] + face.bbox[2]) / 2,
                    (face.bbox[1] + face.bbox[3]) / 2
                ])
                dist = np.linalg.norm(face_center - frame_center)
                if dist < best_dist:
                    best_dist = dist
                    best_face = face
            
            return best_face
        
        else:  # HIGHEST_CONFIDENCE
            # 选择置信度最高的人脸
            best_face = max(face_detections, key=lambda f: f.confidence)
            return best_face
    
    def run(self):
        """运行主循环"""
        if not self.camera.open():
            print("[ERROR] 摄像头打开失败")
            return
        
        print()
        print("-" * 60)
        print("  系统启动! 控制说明:")
        print(f"  - 张开手掌持续 {self.config.gesture.gesture_hold_duration:.0f}秒: 启动/停止跟随")
        print("  - 按 'q': 退出")
        print("  - 按 'r': 重置")
        print("  - 按 'm': 切换目标选择模式")
        print("  - 按 'f': 切换人脸识别器 (MobileFaceNet/ArcFace)")
        print("-" * 60)
        print()
        
        self.is_running = True
        frame_count = 0
        fps = 0.0
        
        # 缓存检测结果
        cached_face_detections = []
        cached_person_detections = []
        cached_face_features = {}
        cached_gesture_result = GestureResult(GestureType.NONE, 0.0)
        
        try:
            while self.is_running:
                loop_start = time.time()
                current_time = time.time()
                
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                frame_count += 1
                current_state = self.state_machine.state
                
                # ===== 1. 手势检测 =====
                if frame_count % self.config.gesture_detect_interval == 0:
                    cached_gesture_result = self.gesture_detector.detect(frame)
                gesture_result = cached_gesture_result
                
                # 2. 处理手势 (传入当前时间)
                prev_state = current_state
                self.state_machine.process_gesture(gesture_result.gesture_type, current_time)
                current_state = self.state_machine.state
                
                # 状态切换时的处理
                if prev_state != current_state:
                    if current_state == SystemState.IDLE:
                        cached_face_detections = []
                        cached_person_detections = []
                        cached_face_features = {}
                    print(f"[STATE] {prev_state.value} --> {current_state.value}")
                
                # ===== 3. 检测逻辑 =====
                is_face_detect_frame = (frame_count % self.config.face_detect_interval == 0)
                is_person_detect_frame = (frame_count % self.config.person_detect_interval == 0)
                
                if current_state == SystemState.IDLE:
                    # IDLE: 检测人脸和人体用于显示
                    if is_face_detect_frame:
                        cached_face_detections = self.face_detector.detect(frame)
                    if is_person_detect_frame and self.config.use_person_detection:
                        cached_person_detections = self.person_detector.detect(frame)
                    cached_face_features = {}
                    
                elif current_state in [SystemState.TRACKING, SystemState.LOST_TARGET]:
                    # TRACKING/LOST: 同时检测人脸和人体
                    if is_face_detect_frame:
                        cached_face_detections = self.face_detector.detect(frame)
                        # TRACKING 和 LOST_TARGET 都需要人脸特征用于匹配
                        # 但只在有人脸且开启识别时才提取
                        if self.config.use_face_recognition and cached_face_detections:
                            cached_face_features = {}
                            for i, face in enumerate(cached_face_detections):
                                feature = self.face_recognizer.extract_feature(
                                    frame, bbox=face.bbox, keypoints=face.keypoints
                                )
                                if feature is not None:
                                    cached_face_features[i] = feature
                        else:
                            cached_face_features = {}
                    
                    # 人体检测 - 始终检测以支持转身跟踪
                    if is_person_detect_frame and self.config.use_person_detection:
                        cached_person_detections = self.person_detector.detect(frame)
                
                # ===== 4. 状态处理 =====
                target_bbox = None
                is_detect_frame = is_face_detect_frame or is_person_detect_frame
                
                if current_state == SystemState.IDLE:
                    pass
                
                elif current_state == SystemState.TRACKING:
                    if self.state_machine.target.face_feature is None:
                        # 锁定目标
                        if cached_face_detections:
                            # 选择目标人脸
                            target_face = self.select_target_face(cached_face_detections, frame.shape)
                            
                            if target_face:
                                # 提取特征
                                target_feature = self.face_recognizer.extract_feature(
                                    frame, bbox=target_face.bbox, keypoints=target_face.keypoints
                                ) if self.config.use_face_recognition else None
                                
                                # 检测人体
                                if self.config.use_person_detection:
                                    cached_person_detections = self.person_detector.detect(frame)
                                
                                # 找对应人体
                                target_person = self._find_person_for_face(target_face, cached_person_detections)
                                person_feature = None
                                if target_person and target_person.keypoints is not None:
                                    person_feature = self.person_detector.compute_person_feature(frame, target_person)
                                
                                # 锁定
                                self.state_machine.lock_target(target_feature, person_feature)
                                target_bbox = target_person.bbox if target_person else target_face.bbox
                                person_bbox = target_person.bbox if target_person else None
                                self.tracker.set_target(target_bbox, target_feature, person_feature, person_bbox)
                                
                                mode_str = "中心最近" if self.target_selection_mode == TargetSelectionMode.NEAREST_CENTER else "置信度最高"
                                print(f"[INFO] 目标已锁定! (选择模式: {mode_str})")
                    else:
                        # 跟踪
                        if is_detect_frame:
                            track_result = self.tracker.track(
                                cached_face_detections, cached_face_features,
                                cached_person_detections, self.face_recognizer
                            )
                            
                            if track_result and track_result.found:
                                target_bbox = track_result.bbox
                                self.state_machine.update_tracking(True)
                                # 显示匹配信息（每30帧或相似度变化较大时）
                                if frame_count % 30 == 0:
                                    threshold = self.face_recognizer.config.similarity_threshold
                                    print(f"[TRACK] {track_result.match_type}, 相似度: {track_result.confidence:.3f} (阈值: {threshold})")
                            else:
                                self.state_machine.update_tracking(False)
                                if self.state_machine.state == SystemState.LOST_TARGET:
                                    target_bbox = self.tracker.target_last_bbox
                                    print("[WARNING] 目标丢失...")
                                else:
                                    target_bbox = self.tracker.target_last_bbox
                        else:
                            target_bbox = self.tracker.target_last_bbox
                
                elif current_state == SystemState.LOST_TARGET:
                    if is_detect_frame:
                        track_result = self.tracker.track(
                            cached_face_detections, cached_face_features,
                            cached_person_detections, self.face_recognizer,
                            require_feature_match=True
                        )
                        
                        if track_result and track_result.found:
                            target_bbox = track_result.bbox
                            self.state_machine.update_tracking(True)
                            print(f"[INFO] 目标已恢复! {track_result.match_type}")
                        else:
                            target_bbox = self.tracker.target_last_bbox
                    else:
                        target_bbox = self.tracker.target_last_bbox
                
                # ===== 5. 计算 FPS =====
                loop_time = time.time() - loop_start
                if loop_time > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / loop_time)
                
                # ===== 6. 可视化 =====
                # 获取手势持续进度
                gesture_progress = self.state_machine.get_gesture_hold_progress()
                holding_gesture = self.state_machine.get_current_holding_gesture()
                
                # 获取识别器类型名称
                recognizer_name = self._get_recognizer_name(self.face_recognizer_type)
                
                if current_state == SystemState.IDLE:
                    output = self.visualizer.draw(
                        frame, state=current_state, gesture_result=gesture_result,
                        face_detections=cached_face_detections,
                        person_detections=cached_person_detections,
                        target_bbox=None, is_target_found=False, fps=fps,
                        gesture_progress=gesture_progress,
                        selection_mode=self.target_selection_mode.value,
                        recognizer_type=recognizer_name
                    )
                elif current_state == SystemState.TRACKING:
                    output = self.visualizer.draw(
                        frame, state=current_state, gesture_result=gesture_result,
                        face_detections=cached_face_detections,
                        person_detections=cached_person_detections,
                        target_bbox=target_bbox, is_target_found=True, fps=fps,
                        gesture_progress=gesture_progress,
                        recognizer_type=recognizer_name
                    )
                else:  # LOST_TARGET
                    output = self.visualizer.draw(
                        frame, state=current_state, gesture_result=gesture_result,
                        face_detections=cached_face_detections,
                        person_detections=cached_person_detections,
                        target_bbox=target_bbox, is_target_found=False, fps=fps,
                        show_non_target_hint=True,
                        gesture_progress=gesture_progress,
                        recognizer_type=recognizer_name
                    )
                
                cv2.imshow("Target Following System v2", output)
                
                # ===== 7. 键盘控制 =====
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                elif key == ord('r'):
                    self.reset()
                    cached_face_detections = []
                    cached_person_detections = []
                    cached_face_features = {}
                    print("[INFO] 系统已重置")
                elif key == ord('m'):
                    # 切换目标选择模式
                    if self.target_selection_mode == TargetSelectionMode.NEAREST_CENTER:
                        self.target_selection_mode = TargetSelectionMode.HIGHEST_CONFIDENCE
                    else:
                        self.target_selection_mode = TargetSelectionMode.NEAREST_CENTER
                    print(f"[INFO] 目标选择模式: {self.target_selection_mode.value}")
                elif key == ord('f'):
                    # 切换人脸识别器 (只能在 IDLE 状态下切换)
                    # 循环顺序: MobileFaceNet → ArcFace → MobileFaceNet (跳过 ShuffleFaceNet)
                    if self.state_machine.state == SystemState.IDLE:
                        cycle_order = [
                            FaceRecognizerType.MOBILEFACENET,  # 默认: 轻量级，适合边缘
                            FaceRecognizerType.ARCFACE,        # 高精度
                        ]
                        current_idx = cycle_order.index(self.face_recognizer_type) if self.face_recognizer_type in cycle_order else 0
                        next_idx = (current_idx + 1) % len(cycle_order)
                        next_type = cycle_order[next_idx]
                        
                        # 重新加载识别器
                        print(f"[INFO] 切换人脸识别器: {self._get_recognizer_name(self.face_recognizer_type)} → {self._get_recognizer_name(next_type)}...")
                        new_recognizer = self._create_face_recognizer(next_type)
                        if new_recognizer.load():
                            # 释放旧识别器
                            if hasattr(self.face_recognizer, 'release'):
                                self.face_recognizer.release()
                            self.face_recognizer = new_recognizer
                            self.face_recognizer_type = next_type
                            print(f"[INFO] 人脸识别器已切换为: {self._get_recognizer_name(next_type)}")
                        else:
                            print(f"[ERROR] {self._get_recognizer_name(next_type)} 加载失败，保持原识别器")
                    else:
                        print("[WARNING] 只能在 IDLE 状态下切换识别器")
        
        finally:
            self.cleanup()
    
    def _find_person_for_face(self, face, persons):
        """找到包含人脸的人体"""
        if not persons:
            return None
            
        face_center = np.array([
            (face.bbox[0] + face.bbox[2]) / 2,
            (face.bbox[1] + face.bbox[3]) / 2
        ])
        
        best_person = None
        best_score = -1
        
        for person in persons:
            if (person.bbox[0] <= face_center[0] <= person.bbox[2] and
                person.bbox[1] <= face_center[1] <= person.bbox[3]):
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
        self.face_detector.release()
        self.face_recognizer.release()
        cv2.destroyAllWindows()


def main():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║       目标跟随系统 v2 (Target Following System)         ║")
    print("║    SCRFD + MobileFaceNet + YOLOv8-Pose + MediaPipe     ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    system = TargetFollowingSystem()
    
    if not system.load():
        print("[ERROR] 系统初始化失败")
        return 1
    
    system.run()
    
    print("\n[INFO] 系统已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
