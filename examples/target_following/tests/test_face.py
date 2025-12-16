"""
人脸识别单独验证测试
Face Recognition Test (Detection + Feature Extraction)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config import FaceDetectorConfig, FaceRecognizerConfig, CameraConfig
from detectors.face_detector import FaceDetector
from detectors.face_recognizer import FaceRecognizer
from core.camera import CameraCapture


def main():
    print("=" * 60)
    print("    人脸识别测试 (SCRFD + ArcFace)")
    print("=" * 60)
    print()
    
    # 初始化
    camera = CameraCapture(CameraConfig())
    face_detector = FaceDetector(FaceDetectorConfig())
    face_recognizer = FaceRecognizer(FaceRecognizerConfig())
    
    # 加载模型
    if not face_detector.load():
        print("[ERROR] 人脸检测器加载失败")
        return 1
    
    if not face_recognizer.load():
        print("[WARNING] 人脸识别器加载失败，将只进行检测")
        has_recognizer = False
    else:
        has_recognizer = True
    
    if not camera.open():
        print("[ERROR] 摄像头打开失败")
        return 1
    
    print()
    print("[INFO] 测试开始! 按 'q' 退出")
    print("[INFO] 按 's' 保存当前人脸特征作为参考")
    print("-" * 60)
    
    reference_feature = None
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                continue
            
            # 人脸检测
            t0 = time.time()
            detections = face_detector.detect(frame)
            detect_time = time.time() - t0
            
            # 绘制结果
            output = frame.copy()
            
            for i, det in enumerate(detections):
                bbox = det.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # 提取特征
                feature = None
                similarity = 0.0
                
                if has_recognizer:
                    t1 = time.time()
                    feature = face_recognizer.extract_feature(
                        frame, det.bbox, det.keypoints
                    )
                    recog_time = time.time() - t1
                    
                    # 与参考特征比较
                    if feature and reference_feature:
                        is_same, similarity = face_recognizer.is_same_person(
                            reference_feature, feature
                        )
                else:
                    recog_time = 0
                
                # 绘制边框
                color = (0, 255, 0) if similarity > 0.4 else (255, 0, 0)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                
                # 绘制关键点
                if det.keypoints is not None:
                    for kp in det.keypoints:
                        cv2.circle(output, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
                
                # 显示信息
                label = f"Face {i+1}: {det.confidence:.2f}"
                if similarity > 0:
                    label += f" Sim:{similarity:.2f}"
                cv2.putText(
                    output, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
            
            # 状态信息
            cv2.putText(
                output, f"Faces: {len(detections)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            cv2.putText(
                output, f"Detect: {detect_time*1000:.1f}ms",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
            
            ref_status = "Saved" if reference_feature else "None"
            cv2.putText(
                output, f"Ref: {ref_status}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
            
            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                output, f"FPS: {fps:.1f}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
            
            cv2.imshow("Face Recognition Test", output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存第一个人脸特征作为参考
                if detections and has_recognizer:
                    det = detections[0]
                    feature = face_recognizer.extract_feature(
                        frame, det.bbox, det.keypoints
                    )
                    if feature:
                        reference_feature = feature
                        print("[INFO] 参考人脸特征已保存")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    print(f"\n[INFO] 测试完成，共处理 {frame_count} 帧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
