"""
人脸识别模型测试脚本
Face Recognition Model Test

功能:
1. 按 's' 保存当前人脸作为目标
2. 实时显示与目标人脸的相似度
3. 按 '+'/'-' 调整阈值
4. 测试不同人脸是否会被误识别

使用方法:
1. 运行脚本
2. 对准你的脸，按 's' 保存为目标
3. 换别人的脸/用手机照片，观察相似度
4. 调整阈值，找到最佳设置
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from config import FaceDetectorConfig, MobileFaceNetConfig, FaceRecognizerConfig

# 选择识别器
USE_MOBILEFACENET = True  # True: MobileFaceNet, False: ArcFace


def main():
    print("=" * 60)
    print("    人脸识别测试工具")
    print("=" * 60)
    print()
    
    # 加载人脸检测器
    print("[1/2] 加载人脸检测器 (SCRFD)...")
    from detectors.face_detector import FaceDetector
    face_detector = FaceDetector(FaceDetectorConfig())
    if not face_detector.load():
        print("[ERROR] 人脸检测器加载失败")
        return
    print("      ✓ 人脸检测器就绪")
    
    # 加载人脸识别器
    print("[2/2] 加载人脸识别器...")
    if USE_MOBILEFACENET:
        from detectors.mobilefacenet_recognizer import MobileFaceNetRecognizer
        config = MobileFaceNetConfig()
        recognizer = MobileFaceNetRecognizer(config)
        recognizer_name = "MobileFaceNet"
    else:
        from detectors.face_recognizer import FaceRecognizer
        config = FaceRecognizerConfig()
        recognizer = FaceRecognizer(config)
        recognizer_name = "ArcFace"
    
    if not recognizer.load():
        print("[ERROR] 人脸识别器加载失败")
        return
    print(f"      ✓ 人脸识别器就绪 ({recognizer_name})")
    
    # 当前阈值
    threshold = config.similarity_threshold
    print(f"\n[INFO] 初始阈值: {threshold}")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print()
    print("-" * 60)
    print("  控制说明:")
    print("  - 's': 保存当前人脸为目标")
    print("  - 'c': 清除目标")
    print("  - '+': 提高阈值 (+0.05)")
    print("  - '-': 降低阈值 (-0.05)")
    print("  - 'q': 退出")
    print("-" * 60)
    print()
    
    # 目标特征
    target_feature = None
    target_image = None
    
    # FPS 计算
    fps_time = time.time()
    fps_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        
        # 检测人脸
        faces = face_detector.detect(frame)
        
        # 显示信息
        display = frame.copy()
        
        # 绘制人脸和相似度
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            
            # 提取特征 (传入 bbox 和 keypoints)
            feature = recognizer.extract_feature(frame, face.bbox, face.keypoints)
            
            # 计算相似度
            similarity = 0.0
            is_match = False
            if target_feature is not None and feature is not None:
                similarity = recognizer.compute_similarity(target_feature, feature)
                is_match = similarity > threshold
            
            # 颜色: 绿色=匹配, 红色=不匹配, 黄色=无目标
            if target_feature is None:
                color = (0, 255, 255)  # 黄色
                label = f"Face {i+1}"
            elif is_match:
                color = (0, 255, 0)    # 绿色
                label = f"MATCH: {similarity:.3f}"
            else:
                color = (0, 0, 255)    # 红色
                label = f"NO: {similarity:.3f}"
            
            # 绘制边界框
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 显示目标图像（左上角）
        if target_image is not None:
            h, w = target_image.shape[:2]
            display[10:10+h, 10:10+w] = target_image
            cv2.rectangle(display, (10, 10), (10+w, 10+h), (0, 255, 0), 2)
            cv2.putText(display, "TARGET", (10, 10+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示状态信息
        info_y = 30
        cv2.putText(display, f"FPS: {fps}", (display.shape[1] - 100, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, f"Recognizer: {recognizer_name}", (200, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display, f"Threshold: {threshold:.2f}", (200, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        status = "Target: SET" if target_feature is not None else "Target: NONE (press 's')"
        cv2.putText(display, status, (200, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if target_feature else (0, 0, 255), 2)
        
        # 底部说明
        cv2.putText(display, "s:Save  c:Clear  +/-:Threshold  q:Quit", 
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Face Recognition Test", display)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存目标
            if faces:
                face = faces[0]  # 取第一个人脸
                feature = recognizer.extract_feature(frame, face.bbox, face.keypoints)
                if feature is not None:
                    # FaceFeature 是 dataclass，直接赋值即可
                    target_feature = feature
                    # 保存人脸图像
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    target_image = frame[y1:y2, x1:x2].copy()
                    # 缩放到固定大小
                    target_image = cv2.resize(target_image, (100, 100))
                    print(f"[INFO] 目标已保存!")
        elif key == ord('c'):
            # 清除目标
            target_feature = None
            target_image = None
            print("[INFO] 目标已清除")
        elif key == ord('+') or key == ord('='):
            # 提高阈值
            threshold = min(1.0, threshold + 0.05)
            print(f"[INFO] 阈值: {threshold:.2f}")
        elif key == ord('-'):
            # 降低阈值
            threshold = max(0.0, threshold - 0.05)
            print(f"[INFO] 阈值: {threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    face_detector.release()
    recognizer.release()
    
    print()
    print(f"[结果] 推荐阈值: {threshold:.2f}")
    print("       请在 config.py 中更新 similarity_threshold")


if __name__ == "__main__":
    main()
