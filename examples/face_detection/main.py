"""
人脸检测主程序
Face Detection Main Program

功能:
1. 打开摄像头采集视频
2. 运行 SCRFD 人脸检测
3. 实时显示检测结果

操作:
- 按 'q' 退出程序
- 按 's' 保存截图
"""

import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime

from config import AppConfig, DEFAULT_CONFIG
from camera import CameraCapture
from detector import SCRFDDetector
from visualizer import Visualizer


def check_model_exists(config: AppConfig) -> bool:
    """检查模型文件是否存在"""
    if not os.path.exists(config.model.model_path):
        print(f"[ERROR] 模型文件不存在: {config.model.model_path}")
        print("[INFO] 请先运行以下命令下载模型:")
        print("       python download_model.py")
        return False
    return True


def save_screenshot(image: np.ndarray, save_dir: str) -> str:
    """保存截图"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"screenshot_{timestamp}.jpg")
    cv2.imwrite(filepath, image)
    return filepath


def main(config: AppConfig = None):
    """主函数"""
    if config is None:
        config = DEFAULT_CONFIG
    
    print("=" * 60)
    print("    SCRFD 人脸检测演示")
    print("    Face Detection Demo using SCRFD")
    print("=" * 60)
    print()
    
    # 检查模型
    if not check_model_exists(config):
        return 1
    
    # 初始化组件
    camera = CameraCapture(config.camera)
    detector = SCRFDDetector(config.model)
    visualizer = Visualizer(config.visualizer)
    
    # 加载模型
    print("[INFO] 正在加载模型...")
    if not detector.load():
        return 1
    
    # 打开摄像头
    print("[INFO] 正在打开摄像头...")
    if not camera.open():
        return 1
    
    print()
    print("[INFO] 系统已就绪!")
    print("[INFO] 按 'q' 退出, 按 's' 保存截图")
    print("-" * 60)
    
    try:
        frame_count = 0
        while True:
            # 读取帧
            ret, frame = camera.read()
            if not ret:
                print("[WARN] 无法读取帧, 重试中...")
                continue
            
            # 人脸检测
            start_time = time.time()
            detections = detector.detect(frame)
            inference_time = time.time() - start_time
            
            # 可视化
            output = visualizer.draw(frame, detections, inference_time)
            
            # 显示
            key = visualizer.show(output)
            
            # 处理按键
            if key == ord('q'):
                print("\n[INFO] 用户退出")
                break
            elif key == ord('s'):
                filepath = save_screenshot(output, config.screenshot_dir)
                print(f"[INFO] 截图已保存: {filepath}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] 程序被中断")
    
    finally:
        # 清理资源
        camera.release()
        visualizer.close()
    
    print(f"[INFO] 总共处理 {frame_count} 帧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
