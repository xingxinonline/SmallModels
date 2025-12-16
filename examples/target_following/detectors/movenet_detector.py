"""
MoveNet 人体姿态检测器 - 超轻量版
MoveNet Person Pose Detector - Ultra Lightweight

模型: MoveNet Lightning (Google)
特点:
  - 输入: 192x192 (vs YOLOv8 的 640x640)
  - 模型大小: ~2.5 MB
  - 速度: 50+ FPS
  - 关键点: 17 COCO keypoints
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# TensorFlow Lite - 优先使用轻量版 tflite-runtime，其次是完整 TensorFlow
tf_lite = None
try:
    # 方案1: 使用 tflite-runtime (推荐, 只有~2MB)
    import tflite_runtime.interpreter as tflite
    tf_lite = tflite
    print("[INFO] 使用 tflite-runtime")
except ImportError:
    try:
        # 方案2: 使用完整 TensorFlow
        import tensorflow as tf
        tf_lite = tf.lite
        print("[INFO] 使用 TensorFlow Lite")
    except ImportError:
        tf_lite = None


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MoveNetConfig:
    """MoveNet 配置"""
    model_path: str = ""
    input_size: int = 192  # Lightning: 192, Thunder: 256
    confidence_threshold: float = 0.3
    
    def __post_init__(self):
        if not self.model_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, "models", "movenet_lightning_int8.tflite")


@dataclass
class PersonDetection:
    """人体检测结果 (与 YOLOv8 接口兼容)"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    keypoints: Optional[np.ndarray] = None  # [17, 3] (x, y, conf)


class MoveNetDetector:
    """MoveNet 人体姿态检测器 (轻量版)"""
    
    # COCO 17 关键点 (与 YOLOv8 相同)
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def __init__(self, config: MoveNetConfig = None):
        self.config = config or MoveNetConfig()
        self.interpreter = None
        self._is_loaded = False
        
        # 输入/输出详情
        self.input_details = None
        self.output_details = None
    
    def load(self) -> bool:
        """加载 TFLite 模型"""
        if tf_lite is None:
            print("[ERROR] TFLite 运行时未安装")
            print("[INFO] 请运行: uv add tflite-runtime")
            return False
        
        if not os.path.exists(self.config.model_path):
            print(f"[ERROR] 模型文件不存在: {self.config.model_path}")
            print("[INFO] 请运行 download_movenet.py 下载模型")
            return False
        
        try:
            # 加载 TFLite 模型
            self.interpreter = tf_lite.Interpreter(model_path=self.config.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self._is_loaded = True
            
            # 获取模型信息
            input_shape = self.input_details[0]['shape']
            model_size = os.path.getsize(self.config.model_path) / 1024 / 1024
            
            print(f"[INFO] MoveNet 已加载: {self.config.model_path}")
            print(f"       输入形状: {input_shape}, 大小: {model_size:.1f} MB")
            return True
            
        except Exception as e:
            print(f"[ERROR] MoveNet 加载失败: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[PersonDetection]:
        """检测人体姿态"""
        if not self._is_loaded:
            return []
        
        h, w = image.shape[:2]
        
        # 预处理
        input_tensor = self._preprocess(image)
        
        # 推理
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # 获取输出 [1, 1, 17, 3]
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 后处理
        detections = self._postprocess(keypoints_with_scores, h, w)
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        input_size = self.config.input_size
        
        # 保持比例缩放并填充
        h, w = image.shape[:2]
        scale = min(input_size / w, input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建填充图像
        padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        pad_x = (input_size - new_w) // 2
        pad_y = (input_size - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # BGR -> RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # 添加 batch 维度
        # MoveNet 输入类型取决于模型版本
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.uint8:
            input_tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)
        else:
            input_tensor = np.expand_dims(rgb, axis=0).astype(np.int32)
        
        return input_tensor
    
    def _postprocess(
        self, 
        keypoints_with_scores: np.ndarray, 
        orig_h: int, 
        orig_w: int
    ) -> List[PersonDetection]:
        """后处理输出"""
        # 输出格式: [1, 1, 17, 3] - 17个关键点, 每个 (y, x, score)
        # 注意: MoveNet 输出的是归一化坐标 (y, x), 不是 (x, y)
        
        kpts = keypoints_with_scores[0, 0]  # [17, 3]
        
        # 计算平均置信度
        avg_confidence = np.mean(kpts[:, 2])
        
        # 过滤低置信度
        if avg_confidence < self.config.confidence_threshold:
            return []
        
        # 转换坐标: (y, x) -> (x, y) 并缩放到原始尺寸
        input_size = self.config.input_size
        scale = max(orig_w, orig_h) / input_size
        
        # 计算偏移 (因为我们做了填充)
        if orig_w > orig_h:
            offset_x = 0
            offset_y = (input_size - orig_h * input_size / orig_w) / 2
        else:
            offset_x = (input_size - orig_w * input_size / orig_h) / 2
            offset_y = 0
        
        keypoints = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            # MoveNet 输出: (y, x, score)
            y_norm, x_norm, score = kpts[i]
            
            # 转换到原始图像坐标
            x = (x_norm * input_size - offset_x) * scale
            y = (y_norm * input_size - offset_y) * scale
            
            keypoints[i] = [x, y, score]
        
        # 计算边界框 (从关键点)
        valid_kpts = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_kpts) > 0:
            x_min = np.min(valid_kpts[:, 0])
            y_min = np.min(valid_kpts[:, 1])
            x_max = np.max(valid_kpts[:, 0])
            y_max = np.max(valid_kpts[:, 1])
            
            # 扩展边界框
            w = x_max - x_min
            h = y_max - y_min
            x_min = max(0, x_min - w * 0.1)
            y_min = max(0, y_min - h * 0.1)
            x_max = min(orig_w, x_max + w * 0.1)
            y_max = min(orig_h, y_max + h * 0.1)
            
            bbox = np.array([x_min, y_min, x_max, y_max])
        else:
            # 如果没有有效关键点，使用全图
            bbox = np.array([0, 0, orig_w, orig_h])
        
        detection = PersonDetection(
            bbox=bbox,
            confidence=avg_confidence,
            keypoints=keypoints
        )
        
        return [detection]  # MoveNet single-pose 只返回一个人
    
    def compute_person_feature(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """计算人体特征向量 (与 YOLOv8 接口兼容)"""
        if keypoints is None or len(keypoints) < 17:
            return None
        
        # 使用关键点的相对位置作为特征
        # 以髋部中心为原点，归一化
        left_hip = keypoints[11, :2]
        right_hip = keypoints[12, :2]
        hip_center = (left_hip + right_hip) / 2
        
        # 计算身体尺度 (肩宽)
        left_shoulder = keypoints[5, :2]
        right_shoulder = keypoints[6, :2]
        body_scale = np.linalg.norm(left_shoulder - right_shoulder)
        
        if body_scale < 10:  # 太小，可能是噪声
            return None
        
        # 归一化关键点位置
        normalized = np.zeros((17, 2), dtype=np.float32)
        for i in range(17):
            if keypoints[i, 2] > 0.3:  # 只用高置信度关键点
                normalized[i] = (keypoints[i, :2] - hip_center) / body_scale
        
        # 展平为特征向量
        feature = normalized.flatten()  # 34维
        
        # 归一化
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        
        return feature
    
    def release(self):
        """释放资源"""
        self.interpreter = None
        self._is_loaded = False


# 测试
if __name__ == "__main__":
    import time
    
    print("\n[测试 MoveNet 检测器]\n")
    
    config = MoveNetConfig()
    detector = MoveNetDetector(config)
    
    if not detector.load():
        print("请先运行 download_movenet.py 下载模型")
        sys.exit(1)
    
    # 打开摄像头测试
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("\n按 'q' 退出\n")
    
    fps = 0
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 检测
        detections = detector.detect(frame)
        
        # 绘制
        for det in detections:
            # 边界框
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 关键点
            if det.keypoints is not None:
                for kpt in det.keypoints:
                    x, y, conf = kpt
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
        
        # FPS
        fps = 0.9 * fps + 0.1 * (1.0 / (time.time() - t0))
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("MoveNet Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
