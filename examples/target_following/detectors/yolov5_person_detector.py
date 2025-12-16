"""
YOLOv5-Nano 人体检测器
YOLOv5-Nano Person Detector (ONNX)

特点:
  - 支持多人检测
  - 轻量化: ~3.9MB
  - 精度好, 速度快
  - 适合 S300 部署 (需替换 SiLU → ReLU)
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class YOLOv5PersonConfig:
    """YOLOv5-Nano 人体检测配置"""
    model_path: str = ""
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    # COCO 类别中 person 的索引
    person_class_id: int = 0


@dataclass
class PersonDetection:
    """人体检测结果"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    keypoints: Optional[np.ndarray] = None  # YOLOv5-Nano 不带姿态


class YOLOv5PersonDetector:
    """YOLOv5-Nano 人体检测器"""
    
    def __init__(self, config: YOLOv5PersonConfig):
        self.config = config
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_names: List[str] = []
        self.input_dtype = np.float32  # 默认 float32
        self._is_loaded = False
    
    def load(self) -> bool:
        """加载模型"""
        if not os.path.exists(self.config.model_path):
            print(f"[ERROR] 模型文件不存在: {self.config.model_path}")
            return False
        
        try:
            self.session = ort.InferenceSession(
                self.config.model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            # 检测输入数据类型
            input_info = self.session.get_inputs()[0]
            self.input_dtype = np.float16 if input_info.type == 'tensor(float16)' else np.float32
            
            self._is_loaded = True
            
            input_shape = self.session.get_inputs()[0].shape
            print(f"[INFO] YOLOv5-Nano 人体检测器已加载: {self.config.model_path}")
            print(f"       输入形状: {input_shape}, 类型: {input_info.type}")
            return True
            
        except Exception as e:
            print(f"[ERROR] YOLOv5-Nano 加载失败: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        检测人体
        
        Args:
            frame: BGR 图像
            
        Returns:
            人体检测结果列表
        """
        if not self._is_loaded:
            return []
        
        # 预处理
        input_tensor, ratio, pad = self._preprocess(frame)
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 后处理
        detections = self._postprocess(outputs[0], ratio, pad, frame.shape[:2])
        
        return detections
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        图像预处理 (letterbox resize)
        
        Returns:
            input_tensor: 预处理后的张量
            ratio: 缩放比例
            pad: 填充 (pad_w, pad_h)
        """
        input_h, input_w = self.config.input_size
        img_h, img_w = frame.shape[:2]
        
        # 计算缩放比例 (保持宽高比)
        ratio = min(input_w / img_w, input_h / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        # 缩放
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 计算填充
        pad_w = (input_w - new_w) // 2
        pad_h = (input_h - new_h) // 2
        
        # 创建填充后的图像 (灰色填充)
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # BGR -> RGB, HWC -> CHW, 归一化
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(blob, axis=0).astype(self.input_dtype)
        
        return input_tensor, ratio, (pad_w, pad_h)
    
    def _postprocess(
        self, 
        output: np.ndarray, 
        ratio: float, 
        pad: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> List[PersonDetection]:
        """
        后处理: NMS + 坐标还原
        
        Args:
            output: 模型输出 [1, num_boxes, 85] 或 [num_boxes, 85]
            ratio: 预处理时的缩放比例
            pad: 预处理时的填充 (pad_w, pad_h)
            original_shape: 原始图像尺寸 (h, w)
        """
        # 确保输出是 2D
        if len(output.shape) == 3:
            output = output[0]
        
        # output: [num_boxes, 85]
        # 85 = cx, cy, w, h, obj_conf, class0_conf, class1_conf, ...
        
        boxes = []
        scores = []
        
        pad_w, pad_h = pad
        orig_h, orig_w = original_shape
        
        for det in output:
            obj_conf = det[4]
            if obj_conf < self.config.confidence_threshold:
                continue
            
            # 获取类别置信度
            class_scores = det[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # 只保留 person 类别 (COCO class 0)
            if class_id != self.config.person_class_id:
                continue
            
            # 综合置信度
            score = obj_conf * class_conf
            if score < self.config.confidence_threshold:
                continue
            
            # 解析 bounding box (center x, center y, width, height)
            cx, cy, w, h = det[:4]
            
            # 转换为 x1, y1, x2, y2
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # 还原到原始图像坐标
            x1 = (x1 - pad_w) / ratio
            y1 = (y1 - pad_h) / ratio
            x2 = (x2 - pad_w) / ratio
            y2 = (y2 - pad_h) / ratio
            
            # 裁剪到图像边界
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
        
        if not boxes:
            return []
        
        # NMS
        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        
        indices = cv2.dnn.NMSBoxes(
            boxes_np.tolist(),
            scores_np.tolist(),
            self.config.confidence_threshold,
            self.config.nms_threshold
        )
        
        # 构建结果
        detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                detections.append(PersonDetection(
                    bbox=boxes_np[idx],
                    confidence=scores_np[idx],
                    keypoints=None
                ))
        
        return detections
    
    def release(self):
        """释放资源"""
        self.session = None
        self._is_loaded = False
