"""
人体检测器 - 使用 YOLOv8-Pose
Person Detector using YOLOv8-Pose (ONNX)
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PersonDetectorConfig


@dataclass
class PersonDetection:
    """人体检测结果"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    keypoints: Optional[np.ndarray] = None  # [17, 3] COCO 姿态关键点 (x, y, conf)


class PersonDetector:
    """YOLOv8-Pose 人体检测器"""
    
    # COCO 姿态关键点名称
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def __init__(self, config: PersonDetectorConfig):
        self.config = config
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
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
            self._is_loaded = True
            
            # 获取输入形状
            input_shape = self.session.get_inputs()[0].shape
            print(f"[INFO] 人体检测器已加载: {self.config.model_path}")
            print(f"       输入形状: {input_shape}")
            return True
        except Exception as e:
            print(f"[ERROR] 人体检测器加载失败: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[PersonDetection]:
        """检测人体"""
        if not self._is_loaded:
            return []
        
        # 预处理
        input_tensor, ratio, pad = self._preprocess(image)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # 后处理
        detections = self._postprocess(outputs[0], ratio, pad, image.shape[:2])
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """图像预处理"""
        input_h, input_w = self.config.input_size
        img_h, img_w = image.shape[:2]
        
        # 计算缩放比例
        ratio = min(input_w / img_w, input_h / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        # 缩放
        resized = cv2.resize(image, (new_w, new_h))
        
        # 填充
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_w = (input_w - new_w) // 2
        pad_h = (input_h - new_h) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # BGR -> RGB, 归一化, HWC -> CHW
        input_tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, ratio, (pad_w, pad_h)
    
    def _postprocess(
        self, 
        output: np.ndarray, 
        ratio: float, 
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> List[PersonDetection]:
        """后处理"""
        # YOLOv8 输出格式: [1, 56, 8400]
        # 56 = 4 (bbox) + 1 (conf) + 17*3 (keypoints)
        
        predictions = output[0].T  # [8400, 56]
        
        # 提取边界框和置信度
        boxes = predictions[:, :4]  # [cx, cy, w, h]
        scores = predictions[:, 4]
        keypoints_raw = predictions[:, 5:]  # [17*3]
        
        # 过滤低置信度
        mask = scores >= self.config.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        keypoints_raw = keypoints_raw[mask]
        
        if len(boxes) == 0:
            return []
        
        # 转换边界框格式: [cx, cy, w, h] -> [x1, y1, x2, y2]
        bboxes = np.zeros_like(boxes)
        bboxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        bboxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        bboxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        bboxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # NMS
        keep = self._nms(bboxes, scores, self.config.nms_threshold)
        
        # 构建检测结果
        detections = []
        pad_w, pad_h = pad
        orig_h, orig_w = orig_shape
        
        for i in keep:
            # 还原边界框坐标
            bbox = bboxes[i].copy()
            bbox[0] = (bbox[0] - pad_w) / ratio
            bbox[1] = (bbox[1] - pad_h) / ratio
            bbox[2] = (bbox[2] - pad_w) / ratio
            bbox[3] = (bbox[3] - pad_h) / ratio
            
            # 裁剪到图像边界
            bbox[0] = max(0, min(bbox[0], orig_w))
            bbox[1] = max(0, min(bbox[1], orig_h))
            bbox[2] = max(0, min(bbox[2], orig_w))
            bbox[3] = max(0, min(bbox[3], orig_h))
            
            # 处理关键点
            kps = keypoints_raw[i].reshape(17, 3).copy()
            kps[:, 0] = (kps[:, 0] - pad_w) / ratio
            kps[:, 1] = (kps[:, 1] - pad_h) / ratio
            
            detections.append(PersonDetection(
                bbox=bbox,
                confidence=float(scores[i]),
                keypoints=kps
            ))
        
        return detections
    
    def _nms(self, bboxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """非极大值抑制"""
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    @staticmethod
    def compute_person_feature(keypoints: np.ndarray) -> np.ndarray:
        """
        根据姿态关键点计算人体特征向量
        用于人体再识别 (ReID)
        
        Args:
            keypoints: [17, 3] 姿态关键点
            
        Returns:
            特征向量
        """
        # 使用关键点的相对位置作为简单特征
        # 计算躯干比例、肢体比例等
        
        valid_mask = keypoints[:, 2] > 0.3
        
        if not np.any(valid_mask):
            return np.zeros(34)
        
        # 归一化关键点
        valid_pts = keypoints[valid_mask, :2]
        center = valid_pts.mean(axis=0)
        scale = np.std(valid_pts) + 1e-6
        
        normalized = (keypoints[:, :2] - center) / scale
        
        return normalized.flatten()
