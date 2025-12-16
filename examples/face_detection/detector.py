"""
SCRFD 人脸检测器
SCRFD Face Detector using ONNX Runtime
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
from dataclasses import dataclass
from config import ModelConfig


@dataclass
class FaceDetection:
    """人脸检测结果"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float         # 置信度
    keypoints: Optional[np.ndarray] = None  # [5, 2] 五个关键点


class SCRFDDetector:
    """SCRFD 人脸检测器"""
    
    # SCRFD 模型的 stride 配置
    _feat_stride_fpn = [8, 16, 32]
    _num_anchors = 2
    
    # 每个 stride 对应的特征图尺寸 (640x640 输入)
    # stride 8: 80x80 = 6400 * 2 anchors = 12800
    # stride 16: 40x40 = 1600 * 2 anchors = 3200
    # stride 32: 20x20 = 400 * 2 anchors = 800
    
    def __init__(self, config: ModelConfig):
        """
        初始化检测器
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_names: List[str] = []
        
        # 用于缓存的变量
        self._anchor_centers_cache = {}
    
    def load(self) -> bool:
        """
        加载模型
        
        Returns:
            是否成功加载
        """
        try:
            # 配置推理设备
            providers = self._get_providers()
            
            self.session = ort.InferenceSession(
                self.config.model_path,
                providers=providers
            )
            
            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            print(f"[INFO] 模型已加载: {self.config.model_path}")
            print(f"       输入: {self.input_name}")
            print(f"       输出: {self.output_names}")
            print(f"       设备: {providers[0]}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            return False
    
    def _get_providers(self) -> List[str]:
        """获取 ONNX Runtime 推理设备"""
        available = ort.get_available_providers()
        
        if self.config.device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.config.device == "dml" and "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
    
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        检测人脸
        
        Args:
            image: BGR 图像
            
        Returns:
            检测结果列表
        """
        if self.session is None:
            return []
        
        # 预处理
        input_tensor, scale, pad = self._preprocess(image)
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 后处理
        detections = self._postprocess(outputs, scale, pad, image.shape[:2])
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        图像预处理
        
        Args:
            image: BGR 图像
            
        Returns:
            (输入张量, 缩放比例, 填充量)
        """
        input_h, input_w = self.config.input_size
        img_h, img_w = image.shape[:2]
        
        # 计算缩放比例 (保持宽高比)
        scale = min(input_w / img_w, input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建填充后的图像
        padded = np.full((input_h, input_w, 3), 127, dtype=np.uint8)
        pad_w = (input_w - new_w) // 2
        pad_h = (input_h - new_h) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # BGR -> RGB, 归一化, HWC -> CHW
        input_tensor = padded[:, :, ::-1].astype(np.float32)
        input_tensor = (input_tensor - 127.5) / 128.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, (pad_w, pad_h)
    
    def _postprocess(
        self, 
        outputs: List[np.ndarray], 
        scale: float, 
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> List[FaceDetection]:
        """
        后处理: 解码输出、NMS、坐标还原
        
        Args:
            outputs: 模型输出
            scale: 缩放比例
            pad: 填充量 (pad_w, pad_h)
            orig_shape: 原图尺寸 (h, w)
            
        Returns:
            检测结果列表
        """
        input_h, input_w = self.config.input_size
        
        # 解码所有尺度的输出
        all_scores = []
        all_bboxes = []
        all_kps = []
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            # 获取当前尺度的输出
            scores = outputs[idx]
            bbox_preds = outputs[idx + len(self._feat_stride_fpn)]
            
            # 关键点 (如果有)
            kps_preds = None
            if self.config.detect_keypoints and len(outputs) > 2 * len(self._feat_stride_fpn):
                kps_preds = outputs[idx + 2 * len(self._feat_stride_fpn)]
            
            # 获取锚点中心
            height = input_h // stride
            width = input_w // stride
            anchor_centers = self._get_anchor_centers(height, width, stride)
            
            # 解码边界框
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            
            # 过滤低置信度
            mask = scores >= self.config.confidence_threshold
            if not np.any(mask):
                continue
            
            scores = scores[mask]
            bbox_preds = bbox_preds[mask]
            anchor_centers = anchor_centers[mask]
            
            # 解码边界框
            bboxes = self._decode_bboxes(anchor_centers, bbox_preds, stride)
            
            # 解码关键点
            kps = None
            if kps_preds is not None:
                kps_preds = kps_preds.reshape(-1, 10)[mask]
                kps = self._decode_keypoints(anchor_centers, kps_preds, stride)
            
            all_scores.append(scores)
            all_bboxes.append(bboxes)
            if kps is not None:
                all_kps.append(kps)
        
        if len(all_scores) == 0:
            return []
        
        # 合并所有尺度
        scores = np.concatenate(all_scores)
        bboxes = np.concatenate(all_bboxes)
        kps = np.concatenate(all_kps) if all_kps else None
        
        # NMS
        keep = self._nms(bboxes, scores, self.config.nms_threshold)
        
        # 构建检测结果
        detections = []
        pad_w, pad_h = pad
        orig_h, orig_w = orig_shape
        
        for i in keep:
            # 还原到原图坐标
            bbox = bboxes[i].copy()
            bbox[0] = (bbox[0] - pad_w) / scale
            bbox[1] = (bbox[1] - pad_h) / scale
            bbox[2] = (bbox[2] - pad_w) / scale
            bbox[3] = (bbox[3] - pad_h) / scale
            
            # 裁剪到图像边界
            bbox[0] = max(0, min(bbox[0], orig_w))
            bbox[1] = max(0, min(bbox[1], orig_h))
            bbox[2] = max(0, min(bbox[2], orig_w))
            bbox[3] = max(0, min(bbox[3], orig_h))
            
            keypoints = None
            if kps is not None:
                keypoints = kps[i].copy().reshape(5, 2)
                keypoints[:, 0] = (keypoints[:, 0] - pad_w) / scale
                keypoints[:, 1] = (keypoints[:, 1] - pad_h) / scale
            
            detections.append(FaceDetection(
                bbox=bbox,
                confidence=float(scores[i]),
                keypoints=keypoints
            ))
        
        return detections
    
    def _get_anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        """获取锚点中心坐标"""
        key = (height, width, stride)
        if key not in self._anchor_centers_cache:
            y, x = np.mgrid[:height, :width]
            anchor_centers = np.stack([x, y], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)
            anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape(-1, 2)
            self._anchor_centers_cache[key] = anchor_centers
        return self._anchor_centers_cache[key].copy()
    
    def _decode_bboxes(
        self, 
        anchor_centers: np.ndarray, 
        bbox_preds: np.ndarray, 
        stride: int
    ) -> np.ndarray:
        """解码边界框"""
        bboxes = np.zeros_like(bbox_preds)
        bboxes[:, 0] = anchor_centers[:, 0] - bbox_preds[:, 0] * stride
        bboxes[:, 1] = anchor_centers[:, 1] - bbox_preds[:, 1] * stride
        bboxes[:, 2] = anchor_centers[:, 0] + bbox_preds[:, 2] * stride
        bboxes[:, 3] = anchor_centers[:, 1] + bbox_preds[:, 3] * stride
        return bboxes
    
    def _decode_keypoints(
        self, 
        anchor_centers: np.ndarray, 
        kps_preds: np.ndarray, 
        stride: int
    ) -> np.ndarray:
        """解码关键点"""
        kps = np.zeros_like(kps_preds)
        for i in range(5):
            kps[:, i*2] = anchor_centers[:, 0] + kps_preds[:, i*2] * stride
            kps[:, i*2+1] = anchor_centers[:, 1] + kps_preds[:, i*2+1] * stride
        return kps
    
    def _nms(
        self, 
        bboxes: np.ndarray, 
        scores: np.ndarray, 
        threshold: float
    ) -> List[int]:
        """非极大值抑制"""
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        
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
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
