"""
MobileFaceNet 人脸识别器 - 轻量级人脸特征提取
MobileFaceNet Face Recognizer - Lightweight Face Feature Extraction

模型来源: InsightFace MobileFaceNet
特点: 模型仅 ~4MB, 推理速度快, 适合边缘设备
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MobileFaceNetConfig


@dataclass
class FaceFeature:
    """人脸特征"""
    embedding: np.ndarray  # 特征向量 (128-d for MobileFaceNet)
    norm: float           # 特征向量范数


class MobileFaceNetRecognizer:
    """MobileFaceNet 人脸识别器 (轻量级)"""
    
    # 标准人脸关键点 (用于对齐)
    REFERENCE_LANDMARKS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    
    def __init__(self, config: MobileFaceNetConfig):
        self.config = config
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self._is_loaded = False
    
    def load(self) -> bool:
        """加载模型"""
        if not os.path.exists(self.config.model_path):
            print(f"[ERROR] MobileFaceNet 模型文件不存在: {self.config.model_path}")
            print(f"[INFO] 请运行 download_mobilefacenet.py 下载模型")
            return False
        
        try:
            self.session = ort.InferenceSession(
                self.config.model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self._is_loaded = True
            print(f"[INFO] MobileFaceNet 已加载: {self.config.model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] MobileFaceNet 加载失败: {e}")
            return False
    
    def release(self):
        """释放资源"""
        self.session = None
        self._is_loaded = False
    
    def extract_feature(
        self, 
        image: np.ndarray, 
        face_bbox: np.ndarray = None,
        face_keypoints: Optional[np.ndarray] = None,
        # 别名参数 (与其他识别器接口兼容)
        bbox: np.ndarray = None,
        keypoints: Optional[np.ndarray] = None
    ) -> Optional[FaceFeature]:
        """
        提取人脸特征
        
        Args:
            image: BGR 图像
            face_bbox: 人脸边界框 [x1, y1, x2, y2]
            face_keypoints: 人脸关键点 [5, 2] (可选，用于对齐)
            
        Returns:
            人脸特征
        """
        # 处理别名参数
        if face_bbox is None and bbox is not None:
            face_bbox = bbox
        if face_keypoints is None and keypoints is not None:
            face_keypoints = keypoints
            
        if face_bbox is None:
            return None
            
        if not self._is_loaded:
            return None
        
        # 裁剪并对齐人脸
        aligned_face = self._align_face(image, face_bbox, face_keypoints)
        if aligned_face is None:
            return None
        
        # 预处理
        input_tensor = self._preprocess(aligned_face)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        embedding = outputs[0][0]
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return FaceFeature(embedding=embedding, norm=norm)
    
    def _align_face(
        self, 
        image: np.ndarray, 
        bbox: np.ndarray,
        keypoints: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """裁剪并对齐人脸"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # 边界检查
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        if keypoints is not None and len(keypoints) >= 5:
            # 使用关键点进行仿射变换对齐
            src_pts = keypoints[:5].astype(np.float32)
            dst_pts = self.REFERENCE_LANDMARKS.copy()
            
            # 缩放到目标尺寸
            scale = self.config.input_size[0] / 112.0
            dst_pts = dst_pts * scale
            
            # 计算仿射变换矩阵
            tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
            if tform is not None:
                aligned = cv2.warpAffine(
                    image, tform, self.config.input_size,
                    borderValue=(0, 0, 0)
                )
                return aligned
        
        # 简单裁剪 + 缩放
        face = image[y1:y2, x1:x2]
        aligned = cv2.resize(face, self.config.input_size)
        return aligned
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [-1, 1]
        image = image.astype(np.float32)
        image = (image - 127.5) / 127.5
        
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        
        # 添加 batch 维度
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def compute_similarity(self, feat1: FaceFeature, feat2: FaceFeature) -> float:
        """
        计算两个人脸特征的相似度 (余弦相似度)
        
        Returns:
            相似度 [-1, 1]，越大越相似
        """
        return float(np.dot(feat1.embedding, feat2.embedding))
    
    def is_same_person(self, feat1: FaceFeature, feat2: FaceFeature) -> Tuple[bool, float]:
        """
        判断是否为同一个人
        
        Returns:
            (是否同一人, 相似度)
        """
        similarity = self.compute_similarity(feat1, feat2)
        is_same = similarity >= self.config.similarity_threshold
        return is_same, similarity
