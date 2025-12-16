"""
人脸识别器 - 使用 ArcFace 提取特征向量
Face Recognizer using ArcFace for feature extraction
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FaceRecognizerConfig


@dataclass
class FaceFeature:
    """人脸特征"""
    embedding: np.ndarray  # 特征向量 (512-d)
    norm: float           # 特征向量范数


class FaceRecognizer:
    """人脸识别器 (特征提取)"""
    
    def __init__(self, config: FaceRecognizerConfig):
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
            print(f"[INFO] 人脸识别器已加载: {self.config.model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 人脸识别器加载失败: {e}")
            return False
    
    def extract_feature(
        self, 
        image: np.ndarray, 
        face_bbox: np.ndarray = None,
        face_keypoints: Optional[np.ndarray] = None,
        # 别名参数 (与 ShuffleFaceNet 接口兼容)
        bbox: np.ndarray = None,
        keypoints: Optional[np.ndarray] = None
    ) -> Optional[FaceFeature]:
        """
        提取人脸特征
        
        Args:
            image: BGR 图像
            face_bbox: 人脸边界框 [x1, y1, x2, y2]
            face_keypoints: 人脸关键点 [5, 2] (可选，用于对齐)
            bbox: face_bbox 的别名
            keypoints: face_keypoints 的别名
            
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
        embedding = embedding / norm
        
        return FaceFeature(embedding=embedding, norm=norm)
    
    def _align_face(
        self, 
        image: np.ndarray, 
        bbox: np.ndarray,
        keypoints: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        裁剪并对齐人脸
        
        如果有关键点，使用仿射变换对齐
        否则直接裁剪
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # 扩展边界框
        face_w = x2 - x1
        face_h = y2 - y1
        margin = int(max(face_w, face_h) * 0.2)
        
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        if keypoints is not None and len(keypoints) >= 5:
            # 使用关键点进行仿射变换对齐
            aligned = self._warp_face(image, keypoints)
            if aligned is not None:
                return aligned
        
        # 直接裁剪和缩放
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        
        aligned = cv2.resize(face_crop, self.config.input_size)
        return aligned
    
    def _warp_face(
        self, 
        image: np.ndarray, 
        keypoints: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        使用仿射变换对齐人脸
        
        标准人脸关键点位置 (112x112):
        左眼, 右眼, 鼻尖, 左嘴角, 右嘴角
        """
        # 标准人脸关键点 (112x112 图像)
        dst_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        src_pts = keypoints[:5].astype(np.float32)
        
        try:
            # 计算仿射变换矩阵
            M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
            if M is None:
                return None
            
            # 应用变换
            aligned = cv2.warpAffine(
                image, M, self.config.input_size,
                borderValue=(127, 127, 127)
            )
            return aligned
        except Exception:
            return None
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理"""
        # BGR -> RGB
        input_tensor = image[:, :, ::-1].astype(np.float32)
        # 归一化
        input_tensor = (input_tensor - 127.5) / 127.5
        # HWC -> CHW
        input_tensor = input_tensor.transpose(2, 0, 1)
        # 添加 batch 维度
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor
    
    @staticmethod
    def compute_similarity(feat1: FaceFeature, feat2: FaceFeature) -> float:
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
    
    def release(self):
        """释放资源"""
        self.session = None
        self._is_loaded = False
