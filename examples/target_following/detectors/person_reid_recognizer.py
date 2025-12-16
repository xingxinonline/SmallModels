"""
MobileNetV2-ReID 人体识别器 (行人重识别)
MobileNetV2-based Person Re-Identification

特点:
  - 轻量化: ~13MB
  - 256D embedding
  - 适合嵌入式 ReID，实时跟随
  - 兼容性好，适合 S300 部署
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class PersonReIDConfig:
    """人体识别 (ReID) 配置"""
    model_path: str = ""
    input_size: Tuple[int, int] = (256, 128)  # (height, width) - Market1501 标准
    embedding_dim: int = 256
    # 特征相似度阈值 (余弦相似度)
    # 0.5-0.6: 严格匹配 (推荐)
    # 0.4-0.5: 中等
    # 0.3-0.4: 宽松
    similarity_threshold: float = 0.5


@dataclass
class PersonFeature:
    """人体特征"""
    embedding: np.ndarray     # 256D 特征向量
    bbox: np.ndarray          # 检测时的边界框
    timestamp: float = 0.0    # 特征提取时间


class PersonReIDRecognizer:
    """MobileNetV2-ReID 人体识别器"""
    
    def __init__(self, config: PersonReIDConfig):
        self.config = config
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self._is_loaded = False
    
    def load(self) -> bool:
        """加载模型"""
        if not os.path.exists(self.config.model_path):
            print(f"[ERROR] ReID 模型文件不存在: {self.config.model_path}")
            return False
        
        try:
            self.session = ort.InferenceSession(
                self.config.model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self._is_loaded = True
            
            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape
            print(f"[INFO] MobileNetV2-ReID 已加载: {self.config.model_path}")
            print(f"       输入: {input_shape}, 输出: {output_shape}")
            return True
            
        except Exception as e:
            print(f"[ERROR] ReID 模型加载失败: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def extract_feature(
        self, 
        frame: np.ndarray, 
        bbox: np.ndarray
    ) -> Optional[PersonFeature]:
        """
        提取人体特征
        
        Args:
            frame: BGR 完整图像
            bbox: 人体边界框 [x1, y1, x2, y2]
            
        Returns:
            PersonFeature 或 None
        """
        if not self._is_loaded:
            return None
        
        # 裁剪人体区域
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = frame.shape[:2]
        
        # 边界检查
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        person_crop = frame[y1:y2, x1:x2]
        
        # 预处理
        input_tensor = self._preprocess(person_crop)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        embedding = outputs[0][0]  # [embedding_dim]
        
        # L2 归一化
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        import time
        return PersonFeature(
            embedding=embedding,
            bbox=bbox.copy(),
            timestamp=time.time()
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: BGR 人体裁剪图像
            
        Returns:
            预处理后的张量 [1, 3, H, W]
        """
        target_h, target_w = self.config.input_size
        
        # 调整尺寸
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # BGR -> RGB
        rgb = resized[:, :, ::-1]
        
        # 归一化 (ImageNet 标准)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
        
        # HWC -> CHW
        chw = normalized.transpose(2, 0, 1)
        
        # 添加 batch 维度
        return np.expand_dims(chw, axis=0).astype(np.float32)
    
    def compute_similarity(
        self, 
        feature1: PersonFeature, 
        feature2: PersonFeature
    ) -> float:
        """
        计算两个人体特征的相似度 (余弦相似度)
        
        Args:
            feature1: 人体特征 1
            feature2: 人体特征 2
            
        Returns:
            相似度 [0, 1]
        """
        return float(np.dot(feature1.embedding, feature2.embedding))
    
    def is_same_person(
        self, 
        feature1: PersonFeature, 
        feature2: PersonFeature,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        判断是否同一人
        
        Args:
            feature1: 人体特征 1
            feature2: 人体特征 2
            threshold: 自定义阈值 (None 使用默认)
            
        Returns:
            (是否同一人, 相似度)
        """
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        similarity = self.compute_similarity(feature1, feature2)
        return similarity >= threshold, similarity
    
    def release(self):
        """释放资源"""
        self.session = None
        self._is_loaded = False
