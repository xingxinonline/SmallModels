"""
ShuffleFaceNet 人脸识别器
ShuffleFaceNet Face Recognizer - 使用 PyTorch 模型
"""

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Optional, Dict
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ShuffleFaceNetConfig:
    """ShuffleFaceNet 配置"""
    model_path: str = ""
    vector_path: str = ""  # 预注册人脸向量
    similarity_threshold: float = 0.5
    input_size: tuple = (112, 112)


class ShuffleFaceNetRecognizer:
    """ShuffleFaceNet 人脸识别器"""
    
    # 标准人脸对齐参考点
    REF_POINTS = np.array([
        [30.2946 + 8, 51.6963],
        [65.5318 + 8, 51.5014],
        [48.0252 + 8, 71.7366],
        [33.5493 + 8, 92.3655],
        [62.7299 + 8, 92.2041]
    ], dtype=np.float32)
    
    def __init__(self, config: ShuffleFaceNetConfig):
        self.config = config
        self.model = None
        self.device = torch.device('cpu')
        self._is_loaded = False
        
        # 预注册的人脸向量
        self.vector_dict: Dict = {}
        self.mu_vector: Optional[np.ndarray] = None  # 均值向量
    
    def load(self) -> bool:
        """加载模型"""
        try:
            # 导入 detection_recognition 的模型
            det_recog_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "detection_recognition"
            )
            sys.path.insert(0, det_recog_path)
            
            from model.model_modify import ShuffleFaceNet
            from model.matlab_cp2tform import get_similarity_transform_for_cv2
            
            self.get_similarity_transform = get_similarity_transform_for_cv2
            
            # 加载模型
            self.model = ShuffleFaceNet()
            self.model = self._load_model(self.model, self.config.model_path)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            torch.set_grad_enabled(False)
            cudnn.benchmark = True
            
            # 加载预注册向量 (如果存在)
            if os.path.exists(self.config.vector_path):
                self._load_vectors()
            
            self._is_loaded = True
            print(f"[INFO] ShuffleFaceNet 人脸识别器已加载: {self.config.model_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] ShuffleFaceNet 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_model(self, model, pretrained_path):
        """加载预训练权重"""
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']
        
        # 移除 'module.' 前缀
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        
        model.load_state_dict(pretrained_dict, strict=False)
        return model
    
    def _load_vectors(self):
        """加载预注册的人脸向量"""
        try:
            data = np.load(self.config.vector_path, allow_pickle=True).item()
            self.mu_vector = data.get('mu', None)
            self.vector_dict = {k: v for k, v in data.items() if k != 'mu'}
            print(f"[INFO] 已加载 {len(self.vector_dict)} 个预注册人脸")
        except Exception as e:
            print(f"[WARNING] 加载预注册向量失败: {e}")
    
    def align_face(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        人脸对齐
        
        Args:
            image: BGR 图像
            keypoints: [5, 2] 关键点
            
        Returns:
            对齐后的人脸图像 (112x112)
        """
        src_pts = keypoints.astype(np.float32)
        tfm = self.get_similarity_transform(src_pts, self.REF_POINTS)
        aligned = cv2.warpAffine(image, tfm, self.config.input_size)
        return aligned
    
    def extract_feature(
        self, 
        image: np.ndarray, 
        bbox: np.ndarray = None,
        keypoints: np.ndarray = None
    ) -> Optional[np.ndarray]:
        """
        提取人脸特征
        
        Args:
            image: BGR 图像 (原图或已对齐的人脸)
            bbox: 人脸边界框 (可选)
            keypoints: 人脸关键点 (可选, 用于对齐)
            
        Returns:
            512维特征向量 (256 + 256 翻转)
        """
        if not self._is_loaded:
            return None
        
        try:
            # 如果有关键点，进行对齐
            if keypoints is not None:
                face_img = self.align_face(image, keypoints)
            elif bbox is not None:
                # 裁剪人脸区域
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                face_img = image[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, self.config.input_size)
            else:
                # 假设输入已经是对齐的人脸
                face_img = cv2.resize(image, self.config.input_size)
            
            # 原图 + 水平翻转
            img1 = face_img
            img2 = face_img[:, ::-1, :]
            
            # 预处理
            imgs = []
            for img in [img1, img2]:
                img = (img - 127.5) / 128.0
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).float().unsqueeze(0)
                imgs.append(img.to(self.device))
            
            # 提取特征
            with torch.no_grad():
                vec1 = self.model(imgs[0]).cpu().numpy()[0]
                vec2 = self.model(imgs[1]).cpu().numpy()[0]
            
            # 拼接特征
            feature = np.concatenate([vec1, vec2], axis=0)
            return feature
            
        except Exception as e:
            print(f"[WARNING] 特征提取失败: {e}")
            return None
    
    def normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        """归一化特征 (减去均值并L2归一化)"""
        if self.mu_vector is not None:
            feature = feature - self.mu_vector
        norm = np.sqrt(np.sum(feature ** 2))
        if norm > 0:
            feature = feature / norm
        return feature
    
    def compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        计算两个特征的相似度
        
        Args:
            feature1: 特征向量1 (已归一化或未归一化)
            feature2: 特征向量2
            
        Returns:
            余弦相似度 [-1, 1]
        """
        # 归一化
        f1 = self.normalize_feature(feature1.copy())
        f2 = self.normalize_feature(feature2.copy())
        
        # 余弦相似度
        similarity = np.dot(f1, f2)
        return float(similarity)
    
    def match_registered(self, feature: np.ndarray) -> tuple:
        """
        与预注册人脸匹配
        
        Args:
            feature: 待匹配特征
            
        Returns:
            (name, similarity) 或 ("unknown", -1)
        """
        if not self.vector_dict:
            return ("unknown", -1.0)
        
        # 归一化
        f = self.normalize_feature(feature.copy())
        
        best_name = "unknown"
        best_sim = -1.0
        
        for name, vec in self.vector_dict.items():
            sim = float(np.dot(f, vec))
            if sim > best_sim and sim > self.config.similarity_threshold:
                best_sim = sim
                best_name = name
        
        return (best_name, best_sim)
    
    def release(self):
        """释放资源"""
        self.model = None
        self._is_loaded = False
