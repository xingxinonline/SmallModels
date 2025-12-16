"""
颜色直方图 ReID - 简单但有效的人体识别
Color Histogram based Person Re-Identification

优点:
  - 不需要预训练模型
  - 计算快速 (<1ms)
  - 对服装颜色敏感（实际场景中有效）

缺点:
  - 对光照变化敏感
  - 相同颜色衣服的人无法区分

策略:
  - 分区域提取颜色直方图（上身/下身）
  - 使用 HSV 色彩空间（对光照更鲁棒）
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class ColorHistogramConfig:
    """颜色直方图 ReID 配置"""
    # 直方图 bins
    h_bins: int = 30   # 色相 (0-180)
    s_bins: int = 32   # 饱和度 (0-255)
    v_bins: int = 32   # 明度 (0-255) - 可选，减少光照影响
    
    # 是否使用 V 通道
    use_value: bool = False  # 不用 V 通道，更鲁棒
    
    # 上下身分割比例
    upper_ratio: float = 0.45  # 上身占 45%
    lower_ratio: float = 0.55  # 下身占 55%
    
    # 相似度阈值 (直方图比较，越高越相似)
    # 使用 HISTCMP_CORREL: 1 = 完全匹配, 0 = 无关, -1 = 相反
    similarity_threshold: float = 0.7


@dataclass
class PersonColorFeature:
    """人体颜色特征"""
    upper_hist: np.ndarray    # 上身直方图
    lower_hist: np.ndarray    # 下身直方图
    combined_hist: np.ndarray # 合并直方图
    bbox: np.ndarray
    timestamp: float = 0.0


class ColorHistogramReID:
    """颜色直方图 ReID"""
    
    def __init__(self, config: ColorHistogramConfig = None):
        self.config = config or ColorHistogramConfig()
        self._is_loaded = True  # 不需要加载模型
    
    def load(self) -> bool:
        """兼容接口"""
        print("[INFO] 颜色直方图 ReID 已初始化")
        return True
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def extract_feature(
        self, 
        frame: np.ndarray, 
        bbox: np.ndarray
    ) -> Optional[PersonColorFeature]:
        """
        提取人体颜色特征
        
        Args:
            frame: BGR 完整图像
            bbox: 人体边界框 [x1, y1, x2, y2]
            
        Returns:
            PersonColorFeature 或 None
        """
        # 裁剪人体区域
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = frame.shape[:2]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        person_crop = frame[y1:y2, x1:x2]
        crop_h = person_crop.shape[0]
        
        # 分割上下身
        split_y = int(crop_h * self.config.upper_ratio)
        upper_body = person_crop[:split_y]
        lower_body = person_crop[split_y:]
        
        # 转换为 HSV
        upper_hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
        lower_hsv = cv2.cvtColor(lower_body, cv2.COLOR_BGR2HSV)
        
        # 计算直方图
        upper_hist = self._compute_histogram(upper_hsv)
        lower_hist = self._compute_histogram(lower_hsv)
        
        # 合并直方图 (上身权重稍高，因为更容易被看到)
        combined_hist = np.concatenate([upper_hist * 0.55, lower_hist * 0.45])
        
        return PersonColorFeature(
            upper_hist=upper_hist,
            lower_hist=lower_hist,
            combined_hist=combined_hist,
            bbox=bbox.copy(),
            timestamp=time.time()
        )
    
    def _compute_histogram(self, hsv_image: np.ndarray) -> np.ndarray:
        """计算 HSV 直方图"""
        if self.config.use_value:
            # H-S-V 直方图
            hist = cv2.calcHist(
                [hsv_image], 
                [0, 1, 2], 
                None, 
                [self.config.h_bins, self.config.s_bins, self.config.v_bins],
                [0, 180, 0, 256, 0, 256]
            )
        else:
            # 只用 H-S 直方图 (对光照更鲁棒)
            hist = cv2.calcHist(
                [hsv_image], 
                [0, 1], 
                None, 
                [self.config.h_bins, self.config.s_bins],
                [0, 180, 0, 256]
            )
        
        # 归一化
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-8)
        
        return hist
    
    def compute_similarity(
        self, 
        feature1: PersonColorFeature, 
        feature2: PersonColorFeature
    ) -> float:
        """
        计算两个人体特征的相似度
        
        使用相关性比较 (HISTCMP_CORREL)
        返回值: [-1, 1], 1 = 完全匹配
        """
        # 比较合并直方图
        similarity = cv2.compareHist(
            feature1.combined_hist.astype(np.float32),
            feature2.combined_hist.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        # 归一化到 [0, 1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def is_same_person(
        self, 
        feature1: PersonColorFeature, 
        feature2: PersonColorFeature,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        判断是否同一人
        
        Returns:
            (是否同一人, 相似度)
        """
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        similarity = self.compute_similarity(feature1, feature2)
        return similarity >= threshold, similarity
    
    def release(self):
        """释放资源"""
        pass
