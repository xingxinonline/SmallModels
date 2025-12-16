"""
增强版人体特征提取器 - 多粒度颜色 + 纹理 + 几何
Enhanced Person Feature Extractor - Multi-level Color + Texture + Geometry

特征组成:
  1. 分区颜色直方图 (6段 HSV/LAB)
  2. 轻量 LBP 纹理特征
  3. 几何特征 (宽高比、面积比例)
  4. 光照鲁棒性增强 (LAB空间 + Bhattacharyya距离)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import time


@dataclass
class EnhancedReIDConfig:
    """增强版 ReID 配置"""
    # 分区设置
    num_horizontal_parts: int = 6  # 水平分段数 (头/肩/胸/腰/大腿/小腿)
    
    # 颜色直方图
    h_bins: int = 16    # 色相 bins (减少以提高鲁棒性)
    s_bins: int = 16    # 饱和度 bins
    use_lab: bool = True  # 使用 LAB 色彩空间 (更鲁棒)
    
    # LBP 纹理
    use_lbp: bool = True
    lbp_radius: int = 1
    lbp_points: int = 8
    
    # 几何特征
    use_geometry: bool = True
    
    # 相似度权重
    color_weight: float = 0.5
    texture_weight: float = 0.3
    geometry_weight: float = 0.2
    
    # 阈值
    similarity_threshold: float = 0.65


@dataclass
class EnhancedPersonFeature:
    """增强版人体特征"""
    # 分区颜色直方图 [num_parts, hist_dim]
    part_color_hists: np.ndarray = None
    
    # LBP 纹理直方图 [num_parts, 256]
    part_lbp_hists: np.ndarray = None
    
    # 几何特征 [aspect_ratio, relative_height, area_ratio]
    geometry: np.ndarray = None
    
    # 合并特征向量 (用于快速比较)
    combined_feature: np.ndarray = None
    
    # 元数据
    bbox: np.ndarray = None
    timestamp: float = 0.0
    frame_height: int = 0  # 用于计算相对高度


class EnhancedReIDExtractor:
    """增强版 ReID 特征提取器"""
    
    # 分区名称 (6段)
    PART_NAMES = ["head", "shoulder", "chest", "waist", "thigh", "calf"]
    
    # 分区比例 (从上到下)
    PART_RATIOS = [0.12, 0.08, 0.15, 0.15, 0.25, 0.25]
    
    def __init__(self, config: EnhancedReIDConfig = None):
        self.config = config or EnhancedReIDConfig()
        self._validate_ratios()
    
    def _validate_ratios(self):
        """确保分区比例和为1"""
        total = sum(self.PART_RATIOS)
        if abs(total - 1.0) > 0.01:
            self.PART_RATIOS = [r / total for r in self.PART_RATIOS]
    
    def load(self) -> bool:
        """兼容接口"""
        print(f"[INFO] 增强版 ReID 已初始化")
        print(f"       分区: {self.config.num_horizontal_parts} 段")
        print(f"       特征: 颜色({self.config.color_weight}) + "
              f"纹理({self.config.texture_weight}) + "
              f"几何({self.config.geometry_weight})")
        return True
    
    @property
    def is_loaded(self) -> bool:
        return True
    
    def extract_feature(
        self,
        frame: np.ndarray,
        bbox: np.ndarray
    ) -> Optional[EnhancedPersonFeature]:
        """
        提取增强版人体特征
        
        Args:
            frame: BGR 完整图像
            bbox: 人体边界框 [x1, y1, x2, y2]
            
        Returns:
            EnhancedPersonFeature 或 None
        """
        # 裁剪人体区域
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = frame.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        person_crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = person_crop.shape[:2]
        
        if crop_h < 20 or crop_w < 10:
            return None
        
        # 分区
        parts = self._split_into_parts(person_crop)
        
        # 提取各分区特征
        part_color_hists = []
        part_lbp_hists = []
        
        for part in parts:
            # 颜色直方图
            color_hist = self._compute_color_histogram(part)
            part_color_hists.append(color_hist)
            
            # LBP 纹理直方图
            if self.config.use_lbp:
                lbp_hist = self._compute_lbp_histogram(part)
                part_lbp_hists.append(lbp_hist)
        
        part_color_hists = np.array(part_color_hists)
        part_lbp_hists = np.array(part_lbp_hists) if part_lbp_hists else None
        
        # 几何特征
        geometry = None
        if self.config.use_geometry:
            geometry = self._compute_geometry(bbox, frame.shape[:2])
        
        # 合并特征
        combined = self._combine_features(part_color_hists, part_lbp_hists, geometry)
        
        return EnhancedPersonFeature(
            part_color_hists=part_color_hists,
            part_lbp_hists=part_lbp_hists,
            geometry=geometry,
            combined_feature=combined,
            bbox=bbox.copy(),
            timestamp=time.time(),
            frame_height=h
        )
    
    def _split_into_parts(self, person_crop: np.ndarray) -> List[np.ndarray]:
        """将人体裁剪图分成多个水平条带"""
        h = person_crop.shape[0]
        parts = []
        
        y_start = 0
        for ratio in self.PART_RATIOS[:self.config.num_horizontal_parts]:
            y_end = y_start + int(h * ratio)
            y_end = min(y_end, h)
            
            if y_end > y_start:
                parts.append(person_crop[y_start:y_end])
            
            y_start = y_end
        
        # 如果分区不够，用剩余部分补充
        if y_start < h and len(parts) < self.config.num_horizontal_parts:
            parts.append(person_crop[y_start:h])
        
        return parts
    
    def _compute_color_histogram(self, part: np.ndarray) -> np.ndarray:
        """计算颜色直方图 (LAB + HSV 融合)"""
        hists = []
        
        # LAB 空间直方图 (光照鲁棒)
        if self.config.use_lab:
            lab = cv2.cvtColor(part, cv2.COLOR_BGR2LAB)
            # 只用 A、B 通道 (颜色)，忽略 L (亮度)
            ab_hist = cv2.calcHist(
                [lab], [1, 2], None,
                [self.config.h_bins, self.config.s_bins],
                [0, 256, 0, 256]
            )
            ab_hist = ab_hist.flatten()
            ab_hist = ab_hist / (ab_hist.sum() + 1e-8)
            hists.append(ab_hist)
        
        # HSV 空间直方图
        hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        hs_hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.config.h_bins, self.config.s_bins],
            [0, 180, 0, 256]
        )
        hs_hist = hs_hist.flatten()
        hs_hist = hs_hist / (hs_hist.sum() + 1e-8)
        hists.append(hs_hist)
        
        # 合并
        return np.concatenate(hists) if len(hists) > 1 else hists[0]
    
    def _compute_lbp_histogram(self, part: np.ndarray) -> np.ndarray:
        """计算 LBP 纹理直方图 (轻量版)"""
        # 转灰度
        gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
        
        # 简化 LBP (3x3 邻域)
        h, w = gray.shape
        if h < 3 or w < 3:
            return np.zeros(256)
        
        # 使用 OpenCV 的快速实现 (近似 LBP)
        # 计算 8 个方向的梯度差
        lbp = np.zeros_like(gray, dtype=np.uint8)
        
        center = gray[1:-1, 1:-1]
        
        # 8 邻域比较
        neighbors = [
            gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],  # 上
            gray[1:-1, 0:-2],                    gray[1:-1, 2:],  # 左右
            gray[2:, 0:-2],   gray[2:, 1:-1],   gray[2:, 2:]     # 下
        ]
        
        for i, neighbor in enumerate(neighbors):
            lbp[1:-1, 1:-1] |= ((neighbor >= center).astype(np.uint8) << i)
        
        # 计算直方图
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        
        return hist
    
    def _compute_geometry(
        self,
        bbox: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """计算几何特征"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        frame_h, frame_w = frame_shape
        
        # 宽高比 (正常人约 0.4-0.5)
        aspect_ratio = w / (h + 1e-8)
        
        # 相对高度 (占画面比例)
        relative_height = h / frame_h
        
        # 面积占比
        area_ratio = (w * h) / (frame_w * frame_h + 1e-8)
        
        # 中心位置 (归一化)
        center_x = ((x1 + x2) / 2) / frame_w
        center_y = ((y1 + y2) / 2) / frame_h
        
        return np.array([
            aspect_ratio,
            relative_height,
            area_ratio,
            center_x,
            center_y
        ], dtype=np.float32)
    
    def _combine_features(
        self,
        color_hists: np.ndarray,
        lbp_hists: Optional[np.ndarray],
        geometry: Optional[np.ndarray]
    ) -> np.ndarray:
        """合并所有特征为单一向量"""
        features = [color_hists.flatten()]
        
        if lbp_hists is not None:
            features.append(lbp_hists.flatten())
        
        if geometry is not None:
            # 几何特征需要归一化
            features.append(geometry)
        
        combined = np.concatenate(features)
        
        # L2 归一化
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        
        return combined
    
    def compute_similarity(
        self,
        feature1: EnhancedPersonFeature,
        feature2: EnhancedPersonFeature
    ) -> Tuple[float, dict]:
        """
        计算两个人体特征的相似度
        
        Returns:
            (总相似度, 各分量相似度字典)
        """
        details = {}
        
        # 1. 颜色相似度 (分区加权 Bhattacharyya)
        color_sims = []
        for i in range(len(feature1.part_color_hists)):
            h1 = feature1.part_color_hists[i].astype(np.float32)
            h2 = feature2.part_color_hists[i].astype(np.float32)
            
            # Bhattacharyya 距离 (0=完全相同)
            bc = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
            # 转换为相似度 [0, 1]
            sim = 1.0 - bc
            color_sims.append(sim)
        
        # 上身权重更高 (更容易被看到)
        part_weights = [0.10, 0.15, 0.20, 0.20, 0.20, 0.15][:len(color_sims)]
        part_weights = np.array(part_weights) / sum(part_weights)
        color_sim = float(np.dot(color_sims, part_weights))
        details['color'] = color_sim
        details['color_parts'] = color_sims
        
        # 2. 纹理相似度
        texture_sim = 0.0
        if feature1.part_lbp_hists is not None and feature2.part_lbp_hists is not None:
            lbp_sims = []
            for i in range(len(feature1.part_lbp_hists)):
                h1 = feature1.part_lbp_hists[i].astype(np.float32)
                h2 = feature2.part_lbp_hists[i].astype(np.float32)
                sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                sim = (sim + 1) / 2  # 归一化到 [0, 1]
                lbp_sims.append(sim)
            
            texture_sim = float(np.mean(lbp_sims))
        details['texture'] = texture_sim
        
        # 3. 几何相似度
        geo_sim = 1.0  # 默认完全匹配
        if feature1.geometry is not None and feature2.geometry is not None:
            # 只比较宽高比和相对高度 (前2个)
            diff = np.abs(feature1.geometry[:2] - feature2.geometry[:2])
            # 允许一定的变化范围
            geo_sim = 1.0 - np.clip(np.mean(diff) * 2, 0, 1)
        details['geometry'] = geo_sim
        
        # 加权融合
        total_sim = (
            color_sim * self.config.color_weight +
            texture_sim * self.config.texture_weight +
            geo_sim * self.config.geometry_weight
        )
        
        return float(total_sim), details
    
    def is_same_person(
        self,
        feature1: EnhancedPersonFeature,
        feature2: EnhancedPersonFeature,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float, dict]:
        """
        判断是否同一人
        
        Returns:
            (是否同一人, 相似度, 详情)
        """
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        similarity, details = self.compute_similarity(feature1, feature2)
        
        return similarity >= threshold, similarity, details
    
    def release(self):
        """释放资源"""
        pass


# 测试
if __name__ == "__main__":
    import cv2
    
    extractor = EnhancedReIDExtractor()
    extractor.load()
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = np.array([200, 50, 400, 450])
    
    feature = extractor.extract_feature(test_img, test_bbox)
    
    if feature:
        print(f"颜色直方图形状: {feature.part_color_hists.shape}")
        print(f"LBP 直方图形状: {feature.part_lbp_hists.shape}")
        print(f"几何特征: {feature.geometry}")
        print(f"合并特征维度: {len(feature.combined_feature)}")
        
        # 自己比较自己
        is_same, sim, details = extractor.is_same_person(feature, feature)
        print(f"\n自比较: is_same={is_same}, sim={sim:.3f}")
        print(f"详情: {details}")
