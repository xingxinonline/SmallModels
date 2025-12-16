"""
融合目标识别器 - 人脸 + 增强版人体特征
Fused Target Recognizer - Face + Enhanced Body Features

策略:
  - 有人脸时: 人脸识别优先 (权重 0.6) + 人体特征 (权重 0.4)
  - 无人脸时: 100% 依赖人体特征
  - 人体特征: 分区颜色 + LBP纹理 + 几何
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class FusedTargetFeature:
    """融合目标特征"""
    face_embedding: Optional[np.ndarray] = None   # 人脸特征 (MobileFaceNet)
    body_feature: Optional[np.ndarray] = None     # 增强版人体特征 (合并向量)
    
    # 分区颜色直方图 (用于详细比较)
    part_color_hists: Optional[np.ndarray] = None
    part_lbp_hists: Optional[np.ndarray] = None
    geometry: Optional[np.ndarray] = None
    
    bbox: Optional[np.ndarray] = None             # 人体边界框
    face_bbox: Optional[np.ndarray] = None        # 人脸边界框
    timestamp: float = 0.0
    
    @property
    def has_face(self) -> bool:
        return self.face_embedding is not None
    
    @property
    def has_body(self) -> bool:
        return self.body_feature is not None or self.part_color_hists is not None


@dataclass
class FusedRecognizerConfig:
    """融合识别器配置"""
    # 权重配置 (两者都有时)
    face_weight: float = 0.6      # 人脸识别权重
    body_weight: float = 0.4      # 人体特征权重
    
    # 单独使用时的阈值
    face_only_threshold: float = 0.45     # 只有人脸时
    body_only_threshold: float = 0.55     # 只有人体时 (增强版)
    
    # 融合时的阈值
    fused_threshold: float = 0.50
    
    # 人脸匹配加分 (如果人脸高度匹配)
    face_match_bonus: float = 0.15
    
    # 人体特征各分量权重
    body_color_weight: float = 0.5
    body_texture_weight: float = 0.3
    body_geometry_weight: float = 0.2


class FusedTargetRecognizer:
    """融合目标识别器"""
    
    def __init__(self, config: FusedRecognizerConfig = None):
        self.config = config or FusedRecognizerConfig()
    
    def _compute_body_similarity(
        self,
        target: FusedTargetFeature,
        candidate: FusedTargetFeature
    ) -> Tuple[float, dict]:
        """
        计算人体特征相似度
        
        支持两种模式:
        1. 增强版: 分区颜色 + LBP + 几何
        2. 简化版: 合并特征向量
        """
        details = {}
        
        # 模式1: 如果有分区直方图，使用增强版
        if (target.part_color_hists is not None and 
            candidate.part_color_hists is not None):
            
            # 分区颜色相似度 (Bhattacharyya)
            color_sims = []
            num_parts = min(len(target.part_color_hists), len(candidate.part_color_hists))
            
            for i in range(num_parts):
                h1 = target.part_color_hists[i].astype(np.float32)
                h2 = candidate.part_color_hists[i].astype(np.float32)
                
                # 确保直方图非空
                if h1.sum() < 1e-8 or h2.sum() < 1e-8:
                    color_sims.append(0.5)
                    continue
                    
                # Bhattacharyya 距离 → 相似度
                bc = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
                sim = 1.0 - bc
                color_sims.append(sim)
            
            # 上身权重更高 (衣服颜色更稳定)
            if num_parts == 6:
                part_weights = [0.10, 0.15, 0.20, 0.20, 0.20, 0.15]
            else:
                part_weights = [1.0 / num_parts] * num_parts
                
            part_weights = np.array(part_weights[:num_parts])
            part_weights = part_weights / part_weights.sum()
            
            color_sim = float(np.dot(color_sims, part_weights))
            details['color'] = color_sim
            details['color_parts'] = color_sims
            
            # LBP 纹理相似度
            texture_sim = 0.5  # 默认值
            if (target.part_lbp_hists is not None and 
                candidate.part_lbp_hists is not None):
                lbp_sims = []
                num_lbp = min(len(target.part_lbp_hists), len(candidate.part_lbp_hists))
                
                for i in range(num_lbp):
                    h1 = target.part_lbp_hists[i].astype(np.float32)
                    h2 = candidate.part_lbp_hists[i].astype(np.float32)
                    
                    if h1.sum() < 1e-8 or h2.sum() < 1e-8:
                        lbp_sims.append(0.5)
                        continue
                    
                    # 相关性比较
                    sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                    sim = (sim + 1) / 2  # 映射到 [0, 1]
                    lbp_sims.append(sim)
                    
                texture_sim = float(np.mean(lbp_sims))
            details['texture'] = texture_sim
            
            # 几何相似度 (宽高比和相对尺寸)
            geo_sim = 0.8  # 默认值 (较高，因为几何变化不大)
            if target.geometry is not None and candidate.geometry is not None:
                # 比较前两个维度: aspect_ratio, relative_height
                g1 = target.geometry[:2]
                g2 = candidate.geometry[:2]
                diff = np.abs(g1 - g2)
                geo_sim = 1.0 - np.clip(np.mean(diff) * 2, 0, 1)
            details['geometry'] = geo_sim
            
            # 加权融合
            body_sim = (
                color_sim * self.config.body_color_weight +
                texture_sim * self.config.body_texture_weight +
                geo_sim * self.config.body_geometry_weight
            )
            
            return body_sim, details
        
        # 模式2: 使用合并特征向量 (余弦相似度)
        elif target.body_feature is not None and candidate.body_feature is not None:
            # 归一化
            t_norm = target.body_feature / (np.linalg.norm(target.body_feature) + 1e-8)
            c_norm = candidate.body_feature / (np.linalg.norm(candidate.body_feature) + 1e-8)
            
            body_sim = float(np.dot(t_norm, c_norm))
            details['combined'] = body_sim
            return body_sim, details
        
        return 0.0, {}
    
    def compute_similarity(
        self,
        target: FusedTargetFeature,
        candidate: FusedTargetFeature
    ) -> Tuple[float, str]:
        """
        计算融合相似度
        
        策略:
        - 目标有人脸，候选也有人脸 → 人脸为主 + 人体辅助
        - 目标有人脸，候选无人脸 → 只用人体 (背身)
        - 目标无人脸，候选有人脸 → 只用人体 (目标背身保存)
        - 都无人脸 → 只用人体
        
        Returns:
            (相似度, 匹配方式描述)
        """
        face_sim = None
        body_sim = None
        body_details = {}
        method = "none"
        
        # 计算人脸相似度 (两边都要有人脸才能比较)
        if target.has_face and candidate.has_face:
            face_sim = float(np.dot(target.face_embedding, candidate.face_embedding))
        
        # 计算人体相似度
        if target.has_body and candidate.has_body:
            body_sim, body_details = self._compute_body_similarity(target, candidate)
        
        # 融合策略
        if face_sim is not None and body_sim is not None:
            # 两者都有，加权融合
            if face_sim >= self.config.face_only_threshold:
                # 人脸匹配度高，给予额外加分
                similarity = (
                    face_sim * self.config.face_weight + 
                    body_sim * self.config.body_weight +
                    self.config.face_match_bonus
                )
                method = f"fused+bonus (F:{face_sim:.2f} B:{body_sim:.2f})"
            else:
                similarity = (
                    face_sim * self.config.face_weight + 
                    body_sim * self.config.body_weight
                )
                method = f"fused (F:{face_sim:.2f} B:{body_sim:.2f})"
            
            similarity = min(1.0, max(0.0, similarity))
            
        elif face_sim is not None:
            # 只有人脸
            similarity = face_sim
            method = f"face_only ({face_sim:.2f})"
            
        elif body_sim is not None:
            # 只有人体 (无人脸，如背身、目标背身保存)
            similarity = body_sim
            if 'color' in body_details:
                method = f"body_only ({body_sim:.2f} C:{body_details.get('color', 0):.2f})"
            else:
                method = f"body_only ({body_sim:.2f})"
            
        else:
            similarity = 0.0
            method = "no_feature"
        
        return similarity, method
    
    def is_same_target(
        self,
        target: FusedTargetFeature,
        candidate: FusedTargetFeature,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        判断是否同一目标
        
        Returns:
            (是否匹配, 相似度, 匹配方式)
        """
        similarity, method = self.compute_similarity(target, candidate)
        
        # 根据匹配方式选择阈值
        if threshold is not None:
            thresh = threshold
        elif "fused" in method:
            thresh = self.config.fused_threshold
        elif "face_only" in method:
            thresh = self.config.face_only_threshold
        elif "body_only" in method:
            thresh = self.config.body_only_threshold
        else:
            thresh = 0.5
        
        is_match = similarity >= thresh
        
        return is_match, similarity, method


# 测试代码
if __name__ == "__main__":
    config = FusedRecognizerConfig()
    recognizer = FusedTargetRecognizer(config)
    
    print("=" * 50)
    print("融合识别器配置")
    print("=" * 50)
    print(f"人脸权重: {config.face_weight}")
    print(f"人体权重: {config.body_weight}")
    print(f"人脸阈值: {config.face_only_threshold}")
    print(f"人体阈值: {config.body_only_threshold}")
    print(f"融合阈值: {config.fused_threshold}")
    print()
    
    # 模拟测试 - 自匹配
    target = FusedTargetFeature(
        face_embedding=np.random.randn(128).astype(np.float32),
        body_feature=np.random.randn(256).astype(np.float32),
        timestamp=time.time()
    )
    
    # 归一化
    target.face_embedding /= np.linalg.norm(target.face_embedding)
    target.body_feature /= np.linalg.norm(target.body_feature)
    
    # 自己匹配自己
    is_match, sim, method = recognizer.is_same_target(target, target)
    print(f"自匹配测试: match={is_match}, sim={sim:.3f}, method={method}")
