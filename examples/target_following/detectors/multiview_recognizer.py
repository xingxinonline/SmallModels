"""
多视角目标识别器 - 支持正面/背面/侧面自动学习
Multi-View Target Recognizer

核心策略:
  1. 多视角特征库 - 自动积累不同角度的特征
  2. 运动一致性 - 短时间内用位置连续性维持跟踪
  3. 场景自适应 - 根据人脸可见性切换策略
  4. 时域平滑 - 多帧投票确认，避免闪烁
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import time
from collections import deque


@dataclass
class ViewFeature:
    """单视角特征"""
    has_face: bool = False
    face_embedding: Optional[np.ndarray] = None
    part_color_hists: Optional[np.ndarray] = None
    part_lbp_hists: Optional[np.ndarray] = None
    geometry: Optional[np.ndarray] = None
    timestamp: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        """有人脸embedding或人体颜色直方图，即为有效"""
        return self.part_color_hists is not None or (self.has_face and self.face_embedding is not None)
    
    @property
    def has_body(self) -> bool:
        """是否有人体特征"""
        return self.part_color_hists is not None


@dataclass
class MultiViewTarget:
    """多视角目标特征库"""
    # 视角特征列表 (最多保存 N 个不同角度)
    view_features: List[ViewFeature] = field(default_factory=list)
    max_views: int = 10  # 默认存储10个视角
    
    # 运动历史 (用于一致性检查)
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # 状态
    last_bbox: Optional[np.ndarray] = None
    last_seen_time: float = 0.0
    track_id: int = 0
    
    # 统计
    total_matches: int = 0
    face_matches: int = 0
    body_matches: int = 0
    
    def add_view(self, view: ViewFeature, min_interval: float = 1.0) -> Tuple[bool, str]:
        """
        添加新视角特征（智能替换策略）
        
        Args:
            view: 新视角特征
            min_interval: 最小时间间隔 (秒)，避免重复添加相似视角
        
        Returns:
            (是否成功添加, 操作描述)
        """
        if not view.is_valid:
            return False, "无效视角"
        
        # 检查时间间隔
        if self.view_features:
            last_time = self.view_features[-1].timestamp
            if view.timestamp - last_time < min_interval:
                return False, "间隔太短"
        
        # 检查是否与现有视角差异足够大
        is_different, most_similar_idx, most_similar_score = self._is_different_view_with_info(view)
        
        if is_different:
            view.timestamp = time.time()
            
            # 视角库未满，直接添加
            if len(self.view_features) < self.max_views:
                self.view_features.append(view)
                return True, f"新增(总数:{len(self.view_features)})"
            
            # 视角库已满，智能替换
            replace_idx = self._find_view_to_replace(view)
            if replace_idx is not None:
                old_view = self.view_features[replace_idx]
                old_has_face = old_view.has_face
                new_has_face = view.has_face
                self.view_features[replace_idx] = view
                return True, f"替换[{replace_idx}]({'有脸→无脸' if old_has_face and not new_has_face else '无脸→有脸' if not old_has_face and new_has_face else '同类型'})"
            else:
                return False, "无可替换"
        
        return False, f"太相似(idx={most_similar_idx},sim={most_similar_score:.2f})"
    
    def _find_view_to_replace(self, new_view: ViewFeature) -> Optional[int]:
        """
        找到最适合替换的视角索引
        
        替换策略优先级：
        1. 保护背面视角：至少保留2个无人脸视角（背面/侧面）
        2. 新视角有人脸时，替换与之最相似的有人脸视角（去除重复角度）
        3. 新视角无人脸时，替换与之最相似的无人脸视角
        4. 永远不替换索引0（初始锁定视角）
        
        Returns:
            要替换的视角索引，如果不应替换则返回None
        """
        if len(self.view_features) < 2:
            return None
        
        new_has_face = new_view.has_face
        
        # 统计各类型视角
        face_indices = [i for i, v in enumerate(self.view_features) if v.has_face and i > 0]
        body_only_indices = [i for i, v in enumerate(self.view_features) if not v.has_face and i > 0]
        
        # 保护背面视角：至少保留 MIN_BACK_VIEWS 个
        MIN_BACK_VIEWS = 2
        
        # 策略1：新视角有人脸
        if new_has_face:
            # 优先替换与新视角最相似的有人脸视角（去除重复角度）
            if face_indices:
                max_sim = -1
                max_idx = face_indices[0]
                for idx in face_indices:
                    sim = self._compute_view_similarity(self.view_features[idx], new_view)
                    if sim > max_sim:
                        max_sim = sim
                        max_idx = idx
                return max_idx
            
            # 如果没有有人脸视角可替换，检查是否可以替换无人脸视角
            # 保护条件：至少保留 MIN_BACK_VIEWS 个无人脸视角
            if body_only_indices and len(body_only_indices) > MIN_BACK_VIEWS:
                return body_only_indices[0]  # 替换最老的无人脸视角
        
        # 策略2：新视角无人脸（背面/侧面）
        else:
            # 优先替换无人脸视角中与新视角最相似的
            if body_only_indices:
                max_sim = -1
                max_idx = body_only_indices[0]
                for idx in body_only_indices:
                    sim = self._compute_view_similarity(self.view_features[idx], new_view)
                    if sim > max_sim:
                        max_sim = sim
                        max_idx = idx
                return max_idx
            
            # 如果只有有人脸视角，检查是否过多
            # 如果有人脸视角超过一半，可以替换最老的有人脸视角
            if len(face_indices) > self.max_views // 2:
                return face_indices[0]  # 替换最老的有人脸视角（保留初始视角）
        
        return None
    
    def _is_different_view_with_info(self, new_view: ViewFeature, threshold: float = 0.7) -> Tuple[bool, int, float]:
        """
        检查是否为不同视角，并返回最相似的视角信息
        
        Returns:
            (是否不同, 最相似视角索引, 最相似度)
        """
        if not self.view_features:
            return True, -1, 0.0
        
        max_sim = -1.0
        max_idx = 0
        
        for i, existing in enumerate(self.view_features):
            sim = self._compute_view_similarity(existing, new_view)
            if sim > max_sim:
                max_sim = sim
                max_idx = i
        
        is_different = max_sim < threshold
        return is_different, max_idx, max_sim
    
    def _is_different_view(self, new_view: ViewFeature, threshold: float = 0.7) -> bool:
        """检查是否为不同视角 (与所有已有视角相似度都低于阈值)"""
        if not self.view_features:
            return True
        
        for existing in self.view_features:
            sim = self._compute_view_similarity(existing, new_view)
            if sim > threshold:
                return False  # 太相似，不需要添加
        
        return True
    
    def _compute_view_similarity(self, v1: ViewFeature, v2: ViewFeature) -> float:
        """
        计算两个视角的相似度
        
        优先使用人脸特征比较（如果都有），否则使用人体颜色直方图
        """
        # 1. 如果都有人脸特征，使用人脸相似度
        if (v1.has_face and v1.face_embedding is not None and 
            v2.has_face and v2.face_embedding is not None):
            face_sim = float(np.dot(v1.face_embedding, v2.face_embedding))
            return face_sim
        
        # 2. 如果都有人体特征，使用颜色直方图相似度
        if v1.part_color_hists is not None and v2.part_color_hists is not None:
            sims = []
            num_parts = min(len(v1.part_color_hists), len(v2.part_color_hists))
            
            for i in range(num_parts):
                h1 = v1.part_color_hists[i].astype(np.float32)
                h2 = v2.part_color_hists[i].astype(np.float32)
                
                if h1.sum() < 1e-8 or h2.sum() < 1e-8:
                    sims.append(0.5)
                    continue
                
                bc = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
                sims.append(1.0 - bc)
            
            return float(np.mean(sims))
        
        # 3. 一个有人脸一个只有人体，视为不同视角
        return 0.0
    
    def update_position(self, bbox: np.ndarray) -> None:
        """更新位置历史"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.position_history.append((cx, cy, time.time()))
        self.last_bbox = bbox.copy()
        self.last_seen_time = time.time()
    
    @property
    def has_face_view(self) -> bool:
        """是否有包含人脸的视角"""
        return any(v.has_face for v in self.view_features)
    
    @property
    def num_views(self) -> int:
        return len(self.view_features)


@dataclass 
class MultiViewConfig:
    """多视角识别器配置"""
    # 特征权重 (动态调整，有高置信人脸时增加人脸权重)
    face_weight: float = 0.6
    body_weight: float = 0.4
    face_priority_threshold: float = 0.65  # 人脸相似度超过此值时，增加人脸权重
    face_priority_weight: float = 0.85  # 高置信人脸时的权重
    
    # 阈值
    face_threshold: float = 0.45
    body_threshold: float = 0.50  # 多视角可以降低一点
    fused_threshold: float = 0.48
    
    # 运动一致性
    motion_weight: float = 0.15  # 运动一致性权重
    max_motion_distance: float = 150  # 最大位移 (像素)
    motion_time_window: float = 0.5  # 时间窗口 (秒)
    
    # 多人场景保护
    require_face_body_consistency: bool = True  # 人脸-人体一致性检查
    multi_person_face_priority: bool = True  # 多人场景下优先信任人脸
    
    # 时域平滑
    smooth_window: int = 5  # 多帧投票窗口
    confirm_threshold: int = 3  # 确认需要的帧数
    
    # 自动学习
    auto_learn: bool = True
    learn_interval: float = 2.0  # 自动学习间隔 (秒)
    
    # 颜色权重 - 正面 (头部少)
    part_weights: List[float] = field(default_factory=lambda: [0.05, 0.15, 0.20, 0.20, 0.25, 0.15])
    # 颜色权重 - 背面 (增加下半身权重，因为背面上半身特征不明显)
    back_part_weights: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.25, 0.30, 0.15])


class MultiViewRecognizer:
    """多视角目标识别器"""
    
    def __init__(self, config: MultiViewConfig = None):
        self.config = config or MultiViewConfig()
        self.target: Optional[MultiViewTarget] = None
        
        # 时域平滑
        self.match_history = deque(maxlen=self.config.smooth_window)
        
        # 上次学习时间
        self.last_learn_time = 0.0
    
    def set_target(self, view: ViewFeature, bbox: np.ndarray) -> None:
        """设置目标"""
        self.target = MultiViewTarget()
        self.target.add_view(view)  # 初始视角，忽略返回值
        self.target.update_position(bbox)
        self.match_history.clear()
        self.last_learn_time = time.time()
    
    def clear_target(self) -> None:
        """清除目标"""
        self.target = None
        self.match_history.clear()
    
    def compute_similarity(
        self,
        candidate: ViewFeature,
        candidate_bbox: np.ndarray
    ) -> Tuple[float, str, dict]:
        """
        计算候选与目标的相似度
        
        Returns:
            (相似度, 匹配方式, 详细信息)
        """
        if self.target is None:
            return 0.0, "no_target", {}
        
        details = {}
        
        # 1. 与所有视角比较，取最大相似度
        best_body_sim = 0.0
        best_view_idx = -1
        
        for i, view in enumerate(self.target.view_features):
            sim = self._compute_body_similarity(view, candidate)
            if sim > best_body_sim:
                best_body_sim = sim
                best_view_idx = i
        
        details['body_sim'] = best_body_sim
        details['best_view'] = best_view_idx
        details['num_views'] = self.target.num_views
        
        # 2. 人脸相似度 (如果两边都有人脸)
        face_sim = None
        if candidate.has_face and candidate.face_embedding is not None:
            for view in self.target.view_features:
                if view.has_face and view.face_embedding is not None:
                    sim = float(np.dot(view.face_embedding, candidate.face_embedding))
                    if face_sim is None or sim > face_sim:
                        face_sim = sim
        
        details['face_sim'] = face_sim
        
        # 3. 运动一致性
        motion_sim = self._compute_motion_consistency(candidate_bbox)
        details['motion_sim'] = motion_sim
        
        # 4. 融合策略 - 动态权重调整
        if face_sim is not None and face_sim > 0.3:
            # 有人脸参与 - 动态调整权重
            if self.config.multi_person_face_priority and face_sim >= self.config.face_priority_threshold:
                # 高置信人脸时，大幅增加人脸权重
                fw = self.config.face_priority_weight  # 默认 0.85
                bw = 1.0 - fw - self.config.motion_weight
                method_prefix = "face_priority"
            else:
                fw = self.config.face_weight
                bw = self.config.body_weight
                method_prefix = "fused"
            
            base_sim = face_sim * fw + best_body_sim * bw
            similarity = base_sim + motion_sim * self.config.motion_weight
            method = f"{method_prefix} (F:{face_sim:.2f}*{fw:.2f} B:{best_body_sim:.2f}*{bw:.2f} M:{motion_sim:.2f})"
        else:
            # 纯人体 + 运动
            similarity = best_body_sim * (1 - self.config.motion_weight) + motion_sim * self.config.motion_weight
            method = f"body+motion (B:{best_body_sim:.2f} M:{motion_sim:.2f} V:{best_view_idx})"
        
        similarity = min(1.0, max(0.0, similarity))
        
        return similarity, method, details
    
    def _compute_body_similarity(self, target_view: ViewFeature, candidate: ViewFeature) -> float:
        """计算人体特征相似度"""
        if target_view.part_color_hists is None or candidate.part_color_hists is None:
            return 0.0
        
        num_parts = min(len(target_view.part_color_hists), len(candidate.part_color_hists))
        
        # 颜色相似度
        color_sims = []
        for i in range(num_parts):
            h1 = target_view.part_color_hists[i].astype(np.float32)
            h2 = candidate.part_color_hists[i].astype(np.float32)
            
            if h1.sum() < 1e-8 or h2.sum() < 1e-8:
                color_sims.append(0.5)
                continue
            
            bc = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
            color_sims.append(1.0 - bc)
        
        # 根据视角选择权重：背面（无人脸）使用 back_part_weights
        is_back_view = not target_view.has_face or not candidate.has_face
        if is_back_view and hasattr(self.config, 'back_part_weights'):
            weights = self.config.back_part_weights[:num_parts]
        else:
            weights = self.config.part_weights[:num_parts]
        weights = np.array(weights) / sum(weights)
        color_sim = float(np.dot(color_sims, weights))
        
        # LBP 纹理
        texture_sim = 0.5
        if target_view.part_lbp_hists is not None and candidate.part_lbp_hists is not None:
            lbp_sims = []
            num_lbp = min(len(target_view.part_lbp_hists), len(candidate.part_lbp_hists))
            
            for i in range(num_lbp):
                h1 = target_view.part_lbp_hists[i].astype(np.float32)
                h2 = candidate.part_lbp_hists[i].astype(np.float32)
                
                if h1.sum() < 1e-8 or h2.sum() < 1e-8:
                    lbp_sims.append(0.5)
                    continue
                
                sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                lbp_sims.append((sim + 1) / 2)
            
            texture_sim = float(np.mean(lbp_sims))
        
        # 几何
        geo_sim = 0.8
        if target_view.geometry is not None and candidate.geometry is not None:
            diff = np.abs(target_view.geometry[:2] - candidate.geometry[:2])
            geo_sim = 1.0 - np.clip(np.mean(diff) * 2, 0, 1)
        
        # 融合: 颜色(0.5) + 纹理(0.3) + 几何(0.2)
        return color_sim * 0.5 + texture_sim * 0.3 + geo_sim * 0.2
    
    def _compute_motion_consistency(self, bbox: np.ndarray) -> float:
        """
        计算运动一致性
        
        如果候选位置与目标的运动轨迹一致，给予额外加分
        """
        if self.target is None or len(self.target.position_history) < 2:
            return 0.5  # 无历史，中等分数
        
        current_time = time.time()
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # 获取最近的位置
        recent_positions = [(x, y, t) for x, y, t in self.target.position_history 
                           if current_time - t < self.config.motion_time_window]
        
        if not recent_positions:
            # 太久没看到，用最后位置
            if self.target.last_bbox is not None:
                last_cx = (self.target.last_bbox[0] + self.target.last_bbox[2]) / 2
                last_cy = (self.target.last_bbox[1] + self.target.last_bbox[3]) / 2
                distance = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                
                # 时间越长，允许的距离越大
                time_since_last = current_time - self.target.last_seen_time
                max_dist = self.config.max_motion_distance * (1 + time_since_last)
                
                return max(0, 1 - distance / max_dist)
            return 0.5
        
        # 预测位置 (简单线性外推)
        if len(recent_positions) >= 2:
            x1, y1, t1 = recent_positions[-2]
            x2, y2, t2 = recent_positions[-1]
            dt = t2 - t1
            
            if dt > 0.01:
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                
                predict_dt = current_time - t2
                pred_x = x2 + vx * predict_dt
                pred_y = y2 + vy * predict_dt
                
                distance = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
                return max(0, 1 - distance / self.config.max_motion_distance)
        
        # 只有一个点，用距离
        x, y, t = recent_positions[-1]
        distance = np.sqrt((cx - x)**2 + (cy - y)**2)
        return max(0, 1 - distance / self.config.max_motion_distance)
    
    def is_same_target(
        self,
        candidate: ViewFeature,
        candidate_bbox: np.ndarray,
        update_history: bool = False,  # 是否更新时域平滑历史
        return_details: bool = False  # 是否返回详细信息
    ) -> Tuple[bool, float, str] | Tuple[bool, float, str, dict]:
        """
        判断是否为同一目标 (可选时域平滑)
        
        Args:
            candidate: 候选特征
            candidate_bbox: 候选边界框
            update_history: 是否更新匹配历史（只有最终选定的匹配才应该更新）
            return_details: 是否返回详细信息（包含 face_sim, body_sim 等）
        
        Returns:
            (is_match, similarity, method) 或
            (is_match, similarity, method, details) 如果 return_details=True
        """
        similarity, method, details = self.compute_similarity(candidate, candidate_bbox)
        
        # 选择阈值 - face_priority 也算融合匹配
        if "fused" in method or "face_priority" in method:
            threshold = self.config.fused_threshold
        elif details.get('face_sim') is not None:
            threshold = self.config.face_threshold
        else:
            threshold = self.config.body_threshold
        
        is_match = similarity >= threshold
        
        # 时域平滑 (多帧投票) - 只有确认匹配时才更新历史
        if update_history:
            self.match_history.append(1 if is_match else 0)
            
            if len(self.match_history) >= self.config.smooth_window:
                vote_count = sum(self.match_history)
                smoothed_match = vote_count >= self.config.confirm_threshold
            else:
                smoothed_match = is_match
            
            if return_details:
                return smoothed_match, similarity, method, details
            return smoothed_match, similarity, method
        else:
            # 不更新历史，直接返回原始结果
            if return_details:
                return is_match, similarity, method, details
            return is_match, similarity, method
    
    def clear_match_history(self):
        """清空匹配历史（目标切换或丢失时调用）"""
        self.match_history.clear()
    
    def auto_learn(
        self,
        candidate: ViewFeature,
        candidate_bbox: np.ndarray,
        is_match: bool
    ) -> Tuple[bool, str]:
        """
        自动学习新视角
        
        如果匹配成功且距离上次学习超过间隔，尝试添加新视角
        
        Returns:
            (是否学习成功, 操作描述)
        """
        if not self.config.auto_learn:
            return False, "自动学习已禁用"
        
        if not is_match or self.target is None:
            return False, "未匹配或无目标"
        
        current_time = time.time()
        if current_time - self.last_learn_time < self.config.learn_interval:
            return False, "间隔太短"
        
        # 尝试添加新视角
        added, reason = self.target.add_view(candidate, min_interval=self.config.learn_interval)
        if added:
            self.last_learn_time = current_time
            return True, reason
        
        return False, reason
    
    def update_tracking(self, bbox: np.ndarray) -> None:
        """更新跟踪状态"""
        if self.target is not None:
            self.target.update_position(bbox)


# 测试代码
if __name__ == "__main__":
    config = MultiViewConfig()
    recognizer = MultiViewRecognizer(config)
    
    print("=" * 50)
    print("多视角识别器配置")
    print("=" * 50)
    print(f"最大视角数: {MultiViewTarget().max_views}")
    print(f"运动权重: {config.motion_weight}")
    print(f"自动学习: {config.auto_learn}")
    print(f"学习间隔: {config.learn_interval}s")
    print(f"时域平滑: {config.smooth_window}帧, 确认{config.confirm_threshold}帧")
