"""
目标跟踪器
Target Tracker
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TrackerConfig


@dataclass
class TrackingResult:
    """跟踪结果"""
    found: bool                      # 是否找到目标
    bbox: Optional[np.ndarray]       # 目标边界框 [x1, y1, x2, y2]
    confidence: float                # 置信度
    match_type: str                  # 匹配类型 ("face", "person", "both")


class TargetTracker:
    """目标跟踪器"""
    
    def __init__(self, config: TrackerConfig):
        self.config = config
        
        # 目标特征
        self.target_face_feature = None
        self.target_person_feature = None
        self.target_last_bbox = None         # 最后的人脸/跟踪框
        self.target_last_person_bbox = None  # 最后的人体框 (用于转身后跟踪)
    
    def set_target(
        self, 
        bbox=None,
        face_feature=None, 
        person_feature=None,
        person_bbox=None
    ):
        """设置跟踪目标"""
        self.target_face_feature = face_feature
        self.target_person_feature = person_feature
        self.target_last_bbox = bbox
        self.target_last_person_bbox = person_bbox  # 保存人体框
        self.last_bbox = bbox
    
    def track(
        self,
        face_detections: List,
        face_features: dict,
        person_detections: List,
        face_recognizer=None,
        require_feature_match: bool = False
    ) -> TrackingResult:
        """
        在当前帧中跟踪目标 (人脸优先策略)
        
        优先级:
        1. 人脸特征匹配 (最可靠)
        2. 人脸 IoU 匹配 (当有人脸检测但无特征时) - 可禁用
        3. 人体 IoU + 特征匹配 (当没有人脸时) - 可禁用
        
        Args:
            face_detections: 人脸检测结果列表
            face_features: 对应的人脸特征字典 {index: feature}
            person_detections: 人体检测结果列表
            face_recognizer: 人脸识别器 (用于计算相似度)
            require_feature_match: 是否必须使用特征匹配 (用于 LOST_TARGET 状态防止误匹配)
            
        Returns:
            跟踪结果
        """
        # 策略1: 人脸特征匹配 (最高优先级)
        if self.target_face_feature is not None and face_recognizer is not None and face_features and face_detections:
            best_face_idx = -1
            best_similarity = 0
            
            for i, feat in face_features.items():
                if feat is None:
                    continue
                # 确保索引有效
                if i >= len(face_detections):
                    continue
                
                similarity = face_recognizer.compute_similarity(
                    self.target_face_feature, feat
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_face_idx = i
            
            # 找到匹配的人脸
            threshold = face_recognizer.config.similarity_threshold
            if (best_similarity > threshold and 
                best_face_idx >= 0 and best_face_idx < len(face_detections)):
                face_bbox = face_detections[best_face_idx].bbox
                
                # 尝试找到对应的人体框
                person_match = self._find_person_for_face(face_bbox, person_detections)
                
                if person_match is not None:
                    # 有人体框，使用人体框作为目标
                    self.target_last_bbox = person_match.bbox.copy()
                    self.target_last_person_bbox = person_match.bbox.copy()  # 更新人体框
                    if person_match.keypoints is not None:
                        from detectors.person_detector import PersonDetector
                        self.target_person_feature = PersonDetector.compute_person_feature(
                            person_match.keypoints
                        )
                    return TrackingResult(
                        found=True,
                        bbox=person_match.bbox,
                        confidence=best_similarity,
                        match_type="face+person"
                    )
                else:
                    # 没有人体框，使用人脸框
                    self.target_last_bbox = face_bbox.copy()
                    return TrackingResult(
                        found=True,
                        bbox=face_bbox,
                        confidence=best_similarity,
                        match_type="face"
                    )
        
        # 策略1.5: 人体特征匹配 (当没有人脸但有人体检测时)
        # 即使 require_feature_match=True，人体特征也是有效的特征匹配
        if self.target_person_feature is not None and person_detections:
            best_person = None
            best_person_score = 0
            
            for det in person_detections:
                if det.keypoints is None:
                    continue
                
                from detectors.person_detector import PersonDetector
                person_feat = PersonDetector.compute_person_feature(det.keypoints)
                feat_sim = self._cosine_similarity(self.target_person_feature, person_feat)
                
                # 人体特征匹配阈值 (需要较高相似度)
                if feat_sim > 0.7 and feat_sim > best_person_score:
                    best_person_score = feat_sim
                    best_person = det
            
            if best_person is not None:
                self.target_last_bbox = best_person.bbox.copy()
                self.target_last_person_bbox = best_person.bbox.copy()
                # 更新人体特征
                from detectors.person_detector import PersonDetector
                self.target_person_feature = PersonDetector.compute_person_feature(best_person.keypoints)
                return TrackingResult(
                    found=True,
                    bbox=best_person.bbox,
                    confidence=best_person_score,
                    match_type="person_feature"
                )
        
        # 如果要求必须特征匹配，则不使用 IoU 回退策略
        if require_feature_match:
            return TrackingResult(
                found=False,
                bbox=self.target_last_bbox,
                confidence=0.0,
                match_type="none"
            )
        
        # 策略2: 人脸 IoU 匹配 (当没有特征但有人脸检测时) - 仅在 TRACKING 状态使用
        if face_detections and self.target_last_bbox is not None:
            best_iou = 0
            best_face_idx = -1
            
            for i, face in enumerate(face_detections):
                iou = self._compute_iou(face.bbox, self.target_last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_face_idx = i
            
            if best_iou > self.config.iou_threshold and best_face_idx >= 0:
                face_bbox = face_detections[best_face_idx].bbox
                person_match = self._find_person_for_face(face_bbox, person_detections)
                
                if person_match is not None:
                    self.target_last_bbox = person_match.bbox.copy()
                    self.target_last_person_bbox = person_match.bbox.copy()  # 更新人体框
                    return TrackingResult(
                        found=True,
                        bbox=person_match.bbox,
                        confidence=best_iou,
                        match_type="face_iou+person"
                    )
                else:
                    self.target_last_bbox = face_bbox.copy()
                    return TrackingResult(
                        found=True,
                        bbox=face_bbox,
                        confidence=best_iou,
                        match_type="face_iou"
                    )
        
        # 策略3: 人体 IoU + 特征匹配 (当没有人脸时)
        # 使用人体框做 IoU 比较，解决转身后跟踪丢失的问题
        reference_bbox = self.target_last_person_bbox if self.target_last_person_bbox is not None else self.target_last_bbox
        
        if person_detections and reference_bbox is not None:
            best_score = 0
            best_person = None
            
            for det in person_detections:
                # IoU 匹配 - 使用人体框与人体框比较
                iou = self._compute_iou(det.bbox, reference_bbox)
                
                # 降低 IoU 阈值要求，因为人可能移动
                person_iou_threshold = max(0.1, self.config.iou_threshold * 0.5)
                
                if iou > person_iou_threshold:
                    score = iou * self.config.iou_weight
                    
                    # 姿态特征匹配
                    if self.target_person_feature is not None and det.keypoints is not None:
                        from detectors.person_detector import PersonDetector
                        person_feat = PersonDetector.compute_person_feature(det.keypoints)
                        feat_sim = self._cosine_similarity(
                            self.target_person_feature, person_feat
                        )
                        score += feat_sim * self.config.feature_weight
                    
                    if score > best_score:
                        best_score = score
                        best_person = det
            
            if best_person is not None:
                self.target_last_bbox = best_person.bbox.copy()
                self.target_last_person_bbox = best_person.bbox.copy()  # 更新人体框
                if best_person.keypoints is not None:
                    from detectors.person_detector import PersonDetector
                    self.target_person_feature = PersonDetector.compute_person_feature(
                        best_person.keypoints
                    )
                return TrackingResult(
                    found=True,
                    bbox=best_person.bbox,
                    confidence=best_score,
                    match_type="person"
                )
        
        # 没有找到目标
        return TrackingResult(
            found=False,
            bbox=self.target_last_bbox,  # 返回最后已知位置
            confidence=0.0,
            match_type="none"
        )
    
    def _find_person_for_face(
        self, 
        face_bbox: np.ndarray, 
        person_detections: List
    ):
        """找到包含该人脸的人体"""
        face_center = [
            (face_bbox[0] + face_bbox[2]) / 2,
            (face_bbox[1] + face_bbox[3]) / 2
        ]
        
        best_person = None
        best_overlap = 0
        
        for det in person_detections:
            # 检查人脸中心是否在人体框内
            x1, y1, x2, y2 = det.bbox
            if x1 <= face_center[0] <= x2 and y1 <= face_center[1] <= y2:
                # 计算重叠程度
                overlap = self._compute_iou(face_bbox, det.bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_person = det
        
        return best_person
    
    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """计算 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    @staticmethod
    def _cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return float(np.dot(feat1, feat2) / (norm1 * norm2))
    
    def clear(self):
        """清除目标"""
        self.target_face_feature = None
        self.target_person_feature = None
        self.target_last_bbox = None
