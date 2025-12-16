"""
YuNet 人脸检测器
YuNet Face Detector - 使用 PyTorch 模型
"""

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class YuNetFaceDetection:
    """YuNet 人脸检测结果"""
    bbox: np.ndarray           # [x1, y1, x2, y2]
    keypoints: np.ndarray      # [5, 2] - 5个关键点
    confidence: float          # 置信度


@dataclass 
class YuNetConfig:
    """YuNet 配置"""
    model_path: str = ""
    confidence_threshold: float = 0.7
    nms_threshold: float = 0.3
    top_k: int = 5000
    keep_top_k: int = 10  # 最多保留多少个人脸


class YuNetDetector:
    """YuNet 人脸检测器"""
    
    def __init__(self, config: YuNetConfig):
        self.config = config
        self.model = None
        self.device = torch.device('cpu')
        self._is_loaded = False
        
        # 检测辅助类
        self.priorbox = None
        self.cfg = None
    
    def load(self) -> bool:
        """加载模型"""
        try:
            # 导入 detection_recognition 的模型
            det_recog_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "detection_recognition"
            )
            sys.path.insert(0, det_recog_path)
            
            from model.yufacedetectnet import YuFaceDetectNet
            from model.detect_utils import cfg, PriorBox, nms, decode
            
            self.cfg = cfg
            self.PriorBox = PriorBox
            self.nms = nms
            self.decode = decode
            
            # 加载模型
            self.model = YuFaceDetectNet(phase='test', size=None)
            self.model = self._load_model(self.model, self.config.model_path)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            torch.set_grad_enabled(False)
            cudnn.benchmark = True
            
            self._is_loaded = True
            print(f"[INFO] YuNet 人脸检测器已加载: {self.config.model_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] YuNet 加载失败: {e}")
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
    
    def detect(self, image: np.ndarray) -> List[YuNetFaceDetection]:
        """
        检测人脸
        
        Args:
            image: BGR 图像
            
        Returns:
            人脸检测结果列表
        """
        if not self._is_loaded:
            return []
        
        img = np.float32(image)
        im_height, im_width = img.shape[:2]
        
        # 预处理
        img_tensor = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # 缩放因子
        scale = torch.Tensor([im_width, im_height, im_width, im_height,
                             im_width, im_height, im_width, im_height,
                             im_width, im_height, im_width, im_height,
                             im_width, im_height])
        scale = scale.to(self.device)
        
        # 推理
        loc, conf, iou = self.model(img_tensor)
        
        # 解码
        priorbox = self.PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)
        
        boxes = self.decode(loc.data.squeeze(0), priors.data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        
        # 计算分数
        cls_scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        iou_scores = iou.squeeze(0).data.cpu().numpy()[:, 0]
        iou_scores = np.clip(iou_scores, 0., 1.)
        scores = np.sqrt(cls_scores * iou_scores)
        
        # 过滤低分
        inds = np.where(scores > self.config.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        
        # Top-K
        order = scores.argsort()[::-1][:self.config.top_k]
        boxes = boxes[order]
        scores = scores[order]
        
        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        selected_idx = np.array([0, 1, 2, 3, 14])
        keep = self.nms(dets[:, selected_idx], self.config.nms_threshold)
        dets = dets[keep, :]
        
        # Keep top-K
        dets = dets[:self.config.keep_top_k, :]
        
        # 转换为结果
        results = []
        for det in dets:
            if det[2] - det[0] > 5 and det[3] - det[1] > 5:
                bbox = det[:4]
                # 关键点: [左眼, 右眼, 鼻子, 左嘴角, 右嘴角]
                keypoints = det[4:14].reshape(5, 2)
                confidence = det[14]
                
                results.append(YuNetFaceDetection(
                    bbox=bbox,
                    keypoints=keypoints,
                    confidence=confidence
                ))
        
        return results
    
    def release(self):
        """释放资源"""
        self.model = None
        self._is_loaded = False
