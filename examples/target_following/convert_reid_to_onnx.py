"""
将 MobileNetV2-ReID 模型导出为 ONNX 格式
Export MobileNetV2-ReID model to ONNX format

依赖: pip install torch
"""

import torch
import torch.nn as nn
from pathlib import Path
import os


# 简化版 MobileNetV2 结构
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2ReID(nn.Module):
    """
    MobileNetV2 用于 ReID 任务
    输出: 256D embedding
    """
    def __init__(self, embedding_dim: int = 256, width_mult: float = 1.0):
        super().__init__()
        
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        
        features = [ConvBNReLU(3, input_channel, stride=2)]
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ReID embedding 层
        self.embedding = nn.Sequential(
            nn.Linear(self.last_channel, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # x: [N, 3, 256, 128]
        x = self.features(x)   # [N, 1280, H, W]
        x = self.pool(x)       # [N, 1280, 1, 1]
        x = x.view(x.size(0), -1)  # [N, 1280]
        x = self.embedding(x)  # [N, 256]
        return x


def export_onnx():
    """导出 ONNX 模型"""
    model = MobileNetV2ReID(embedding_dim=256)
    model.eval()
    
    # 输入尺寸: [1, 3, 256, 128] (height=256, width=128)
    dummy_input = torch.randn(1, 3, 256, 128)
    
    # 输出路径
    save_path = Path(__file__).parent / "models" / "mobilenetv2_reid.onnx"
    save_path.parent.mkdir(exist_ok=True)
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        opset_version=11,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"}
        }
    )
    
    size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"[✓] 模型已导出: {save_path} ({size_mb:.1f} MB)")
    
    # 验证
    try:
        import onnxruntime as ort
        import numpy as np
        
        sess = ort.InferenceSession(str(save_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        
        test_input = np.random.randn(1, 3, 256, 128).astype(np.float32)
        output = sess.run(None, {input_name: test_input})[0]
        
        print(f"[✓] 验证通过: 输入 {test_input.shape} → 输出 {output.shape}")
        
    except Exception as e:
        print(f"[!] 验证跳过: {e}")


if __name__ == "__main__":
    export_onnx()
