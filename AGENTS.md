# S300 端侧 AI 网络推荐列表 (AGENTS.md)

本文档汇总了适合 S300 芯片部署的开源模型、对应任务、源码位置以及优化方式。

## 🧠 S300 NPU 规格详情 (NPU Specifications)

在选择和优化模型前，需严格遵循 S300 NPU 的硬件限制：

*   **工作模式**:
    *   **NPU 模式**: 典型的神经网络加速功能。
    *   **Memory 模式**: 将 NPU 内存释放给系统作为通用存储使用。
*   **计算模式**: 支持 **8x8** 和 **16x8** 两种模式。
*   **精度支持**:
    *   支持 8-bit 有符号/无符号计算。
    *   支持 8-bit 与 16-bit 混合计算（如权重 8-bit，特征图 16-bit）。
*   **算子支持 (Operators)**:
    *   **Conv2D / Depthwise**: Kernel <= 7x7, Stride 1~3, Padding 1~(k-1).
    *   **Pooling**: Max/Avg, Kernel 1~15.
    *   **Activation**: ReLU, P-ReLU, Leaky-ReLU, ReLU6, Softmax (非 reduction).
    *   **Other**: LSTM, Concat, Element-wise (add/sub/mul...), Upsampling (bilinear/nearest).
*   **支持网络架构**: MobileNet, ResNet, YOLO, DCCRN, UNet, SqueezeNet, ShuffleNet, EfficientNet 等。

## 🔌 S300 DSP 规格详情 (CEVA SensPro 250)

S300 集成了 **CEVA SensPro 250 (SP250)**，这是一款专为传感器融合和 AI 设计的高性能 DSP IP。

*   **核心架构**:
    *   **标量单元**: 集成 **CEVA-BX2** 标量 DSP (4.3 CoreMark/MHz)，负责复杂的控制逻辑和非向量化算法。
    *   **向量单元**: 256 个 8x8 MACs (乘加器)，支持 8-bit 和 16-bit 并行处理。
*   **关键特性**:
    *   **AI 加速**: 支持 CDNN (CEVA Deep Neural Network) 库，可分担部分轻量级 AI 推理任务。
    *   **音频/语音**: 硬件级支持 **ClearVox** (降噪) 和 **WhisPro** (语音识别) 算法库。
    *   **传感器融合**: 支持 **MotionEngine**，用于 IMU、磁力计等传感器的数据融合 (6/9轴融合)。
    *   **CV 能力**: 支持 OpenVX 和 CEVA-CV 库，加速传统计算机视觉算法 (如 SLAM 前端特征提取)。
*   **适用场景**: 
    *   音频前处理 (AEC/AGC/VAD)。
    *   传感器数据融合 (IMU/GPS)。
    *   轻量级 CV 算子 (Resize, Color Convert, Feature Extraction)。

## 🕹️ S300 MCU 规格详情 (Cortex-M4 & Peripherals)

S300 内置了一颗 **ARM Cortex-M4** 处理器作为主控核心，负责系统调度、外设控制以及轻量级业务逻辑。

*   **核心参数**:
    *   **架构**: ARM Cortex-M4 (带 FPU)，支持浮点运算。
    *   **主频**: 典型工作频率 **192MHz** (CPTCLK)，AHB 总线频率可达 384MHz。
    *   **存储**:
        *   **ROM**: 10KB (启动)。
        *   **SRAM**: 控制域 392KB (8+384) + 计算域 512KB (128+128+256) + NPU 专用 516KB。
        *   **Flash/PSRAM**: 支持 XIP (Execute in Place)，最大支持 200MHz DDR Octal SPI。
*   **丰富外设 (Peripherals)**:
    *   **视觉接口**:
        *   **DVP**: 1路，支持 8/10-bit，最大分辨率 2592*1444 (5MP)，支持 YUV/RGB。
    *   **音频接口**:
        *   **I2S**: 2路，支持 Philips 标准，全双工，8K-48K 采样率。
        *   **PDM**: 2路，支持数字麦克风阵列，集成语音唤醒检测 (VAD)。
    *   **通讯接口**:
        *   **GMAC**: 1路千兆以太网 MAC，支持 RGMII/RMII。
        *   **SD/MMC**: 2路，支持 SD3.0/eMMC4.5，可接 WiFi 模组或大容量存储。
        *   **UART**: 4路，支持 RS232/RS485/IrDA，最高波特率可配置。
        *   **SPI**: 2路 (Master/Slave)，支持 DMA。
        *   **I2C**: 2路 (Master/Slave)，支持高速模式 (3.4Mbps)。
    *   **控制接口**:
        *   **PWM**: 3路独立 PWM，用于电机控制或调光。
        *   **GPIO**: 40个 (32 Multimedia + 8 AON)，支持中断唤醒。

---

## 💡 核心架构分析：WebRTC 3A vs S300 Native

关于 **WebRTC 3A (AEC/ANS/AGC)** 是否适合 S300，我的结论是：**不推荐直接移植作为主力方案，但强烈建议作为“效果基线 (Golden Baseline)”进行对比。**

### 1. 为什么 WebRTC 3A 不适合直接跑在 S300 上？
*   **算力浪费**: WebRTC 的 NS (Noise Suppression) 是基于传统信号处理（维纳滤波等）的，主要运行在 CPU/DSP 上，**完全无法利用 S300 强大的 NPU**。
*   **浮点依赖**: WebRTC 原生代码大量依赖浮点运算 (Float32)，而 S300 的优势在于 Int8/Int16 定点计算。强行移植会导致 CPU 负载过高。
*   **效果瓶颈**: 传统 NS 对非稳态噪声（如键盘声、瞬态噪音）处理能力远不如 AI 模型。

### 2. WebRTC 3A 的价值：作为“效果基线”
尽管不适合作为最终产品方案，但移植 WebRTC 3A (特别是定点化版本) 具有重要意义：
*   **基准测试 (Benchmark)**: 用 WebRTC 的处理效果作为“及格线”。如果 NPU 跑出来的 DCCRN 效果还不如 WebRTC，说明模型训练或量化有问题。
*   **兜底方案 (Fallback)**: 在 NPU 算力被视觉任务占满时，可以动态切换回 CPU 跑 WebRTC 3A，保证基本的音频通话功能。

### 3. S300 的“原生”音频架构建议 (Native Audio Pipeline)
既然 S300 规格书中明确支持 **DCCRN** 和 **LSTM**，我们应该构建一套利用 NPU 的现代音频链路：

| 模块               | 传统方案 (WebRTC)        | **S300 推荐方案**            | 理由                                                                                                                                            |
| :----------------- | :----------------------- | :--------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| **AEC (回声消除)** | WebRTC AEC3 / AEC Mobile | **CEVA ClearVox / SpeexDSP** | S300 的 DSP (SensPro 250) 原生支持 **ClearVox** 算法库，这是针对该硬件极致优化的商业级方案，效果和效率远超开源的 SpeexDSP。                     |
| **NS (降噪)**      | WebRTC NS (Transient)    | **DCCRN / CRN (NPU)**        | **这是核心差异点**。S300 NPU 支持复数卷积 (DCCRN)，这是目前端侧降噪的 SOTA 方案。利用 NPU 跑 Int8 量化的 DCCRN，功耗更低，降噪效果碾压 WebRTC。 |
| **AGC (增益控制)** | WebRTC AGC               | **WebRTC AGC (定点版)**      | AGC 计算量很小，可以直接复用 WebRTC 的定点版本 (AGC1)，运行在 CPU/DSP 上即可。                                                                  |

---

## 📋 S300 可部署的开源模型完整表格

| 任务类别             | 推荐模型 (Safe Choice)  | 进阶模型 (Pro Choice)    | 源码/来源                                                                               | S300 适配理由                                                                         |
| :------------------- | :---------------------- | :----------------------- | :-------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| **AI 语音降噪 (NS)** | **DCCRN** (Int8)        | **DeepFilterNet** (Lite) | [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)                              | DCCRN 适合 NPU 卷积加速；DeepFilterNet 效果更好但需 DSP 配合处理复杂 GRU。            |
| **关键词唤醒 (KWS)** | **DS-CNN**              | **BC-ResNet**            | [BC-ResNet](https://github.com/Qualcomm-AI-research/bc_resnet)                          | BC-ResNet 是目前 KWS 的 SOTA，参数更少，抗噪更强，完美契合 NPU 广播机制。             |
| **声纹识别**         | **ECAPA-TDNN**          | **RawNet3** (Pruned)     | [RawNet](https://github.com/Jungjee/RawNet)                                             | RawNet 直接处理原始波形，省去 STFT 预处理，适合端到端 NPU 加速。                      |
| **人脸检测**         | **YOLOv5-Face**         | **SCRFD** (0.5g)         | [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)         | SCRFD 专为边缘计算设计，计算量仅为 YOLO 的 1/3，且精度更高。                          |
| **人脸识别**         | **MobileFaceNet**       | **MagFace** (Nano)       | [MagFace](https://github.com/IrvingMeng/MagFace)                                        | MagFace 在识别同时能输出图像质量评分，非常适合门禁等非受控场景。                      |
| **通用目标检测**     | **YOLOv5-Nano**         | **NanoDet-Plus**         | [NanoDet-Plus](https://github.com/RangiLyu/nanodet)                                     | NanoDet-Plus 使用 ShuffleNetV2 骨干，无 Anchor 设计，速度比 YOLOv5-Nano 快 50% 以上。 |
| **人体检测**         | **SSDLite**             | **YOLOv8-Nano** (Export) | [Ultralytics](https://github.com/ultralytics/ultralytics)                               | YOLOv8 精度极高，但需注意算子支持（如 SiLU 转 ReLU），适合追求极致精度的场景。        |
| **表情识别**         | **Mini-Xception**       | **DAN** (Disturb Label)  | [DAN](https://github.com/yaoyao-liu/class-balanced-loss)                                | DAN 对遮挡和姿态变化鲁棒性更强，适合真实互动场景。                                    |
| **多模态语音+视频**  | **AVSR**                | **AutoAVSR**             | [AutoAVSR](https://github.com/mpc001/AutoAVSR)                                          | 自动搜索出的架构比手动设计的 AVSR 更轻量，适合异构计算。                              |
| **语音+手势交互**    | **GesRec**              | **TSM** (Temporal Shift) | [TSM](https://github.com/mit-han-lab/temporal-shift-module)                             | TSM 通过“时序移位”用 2D 卷积实现 3D 效果，零参数量增加，完美适配 S300 NPU。           |  | **姿态估计** | **MoveNet** (Lightning) | **PP-TinyPose** | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | 识别人体骨骼点，用于健身、手势控制等场景。PP-TinyPose 对国产 NPU 支持更好。 |
| **图像分类**         | **MobileNetV3** (Small) | **EfficientNet-Lite**    | [TensorFlow Models](https://github.com/tensorflow/models)                               | 通用分类 Backbone，适合垃圾分类、病虫害识别等场景。需 INT8 量化。                     |
| **目标跟踪**         | **NanoTrack**           | **LightTrack**           | [LightTrack](https://github.com/researchmm/LightTrack)                                  | Siamese 跟踪网络，LightTrack 通过 NAS 搜索，更轻量。适合无人机跟随。                  |
| **异常检测**         | **AutoEncoder**         | **Micro-TCN**            | 通用/自研                                                                               | AE 学习正常模式，偏离即为异常；TCN 比 LSTM 更适合 NPU 并行。                          |
| **时序分类**         | **1D-CNN**              | **InceptionTime**        | [tsai](https://github.com/timeseriesAI/tsai)                                            | 处理 IMU/振动/雷达点云等一维时序数据，识别跌倒、电机故障等。                          |
| **医疗音频**         | **1D-CNN**              | **Wav2Vec2** (Tiny)      | [HuggingFace](https://huggingface.co/facebook/wav2vec2-base)                            | 心肺音分类。Wav2Vec 需大幅裁剪；1D-CNN 更实用。                                       |
| **单目深度估计**     | **FastDepth**           | **Pydnet**               | [mobilePydnet](https://github.com/FilippoAleotti/mobilePydnet)                          | Pydnet 金字塔结构适合 NPU 并行计算，用于无人机/机器人避障。                           |
| **车道检测**         | **PilotNet**            | **UFLD**                 | [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)          | UFLD 将分割转为行分类，速度极快；PilotNet 端到端驾驶经典方案。                        |
| **文字识别 (OCR)**   | **CRNN**                | **PP-OCRv4** (Mobile)    | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)                                  | PP-OCRv4 是端侧 OCR 事实标准，支持检测+识别+方向分类。                                |
| **人像分割**         | **PP-HumanSeg** (Lite)  | **BiSeNet V2** (Pruned)  | [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)                                  | PP-HumanSeg 超轻量，适合背景虚化；BiSeNet 双边结构速度极快。                          |
| **活体检测**         | **MiniFASNet**          | **Silent-Face**          | [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | 防止照片/视频攻击，配合人脸识别使用，单模型 <1MB。                                    |
| **声音事件检测**     | **YAMNet** (Lite)       | **AudioSet-MobileNet**   | [TensorFlow Models](https://github.com/tensorflow/models/tree/master/research/audioset) | 识别婴儿哭声、玻璃破碎、狗叫等 500+ 类声音事件。                                      |

---

## 📊 性能基准测试 (Performance Benchmarking)

为了客观评估 S300 的算力水平（CPU vs NPU），建议移植以下开源基准测试工具：

| 测试维度         | 工具/项目           | GitHub 链接                                                         | 作用与意义                                                                                                                             |
| :--------------- | :------------------ | :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------- |
| **CPU 核心性能** | **CoreMark**        | [eembc/coremark](https://github.com/eembc/coremark)                 | **行业标准**。测试 MCU/CPU 的整数运算性能，用于评估非 AI 任务（如逻辑控制、协议栈）的处理能力。                                        |
| **AI 推理基准**  | **MLPerf Tiny**     | [mlcommons/tiny](https://github.com/mlcommons/tiny)                 | **TinyML 权威基准**。包含 KWS、VWW (Visual Wake Words)、AD (异常检测) 等标准负载，用于横向对比 S300 与 STM32、ESP32 等竞品的 AI 效率。 |
| **内存带宽**     | **tinymembench**    | [ssvb/tinymembench](https://github.com/ssvb/tinymembench)           | 测试 RAM/Flash 的读写带宽和延迟。**这对 NPU 极其重要**，因为图像/音频数据的搬运往往是推理速度的瓶颈。                                  |
| **DSP 性能**     | **CMSIS-DSP Tests** | [ARM-software/CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) | 如果 S300 使用 ARM 核或兼容 DSP 指令，可运行其自带的测试集，评估 FFT、矩阵运算等信号处理能力。                                         |
| **功耗测量**     | **INA226 + 脚本**   | 硬件模块 + Python 脚本                                              | 使用 INA226 电流传感器实时采集功耗，结合推理任务绘制功耗曲线，评估 TOPS/W 效率。                                                       |
| **热稳定性**     | **压力测试脚本**    | 自研                                                                | 连续运行推理任务 1-4 小时，监控芯片温度和降频情况，评估散热设计是否满足要求。                                                          |

---

## 🧪 AI 模型功能测试 (Model Validation)

模型部署后需进行严格的功能验证，确保量化和转换未导致精度损失：

### 1. 模型精度评估工具

| 测试类型         | 工具/方法                 | GitHub 链接 / 说明                                             | 适用场景                              |
| :--------------- | :------------------------ | :------------------------------------------------------------- | :------------------------------------ |
| **分类精度**     | **Top-1/Top-5 Accuracy**  | 使用标准数据集 (ImageNet, CIFAR) 计算                          | 图像分类 (MobileNetV3, EfficientNet)  |
| **检测精度**     | **mAP (COCO Eval)**       | [pycocotools](https://github.com/cocodataset/cocoapi)          | 目标检测 (NanoDet, SCRFD, YOLOv5)     |
| **分割精度**     | **mIoU / Dice Score**     | [PaddleSeg Eval](https://github.com/PaddlePaddle/PaddleSeg)    | 人像分割 (PP-HumanSeg, BiSeNet)       |
| **OCR 精度**     | **CER / WER**             | 字符/词错误率                                                  | 文字识别 (PP-OCRv4, CRNN)             |
| **KWS 精度**     | **FAR / FRR**             | 错误接受率 / 错误拒绝率                                        | 关键词唤醒 (BC-ResNet, DS-CNN)        |
| **人脸识别精度** | **TAR@FAR / AUC**         | [InsightFace Eval](https://github.com/deepinsight/insightface) | 人脸识别 (MobileFaceNet, MagFace)     |
| **量化对比**     | **FP32 vs INT8 精度对比** | [ONNX Runtime Quantization](https://onnxruntime.ai/)           | 对比量化前后精度下降幅度，一般应 < 1% |

### 2. 开源测试数据集推荐

| 任务类型     | 数据集                     | 规模          | 下载链接                                                        |
| :----------- | :------------------------- | :------------ | :-------------------------------------------------------------- |
| **图像分类** | **ImageNet-1K (Mini)**     | 1000 类/50K   | [Kaggle](https://www.kaggle.com/c/imagenet-object-localization) |
| **目标检测** | **COCO 2017 Val**          | 80 类/5K      | [cocodataset.org](https://cocodataset.org/#download)            |
| **人脸检测** | **WIDER FACE**             | 32K 图/393K脸 | [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)                 |
| **人脸识别** | **LFW (Labeled Faces)**    | 5749 人/13K   | [LFW](http://vis-www.cs.umass.edu/lfw/)                         |
| **KWS**      | **Google Speech Commands** | 35 词/105K    | [TensorFlow Datasets](https://www.tensorflow.org/datasets)      |
| **声音事件** | **ESC-50**                 | 50 类/2K      | [ESC-50](https://github.com/karolpiczak/ESC-50)                 |
| **OCR**      | **ICDAR 2015/2019**        | 场景文字      | [ICDAR](https://rrc.cvc.uab.es/)                                |
| **姿态估计** | **COCO Keypoints**         | 17 关键点     | [cocodataset.org](https://cocodataset.org/#keypoints-2017)      |

---

## ⏱️ AI 模型性能测试 (Inference Profiling)

除了功能正确性，推理性能直接决定产品体验：

### 1. 性能指标定义

| 指标               | 定义                         | 目标参考值 (S300)                         |
| :----------------- | :--------------------------- | :---------------------------------------- |
| **Latency (延迟)** | 单次推理耗时 (ms)            | KWS < 30ms, 人脸检测 < 100ms, 分割 < 50ms |
| **Throughput**     | 每秒推理次数 (FPS/QPS)       | 视频流 > 15 FPS, 音频实时率 > 1.0         |
| **TOPS**           | 每秒万亿次运算               | S300 NPU 典型值需实测                     |
| **TOPS/W**         | 能效比 (每瓦算力)            | 端侧 AI 关键指标，越高越好                |
| **首帧延迟**       | 模型加载到首次推理输出的时间 | < 500ms (含模型加载)                      |
| **内存占用**       | 推理时 Peak RAM 占用         | < 512KB (SRAM) 或 < 2MB (PSRAM)           |

### 2. 性能测试工具

| 工具                  | 功能                               | 链接                                                                   |
| :-------------------- | :--------------------------------- | :--------------------------------------------------------------------- |
| **TFLM Profiler**     | TensorFlow Lite Micro 内置性能分析 | [TFLM Profiling](https://www.tensorflow.org/lite/microcontrollers)     |
| **ONNX Runtime Perf** | ONNX 模型推理性能测试              | [onnxruntime perf_test](https://onnxruntime.ai/)                       |
| **Netron**            | 模型结构可视化，查看算子分布       | [Netron](https://github.com/lutzroeder/netron)                         |
| **tflite-benchmark**  | TFLite 模型延迟/内存基准测试       | [TFLite Tools](https://www.tensorflow.org/lite/performance/benchmarks) |
| **pyinstrument**      | Python 代码性能分析                | [pyinstrument](https://github.com/joerick/pyinstrument)                |

### 3. 性能测试脚本模板

```python
# S300 推理性能测试模板
import time
import numpy as np

def benchmark_inference(model, input_data, warmup=10, runs=100):
    """
    测量模型推理延迟
    Args:
        model: 推理模型对象
        input_data: 输入数据
        warmup: 预热次数 (排除首次加载开销)
        runs: 正式测试次数
    Returns:
        dict: 包含 avg/min/max/p95 延迟
    """
    # 预热
    for _ in range(warmup):
        model.predict(input_data)
    
    # 正式测试
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(input_data)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    return {
        "avg_ms": np.mean(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "throughput": 1000 / np.mean(latencies)  # FPS
    }
```

---

## 🔊 音频 AI 质量测试 (Audio Quality Metrics)

针对降噪、回声消除、语音增强等音频任务，需使用专业音频指标：

| 指标             | 全称                                    | 工具/库                                                                 | 说明                                                  |
| :--------------- | :-------------------------------------- | :---------------------------------------------------------------------- | :---------------------------------------------------- |
| **SI-SNR**       | Scale-Invariant SNR                     | [asteroid-metrics](https://github.com/asteroid-team/asteroid)           | **首选指标**。比传统 SNR 更鲁棒，评估降噪效果的标准。 |
| **PESQ**         | Perceptual Evaluation of Speech Quality | [pesq](https://github.com/ludlows/PESQ)                                 | ITU 标准，模拟人耳感知，评分 1-4.5。                  |
| **STOI**         | Short-Time Objective Intelligibility    | [pystoi](https://github.com/mpariente/pystoi)                           | 评估语音可懂度，范围 0-1。                            |
| **POLQA**        | Perceptual Objective Listening Quality  | 商业工具                                                                | PESQ 升级版，支持超宽带语音。                         |
| **ERLE**         | Echo Return Loss Enhancement            | 自行计算                                                                | 评估 AEC 回声消除效果，单位 dB。                      |
| **DNS-MOS**      | Deep Noise Suppression MOS              | [DNS Challenge](https://github.com/microsoft/DNS-Challenge)             | 微软 DNS 挑战赛指标，基于深度学习的 MOS 预测。        |
| **DNSMOS P.835** | 主观评分预测                            | [dnsmos](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | 预测 SIG/BAK/OVRL 三项主观分数。                      |

### 音频测试数据集

| 数据集            | 用途              | 链接                                                                |
| :---------------- | :---------------- | :------------------------------------------------------------------ |
| **DNS Challenge** | 降噪模型训练/测试 | [Microsoft DNS](https://github.com/microsoft/DNS-Challenge)         |
| **VCTK-DEMAND**   | 降噪评估          | [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791) |
| **AEC Challenge** | 回声消除评估      | [Microsoft AEC](https://github.com/microsoft/AEC-Challenge)         |
| **LibriSpeech**   | ASR 训练/测试     | [OpenSLR](https://www.openslr.org/12/)                              |
| **CommonVoice**   | 多语言语音        | [Mozilla](https://commonvoice.mozilla.org/)                         |

---

## 👁️ 视觉 AI 质量测试 (Vision Quality Metrics)

| 任务         | 核心指标              | 测试工具/方法                                              | 备注                          |
| :----------- | :-------------------- | :--------------------------------------------------------- | :---------------------------- |
| **目标检测** | mAP@0.5, mAP@0.5:0.95 | pycocotools                                                | COCO 标准评估协议             |
| **人脸检测** | AP (Easy/Medium/Hard) | WIDER FACE 官方评估脚本                                    | 分难度评估                    |
| **人脸识别** | TAR@FAR=1e-4, AUC     | InsightFace 评估工具                                       | LFW/CFP-FP/AgeDB 跨数据集测试 |
| **活体检测** | APCER, BPCER, ACER    | [ISO/IEC 30107-3](https://www.iso.org/standard/67381.html) | 攻击呈现分类错误率            |
| **人像分割** | mIoU, Boundary IoU    | PaddleSeg 评估                                             | 边缘精度更重要                |
| **姿态估计** | AP, PCK@0.2           | COCO Keypoint Eval                                         | 百分比正确关键点              |
| **深度估计** | Abs Rel, RMSE, δ<1.25 | [Eigen Split](https://github.com/mrharicot/monodepth)      | 单目深度标准指标              |
| **跟踪**     | MOTA, IDF1, HOTA      | [TrackEval](https://github.com/JonathonLuiten/TrackEval)   | MOT Challenge 官方评估        |

---

## 🔄 端到端系统测试 (E2E System Testing)

完整产品需进行端到端测试，验证多模块协同工作：

### 1. 系统级测试场景

| 测试场景         | 测试内容                        | 通过标准                         |
| :--------------- | :------------------------------ | :------------------------------- |
| **多任务并行**   | NPU 跑视觉 + DSP 跑音频同时工作 | 无资源冲突，延迟无明显增加       |
| **长时间稳定性** | 连续运行 24/72 小时             | 无内存泄漏，无崩溃，精度无衰减   |
| **极端输入**     | 全黑/全白图像，静音/爆音音频    | 不崩溃，输出合理                 |
| **边界条件**     | 最大/最小输入尺寸，极端光照     | 正确处理或优雅降级               |
| **功耗预算**     | 典型场景下系统功耗              | 满足产品功耗约束 (如 < 500mW)    |
| **热保护**       | 高负载下触发降频/停机保护       | 保护机制正常触发，恢复后功能正常 |
| **OTA 升级**     | 模型在线更新                    | 更新成功，回滚机制正常           |
| **异构调度**     | MCU/DSP/NPU 任务调度            | 调度延迟可预测，无死锁           |

### 2. 自动化测试框架推荐

| 框架                | 用途                 | 链接                                                   |
| :------------------ | :------------------- | :----------------------------------------------------- |
| **pytest**          | Python 单元测试      | [pytest](https://pytest.org/)                          |
| **Robot Framework** | 关键词驱动自动化测试 | [Robot Framework](https://robotframework.org/)         |
| **Unity Test**      | 嵌入式 C 单元测试    | [Unity](https://github.com/ThrowTheSwitch/Unity)       |
| **Ceedling**        | 嵌入式 TDD 框架      | [Ceedling](https://github.com/ThrowTheSwitch/Ceedling) |
| **Renode**          | 硬件仿真测试         | [Renode](https://renode.io/)                           |

---

## 🛠️ 推荐工具链与框架

除了标准的 TFLM，以下轻量级框架也非常适合 S300 这类资源受限的 MCU/NPU 环境：

*   **TensorFlow Lite for Microcontrollers (TFLM)**: 行业标准，支持 INT8 量化，适合 Cortex-M4/M7。
*   **CMSIS-NN**: ARM 官方优化库，能显著提升 DSP/MCU 上的推理性能。
*   **RKNN Toolkit Lite**: 针对国产 NPU 的模型转换与量化工具。
*   **Nnom (Neural Network on Microcontroller)**: [GitHub](https://github.com/majianjia/nnom)
    *   **特点**: 纯 C 语言编写，无依赖，专为 MCU 设计。支持 Keras 模型直接转换，部署极其简单。
    *   **适用性**: 非常适合 S300，特别是当 TFLM 占用资源过大时，Nnom 是极佳的替代方案。
*   **TinyMaix**: [GitHub](https://github.com/sipeed/TinyMaix)
    *   **特点**: 超轻量级（核心代码 <400行），支持 INT8/FP32，专为单片机设计。
    *   **适用性**: 如果 S300 的某些小核或协处理器需要运行极简模型，TinyMaix 是首选。

## ⚠️ 部署注意事项

1.  **模型大小控制**: 建议单模型 **< 2MB**，以便能放入片外 PSRAM 或 Flash 中运行。
2.  **延迟要求**: KWS < 30ms，人脸检测 < 100ms。
3.  **安全子系统**: 结合 AES/SM4/RSA 硬件加密，保证模型与数据安全。

---

## 扩展场景：机器人与无人系统 (Robotics & UAVs)

针对无人机、机器狗、具身智能等场景，S300 可承担**感知 (Perception)** 与 **轻量级决策 (Lightweight Decision)** 任务。

| 场景              | 任务         | 推荐模型 (Safe Choice) | 进阶模型 (Pro Choice)    | 源码/来源                                                                      | S300 适配理由                                                                |
| :---------------- | :----------- | :--------------------- | :----------------------- | :----------------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **无人机 (UAV)**  | **视觉跟随** | **NanoTrack**          | **LightTrack**           | [LightTrack](https://github.com/research/LightTrack)                           | LightTrack 比 NanoTrack 更轻量，NAS 搜索出的架构更适合端侧 NPU。             |
| **无人机/小车**   | **单目避障** | **FastDepth**          | **Pydnet**               | [Pydnet](https://github.com/FilippoAleotti/mobilePydnet)                       | Pydnet 专为手机端设计，金字塔结构非常适合 NPU 并行计算，延迟更低。           |
| **自动驾驶小车**  | **车道保持** | **PilotNet**           | **UFLD**                 | [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) | UFLD 将分割问题转化为行分类问题，速度极快且抗遮挡能力强，适合高速场景。      |
| **机器狗/机械臂** | **姿态估计** | **MoveNet**            | **PP-TinyPose**          | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)             | PP-TinyPose 在精度和速度上做了极致平衡，且 Paddle Lite 对国产 NPU 支持较好。 |
| **工业/仪表**     | **文字识别** | **CRNN**               | **PP-OCRv3 (Distilled)** | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)                         | PP-OCRv3 的超轻量版本是目前端侧 OCR 的事实标准，效果远超传统 CRNN。          |
| **具身智能**      | **异常检测** | **AutoEncoder**        | **Micro-TCN**            | 自研/通用                                                                      | TCN (时域卷积) 比 LSTM 更适合 NPU 并行，处理时序数据效率更高。               |

---

## 🌪️ 头脑风暴：S300 异构计算杀手级应用 (Killer Apps)

基于 S300 **"MCU控制 + DSP信号处理 + NPU深度学习"** 的黄金三角架构，以下场景能最大化其能效比：

### 1. 智能会议全向麦 (AI Conference Speakerphone)
*   **痛点**: 传统会议宝降噪差，无法分离人声，无法识别谁在说话。
*   **S300 分工**:
    *   **DSP (SensPro)**: 运行 **ClearVox** 进行 360° 波束成形 (Beamforming) 和回声消除 (AEC)。这是 DSP 的强项，低延迟处理 4-8 路麦克风阵列。
    *   **NPU**: 运行 **DCCRN** 或 **DeepFilterNet** 进行深度降噪，滤除键盘声、空调声；运行 **KWS** 识别“静音/结束”指令。
    *   **MCU**: 处理 USB UAC 协议（连接电脑）或蓝牙协议，控制 LED 灯环指示声源方向。
*   **优势**: 单芯片替代了传统的 "XMOS (DSP) + ARM (SoC)" 方案，成本降低 50%。

### 2. 工业电机“听诊器” (Industrial Motor Doctor)
*   **痛点**: 老师傅听声音辨故障难传承，云端分析带宽成本高。
*   **S300 分工**:
    *   **DSP**: 对高频振动传感器数据做 **FFT (快速傅里叶变换)**，提取频谱特征；做多传感器数据对齐。
    *   **NPU**: 运行 **1D-CNN** 或 **InceptionTime** 模型，对频谱图进行分类（轴承磨损、转子不平衡、气蚀）。
    *   **MCU**: 通过 RS485/CAN 总线上传报警信息，或通过 SDIO 接口记录黑匣子数据。
*   **优势**: 本地实时推理，毫秒级停机保护，无需上传原始数据，保护工厂隐私。

### 3. 极低功耗“哨兵”相机 (Low-Power Sentry Camera)
*   **痛点**: 电池供电的安防相机续航短，误报率高（风吹草动都录像）。
*   **S300 分工**:
    *   **MCU (AON)**: 处于休眠状态，仅保留 PIR 或 **VAD (DSP)** 监听环境。
    *   **DSP**: 当声音触发（如玻璃破碎声）或 PIR 触发时，DSP 快速预处理图像（ISP Lite）。
    *   **NPU**: 启动 **NanoDet-Plus** 检测是否为“人”或“车”。如果是误报（如猫狗、树叶），立即休眠；如果是真实目标，唤醒主系统录像并报警。
*   **优势**: 利用异构多级唤醒机制，将待机功耗压到极致，误报率远低于传统 PIR 方案。
### 4. 智能门锁/门禁 (AI Smart Lock)
*   **痛点**: 传统人脸门锁只认脸，不防照片/视频攻击；指纹识别卫生堪忧。
*   **S300 分工**:
    *   **DSP**: 运行 **WhisPro** 或轻量级 KWS，支持语音指令 "开门"、"访客模式"。
    *   **NPU**: 同时运行 **SCRFD (人脸检测)** + **MobileFaceNet (人脸识别)** + **活体检测 (Anti-Spoofing)**，三网级联，<100ms 完成。
    *   **MCU**: 控制电机开锁，管理访客日志（存 SD 卡），处理 BLE/WiFi 远程解锁。
*   **优势**: 单芯片实现**多模态认证 (人脸+语音)**，比单一方式更安全，比云端方案更私密。

### 5. 智能养老看护 (Elderly Care Guardian)
*   **痛点**: 独居老人摔倒后无人知晓，传统穿戴设备老人不愿意戴。
*   **S300 分工**:
    *   **DSP**: 通过毫米波雷达 (UART接口) 感知人体存在和粗略运动轨迹，不侵犯隐私。
    *   **NPU**: 运行 **InceptionTime** 或 **1D-CNN** 对雷达点云时序数据进行分类 (正常活动、跌倒、长时间静止)。
    *   **MCU**: 异常时通过 **GMAC (千兆以太网)** 或 **SDIO WiFi** 推送报警到家属手机/社区服务中心。
*   **优势**: 雷达方案完全**无视觉隐私**问题，老人接受度极高，且不受光照影响，黑夜也能工作。

### 6. 农业病虫害识别仪 (Crop Disease Detector)
*   **痛点**: 农民不懂病虫害，等发现时往往已大面积扩散。
*   **S300 分工**:
    *   **MCU**: 通过 **DVP 接口**采集叶片图像，控制补光灯和LCD屏显示结果。
    *   **DSP**: 对图像进行预处理（白平衡、裁剪感兴趣区域）。
    *   **NPU**: 运行专门训练的 **MobileNetV3** 或 **EfficientNet-Lite** 分类模型，识别"锈病"、"蚜虫"、"健康"等状态。
*   **优势**: 离线工作，带到田间即用，不需要联网，适合偏远农村场景。

### 7. 宠物/婴儿监控器 (Smart Pet/Baby Cam)
*   **痛点**: 普通监控只录像，不知道宠物/婴儿何时异常（哭闹、啃家具）。
*   **S300 分工**:
    *   **DSP (PDM)**: 实时监听哭声/吠叫声，运行 **WhisPro** 做声音事件分类 (哭泣 vs 正常玩耍)。
    *   **NPU**: 运行 **NanoDet-Plus** 检测物体（婴儿位置、狗接近危险物品），或 **MoveNet** 判断婴儿是否翻身。
    *   **MCU (I2S)**: 声音异常时通过喇叭播放安抚音乐。
*   **优势**: 真正"理解"场景，而非只是录像回放；推送有意义的事件而非无用片段。

### 8. 智能语音玩具 (AI Voice Toy)
*   **痛点**: 传统玩具只能播放预录语音，互动性差。
*   **S300 分工**:
    *   **DSP**: 通过 **PDM** 麦克风采集童音，运行 **ClearVox** 降噪，避免背景音干扰。
    *   **NPU**: 运行 **BC-ResNet (KWS)** 识别简单指令（"讲故事"、"唱儿歌"），或运行轻量级 **意图分类模型**。
    *   **MCU**: 控制 **I2S** 音频输出播放对应内容，控制 LED 眼睛/嘴巴动画，通过 WiFi 更新语料库。
*   **优势**: 无需联网也能基础交互，配合云端可实现更复杂对话，成本远低于基于 Linux 的语音方案。

### 9. 智能门铃 (AI Video Doorbell)
*   **痛点**: 传统门铃只能通知"有人按铃"，不能告诉你是谁、是否可疑。
*   **S300 分工**:
    *   **DSP (PDM)**: 监听门口声音，识别敲门声、门铃声、异常撬锁声。
    *   **NPU**: 运行 **SCRFD+MobileFaceNet** 识别家人/陌生人；运行 **NanoDet** 检测快递包裹。
    *   **MCU**: 通过 **GMAC/WiFi** 推送通知到手机，支持双向语音对讲 (I2S)。
*   **优势**: 本地人脸识别，响应更快，且不依赖云端服务器，保护家庭隐私。

### 10. 智能健身镜/私教 (AI Fitness Mirror)
*   **痛点**: 健身动作不标准容易受伤，请私教太贵。
*   **S300 分工**:
    *   **DSP**: 采集健身音乐节奏，同步动作提示。
    *   **NPU**: 运行 **MoveNet/PP-TinyPose** 实时追踪用户骨骼点，判断动作是否标准（如深蹲膝盖是否超过脚尖）。
    *   **MCU (DVP)**: 采集用户图像，控制屏幕显示动作纠正提示。
*   **优势**: 真正的"AI 私教"，实时反馈，成本仅为真人私教的零头。

### 11. 工业视觉质检仪 (Industrial QC Inspector)
*   **痛点**: 传统工业相机只拍照，缺陷检测依赖昂贵的 PC 软件。
*   **S300 分工**:
    *   **MCU (DVP)**: 接收工业相机图像（支持 5MP），控制光源和剔除机构。
    *   **DSP**: 做图像预处理（去噪、增强对比度）。
    *   **NPU**: 运行 **NanoDet** 检测缺陷位置，或运行 **AutoEncoder** 做异常检测（学习正常样本，偏离即为缺陷）。
*   **优势**: 嵌入式方案，体积小、功耗低、响应快，适合部署在流水线各个工位。

### 12. 智能垃圾分类桶 (Smart Trash Bin)
*   **痛点**: 居民不会分类垃圾，罚款也不管用。
*   **S300 分工**:
    *   **MCU (DVP)**: 采集投放口的图像。
    *   **NPU**: 运行 **MobileNetV3** 或 **EfficientNet-Lite** 对垃圾进行分类（可回收/有害/湿垃圾/干垃圾）。
    *   **MCU (PWM/GPIO)**: 控制分类挡板打开对应的桶，LED 显示分类结果。
*   **优势**: 全自动分类，用户无需思考，解决"不会分"的痛点。

### 13. 车载疲劳驾驶预警 (Driver Fatigue Monitor)
*   **痛点**: 疲劳驾驶是重大事故隐患，传统方案基于方向盘握力，误报多。
*   **S300 分工**:
    *   **MCU (DVP)**: 红外摄像头采集驾驶员面部（夜间可用）。
    *   **NPU**: 运行 **SCRFD** 检测人脸关键点，计算眼睛闭合度 (PERCLOS) 和打哈欠频率。
    *   **DSP**: 通过 **I2S** 播放警示音或语音提醒"您已疲劳，请休息"。
*   **优势**: 基于视觉的方案比握力传感器更准确，红外相机不受光照影响。

### 14. 智能听诊器 (AI Stethoscope)
*   **痛点**: 基层医生缺乏经验，难以通过听诊判断心肺疾病。
*   **S300 分工**:
    *   **DSP (PDM/I2S)**: 采集心音/肺音，做带通滤波和降噪。
    *   **NPU**: 运行 **1D-CNN** 或 **Wav2Vec Tiny** 对音频进行分类（正常/心律不齐/哮鸣音/湿罗音）。
    *   **MCU**: 通过蓝牙将结果发送到手机 App，显示诊断建议。
*   **优势**: 辅助基层医生快速筛查，降低漏诊率；设备成本低，易于普及。

### 15. 无人机智能吊舱 (UAV AI Gimbal)
*   **痛点**: 消费级无人机拍摄的视频需要人工回看，找不到关键画面。
*   **S300 分工**:
    *   **MCU (DVP)**: 接收云台相机图像。
    *   **NPU**: 运行 **NanoDet-Plus** 实时检测目标（人/车/动物），运行 **LightTrack** 自动跟踪。
    *   **DSP (MotionEngine)**: 通过 **IMU** 数据补偿相机抖动，辅助稳像。
*   **优势**: 实现"边飞边识别"，自动标记关键帧，大幅减少后期工作量。

---

## 🧩 更多潜力模型推荐 (Model Expansion)

除了前文提到的，以下领域的模型也非常适合 S300 的 NPU 架构：

| 领域         | 任务          | 推荐模型                | 理由                                                                                                                   |
| :----------- | :------------ | :---------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **视觉分割** | **人像分割**  | **PP-HumanSeg** (Lite)  | 百度推出的超轻量级分割模型，适合做会议背景虚化或替换，NPU 跑起来很轻松。                                               |
| **视觉分割** | **车道/区域** | **BiSeNet V2** (Pruned) | 双边分割网络，一条路径处理细节，一条处理语义，速度极快，适合扫地机避障区域识别。                                       |
| **OCR**      | **仪表读数**  | **PP-OCRv4-mobile**     | 目前最强的轻量级 OCR 系统。S300 可仅部署其 **Det (检测)** 和 **Rec (识别)** 模型，用于读取水表、电表数字。             |
| **时序数据** | **动作分类**  | **InceptionTime**       | 类似于图像的 Inception 结构，但用于一维时间序列。非常适合处理 IMU 数据来识别“跌倒”、“挥手”等动作。                     |
| **音频**     | **混合降噪**  | **RNNoise**             | 经典的混合方案。DSP 跑特征提取（Bark scale），NPU (GRU/LSTM) 跑增益预测。如果 S300 NPU 对 RNN 支持有限，这是最佳备选。 |

---

## 总结

真正能在 S300 上落地的方案是：**SpeexDSP (AEC) + DCCRN (NPU降噪) + TFLM/Nnom (推理框架)**。

*   **WebRTC 3A**: 仅建议复用其 **AGC** 模块，AEC 和 NS 建议替换为更适合 S300 硬件的方案。**但 WebRTC 3A 可作为“效果基线”进行移植对比。**
*   **模型选择**: 请严格参考上表中的开源项目，并结合 S300 的 NPU 算子限制（如 Kernel <= 7x7）进行裁剪和训练。
---

## 🐍 Python 环境管理 (UV)

本项目使用 **[UV](https://github.com/astral-sh/uv)** 管理 Python 版本和虚拟环境。UV 是一个极速的 Python 包管理器，比 pip 快 10-100 倍。

### 1. 安装 UV

```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip 安装
pip install uv
```

### 2. Python 版本管理

```bash
# 查看可用的 Python 版本
uv python list

# 安装指定版本的 Python
uv python install 3.11

# 固定项目使用的 Python 版本 (生成 .python-version 文件)
uv python pin 3.11
```

### 3. 虚拟环境管理

```bash
# 创建虚拟环境 (默认使用 .venv 目录)
uv venv

# 创建指定 Python 版本的虚拟环境
uv venv --python 3.11

# 激活虚拟环境
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

### 4. 依赖管理

```bash
# 安装单个包 (自动激活虚拟环境)
uv pip install torch onnx

# 从 requirements.txt 安装
uv pip install -r requirements.txt

# 导出当前环境的依赖
uv pip freeze > requirements.txt

# 同步依赖 (确保环境与 requirements.txt 一致)
uv pip sync requirements.txt
```

### 5. 项目初始化 (推荐)

```bash
# 初始化一个新项目 (生成 pyproject.toml)
uv init

# 添加依赖到 pyproject.toml
uv add torch numpy onnxruntime

# 添加开发依赖
uv add --dev pytest black ruff

# 同步项目依赖
uv sync
```

### 6. S300 模型开发常用依赖

```bash
# 模型训练与转换
uv add torch torchvision torchaudio onnx onnxruntime

# TensorFlow Lite 转换
uv add tensorflow tflite-support

# PaddlePaddle 系列模型
uv add paddlepaddle paddleslim paddle2onnx

# 音频处理
uv add librosa soundfile pydub

# 模型量化工具
uv add onnxruntime-tools neural-compressor
```

### 7. UV vs 其他工具对比

| 特性                | UV                 | pip + venv    | Conda    |
| :------------------ | :----------------- | :------------ | :------- |
| **安装速度**        | ⚡ 极快 (Rust 实现) | 🐢 较慢        | 🐢 较慢   |
| **Python 版本管理** | ✅ 原生支持         | ❌ 需 pyenv    | ✅ 支持   |
| **虚拟环境**        | ✅ 内置             | ✅ 内置 (venv) | ✅ 内置   |
| **依赖解析**        | ✅ 快速精确         | ⚠️ 较慢        | ✅ 精确   |
| **磁盘占用**        | ✅ 小               | ✅ 小          | ❌ 较大   |
| **pyproject.toml**  | ✅ 原生支持         | ⚠️ 部分        | ❌ 不支持 |

---

## 📝 Git 提交规范 (Commit Convention)

本项目遵循 **Conventional Commits** 规范，并结合 **Commit-As-Prompt** 哲学，使提交信息对人类和 AI 都友好可读。

### 1. 提交格式

```
<type>(<scope>): <简短描述>

WHAT: 一句话描述动作与对象

WHY: 阐述业务目标、用户需求或缺陷背景

HOW: 概述实现策略、验证方式、风险提示
```

### 2. Type 类型定义

| Type       | 含义   | 使用场景                           |
| :--------- | :----- | :--------------------------------- |
| `feat`     | 新功能 | 新增功能特性                       |
| `fix`      | 修复   | 修复 Bug                           |
| `docs`     | 文档   | 文档变更                           |
| `style`    | 格式   | 代码格式调整（不影响逻辑）         |
| `refactor` | 重构   | 代码重构（不新增功能、不修复 Bug） |
| `perf`     | 性能   | 性能优化                           |
| `test`     | 测试   | 添加或修改测试                     |
| `build`    | 构建   | 构建系统或外部依赖变更             |
| `ci`       | CI/CD  | 持续集成配置变更                   |
| `chore`    | 杂务   | 其他不修改源码的变更               |
| `revert`   | 回滚   | 撤销之前的提交                     |
| `prompt`   | 提示词 | 上下文/Prompt 专用提交             |

### 3. Scope 建议值

| Scope       | 说明                  |
| :---------- | :-------------------- |
| `npu`       | NPU 相关模型/算子     |
| `dsp`       | DSP 音频/传感器处理   |
| `mcu`       | MCU 控制逻辑/外设     |
| `audio`     | 音频模型 (降噪/KWS)   |
| `vision`    | 视觉模型 (检测/识别)  |
| `test`      | 测试相关              |
| `docs`      | 文档相关              |
| `env`       | 环境配置 (UV/Python)  |
| `agents`    | AGENTS.md 文档本身    |

### 4. 提交示例

```bash
# 新增功能
feat(audio): 集成 DCCRN 降噪模型

WHAT: 为音频 Pipeline 添加 DCCRN 实时降噪支持

WHY: 替代传统 WebRTC NS，提升非稳态噪声（键盘声、空调声）的抑制效果

HOW: 
- 使用 INT8 量化版本，模型大小 1.2MB
- NPU 推理延迟 < 10ms (16kHz 帧)
- 已通过 DNS-MOS 评估，OVRL 提升 0.3 分

# 文档更新
docs(agents): 补充性能测试章节

WHAT: 在 AGENTS.md 中新增完整的测试体系文档

WHY: 为开发团队提供标准化的模型验证流程

HOW: 涵盖功能测试、性能测试、音频质量、视觉质量、E2E 系统测试五大维度
```

### 5. 语言规范

- **Type/Scope**: 使用英文小写
- **Subject**: 中文，不超过 50 字，末尾无标点
- **Body (WHAT/WHY/HOW)**: 中文为主，技术术语可用英文