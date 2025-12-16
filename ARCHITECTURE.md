# S300 技术架构与测试指南 (Technical Architecture & Testing Guide)

本文档详细描述了 S300 芯片的 AI 架构分析、性能基准测试、模型验证方法以及端到端系统测试流程。

## 💡 核心架构分析：WebRTC 3A vs S300 Native

关于 **WebRTC 3A (AEC/ANS/AGC)** 是否适合 S300，结论是：**不推荐直接移植作为主力方案，但强烈建议作为“效果基线 (Golden Baseline)”进行对比。**

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

## 🧩 更多潜力模型推荐 (Model Expansion)

除了前文提到的，以下领域的模型也非常适合 S300 的 NPU 架构：

| 领域         | 任务          | 推荐模型                | 理由                                                                                                                   |
| :----------- | :------------ | :---------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **视觉分割** | **人像分割**  | **PP-HumanSeg** (Lite)  | 百度推出的超轻量级分割模型，适合做会议背景虚化或替换，NPU 跑起来很轻松。                                               |
| **视觉分割** | **车道/区域** | **BiSeNet V2** (Pruned) | 双边分割网络，一条路径处理细节，一条处理语义，速度极快，适合扫地机避障区域识别。                                       |
| **OCR**      | **仪表读数**  | **PP-OCRv4-mobile**     | 目前最强的轻量级 OCR 系统。S300 可仅部署其 **Det (检测)** 和 **Rec (识别)** 模型，用于读取水表、电表数字。             |
| **时序数据** | **动作分类**  | **InceptionTime**       | 类似于图像的 Inception 结构，但用于一维时间序列。非常适合处理 IMU 数据来识别“跌倒”、“挥手”等动作。                     |
| **音频**     | **混合降噪**  | **RNNoise**             | 经典的混合方案。DSP 跑特征提取（Bark scale），NPU (GRU/LSTM) 跑增益预测。如果 S300 NPU 对 RNN 支持有限，这是最佳备选。 |
