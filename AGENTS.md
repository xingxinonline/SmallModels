# Project Context for AI Agents

This repository contains the recommended AI models, architecture analysis, and deployment guides for the **S300 Chip** (NPU/DSP/MCU).

## 📂 Documentation Structure

*   **[README.md](README.md)**: Project overview, hardware specifications (NPU/DSP/MCU), and the list of deployable open-source models. **Start here for general context.**
*   **[ARCHITECTURE.md](ARCHITECTURE.md)**: Technical deep dive, including architecture analysis (WebRTC vs Native), performance benchmarking, model validation protocols, and end-to-end system testing.

## 🤖 Hardware Constraints (Critical for Code Generation)

When generating code or suggesting models for this project, **ALWAYS** keep the following S300 hardware constraints in mind:

*   **NPU (Neural Processing Unit)**:
    *   **Precision**: Int8 / Int16 only. **No Float32 support in NPU.**
    *   **Operators**: Conv2D (Kernel <= 7x7), Pooling (Kernel <= 15), ReLU/Leaky-ReLU/Softmax.
    *   **Unsupported**: Complex dynamic control flow, large kernels (>7x7), some advanced activations (e.g., GELU, Swish need approximation).
*   **DSP (SensPro 250)**:
    *   Best for: Audio pre-processing (AEC, AGC, VAD), FFT, Sensor Fusion.
    *   Libraries: CEVA ClearVox, CMSIS-DSP.
*   **MCU (Cortex-M4)**:
    *   Best for: System control, peripheral management, lightweight business logic.
    *   Avoid heavy computation here.

## 🛠️ Development Guidelines

### Python Environment (UV)

We use **[uv](https://github.com/astral-sh/uv)** for fast Python package management.

*   **Install**: `pip install uv`
*   **Sync**: `uv pip sync requirements.txt`
*   **Add Package**: `uv add <package>`

### Git Commit Convention

Follow the **Conventional Commits** format:

```
<type>(<scope>): <short description>

WHAT: ...
WHY: ...
HOW: ...
```

*   **Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
*   **Scopes**: `npu`, `dsp`, `mcu`, `audio`, `vision`, `test`, `docs`, `gesture`, `recognizer`, `matching`.

### Git Commit Splitting (拆分提交规范) ⭐

**当一次开发涉及多个独立功能时，必须拆分提交**：

1. **按功能模块拆分**：每个独立的功能点一个提交
2. **按文件类型拆分**：代码、文档、测试分开提交
3. **保持提交原子性**：每个提交应该是可独立理解的完整变更

**拆分示例**：

```bash
# 错误：一个大提交包含所有改动
git commit -m "feat: 完成目标跟随优化"

# 正确：按功能拆分
git commit -m "feat(gesture): 手势检测器优化 - 选择做有效手势的手"
git commit -m "refactor(recognizer): 多视角识别器改进 - 视角库管理策略"
git commit -m "feat(matching): 分层匹配策略 - 基于人脸质量分级"
git commit -m "docs: 更新方案D启动逻辑和匹配策略文档"
```

**拆分原则**：
- ✅ 一个功能点 = 一个提交
- ✅ 相关的代码+测试可以放一起
- ✅ 文档更新单独提交
- ❌ 不要把不相关的改动混在一起
- ❌ 不要为了"干净"而 squash 有意义的历史

## 📝 Task Instructions

*   If asked to **recommend a model**, check `README.md` first.
*   If asked about **testing or validation**, refer to `ARCHITECTURE.md`.
*   If asked to **write code**, ensure it is compatible with the S300 constraints (e.g., use quantization-aware training, avoid unsupported ops).

## 💻 Local PC Environment (Pre-validation)

Before porting to the S300 chip, models will be validated on the local PC to verify logic and performance baselines.

*   **OS**: Windows 11 Pro (10.0.22631)
*   **CPU**: Intel Core i7-12700KF (12 Cores, 20 Threads)
*   **RAM**: 32 GB
*   **GPU**: NVIDIA GeForce RTX 3070 Ti
*   **Goal**: Run FP32/Int8 models locally to check functional correctness and simulate NPU constraints (e.g., using TFLite interpreter with Int8 delegates).

## 🔄 Development Workflow (开发流程)

When implementing a new feature or model validation, follow this **complete workflow**:

### 1. Architecture Design (架构设计)

*   Define system modules and their responsibilities
*   Draw data flow diagrams
*   Specify interfaces between components
*   Document in `examples/<feature>/README.md`

### 2. Code Implementation (代码实现)

*   Create modular, reusable code structure
*   Follow Python best practices (type hints, docstrings)
*   Separate concerns: config, capture, inference, visualization

### 3. Dependency Verification (依赖验证)

```bash
# Install dependencies using uv
uv add <package_name>

# Verify installation
uv run python -c "import <package>; print(<package>.__version__)"
```

### 4. Compile Verification (编译验证)

```bash
# Check for syntax errors
uv run python -m py_compile <file.py>

# Or use IDE's built-in linting
```

### 5. Execution Testing (执行测试)

```bash
# Run the application
uv run python <main_script.py>
```

### 5.1 Interactive Testing with Background Processes (交互式后台测试)

**⚠️ CRITICAL: When running interactive applications (camera, GUI, gesture control, etc.):**

1. **Do NOT use `isBackground=true` for tests requiring user interaction**
   - Background mode cannot capture user input or show real-time output
   - Agent will not receive test results automatically

2. **Correct approach for interactive tests:**
   ```bash
   # Run in foreground (isBackground=false)
   uv run python <interactive_script.py>
   ```

3. **If background mode is necessary:**
   - Wait sufficient time for user to complete testing
   - Use `get_terminal_output` to actively check results
   - Do NOT assume test passed without checking output
   - Agent MUST call `get_terminal_output` after reasonable wait time (5-10 seconds)

4. **Debug logging best practice:**
   - Add debug logs for state transitions and key events
   - Use conditional debug flags: `process_gesture(..., debug=True)`
   - Print logs in a parseable format for automated analysis

**Example - Wrong approach:**
```python
# ❌ Wrong: Start background process and immediately ask user for results
run_in_terminal(command, isBackground=true)
# Then ask user: "请告诉我结果"
```

**Example - Correct approach:**
```python
# ✅ Correct: Start background process, wait, then check output
run_in_terminal(command, isBackground=true)
# Wait for user to interact...
get_terminal_output(terminal_id)  # Agent actively fetches results
# Analyze output and provide feedback
```

### 6. Documentation Update (文档更新)

*   Update `AGENTS.md` with new workflow requirements
*   Update `README.md` if new models are added
*   Create example-specific documentation

## 📁 Examples Directory Structure

```
examples/
├── face_detection/           # 人脸检测示例
│   ├── README.md             # 架构设计文档
│   ├── config.py             # 配置参数
│   ├── camera.py             # 摄像头采集
│   ├── detector.py           # SCRFD 检测器
│   ├── visualizer.py         # 可视化模块
│   ├── download_model.py     # 模型下载
│   ├── main.py               # 主程序入口
│   └── models/               # 模型文件
├── target_following/         # 目标跟随示例 (手势控制)
│   ├── README.md             # 架构设计文档
│   ├── config.py             # 配置参数与状态枚举
│   ├── main.py               # 主程序入口
│   ├── core/                 # 核心模块
│   │   ├── camera.py         # 摄像头采集
│   │   └── state_machine.py  # 状态机控制器
│   ├── detectors/            # 检测器模块
│   │   ├── gesture_detector.py    # 手势检测 (MediaPipe)
│   │   ├── face_detector.py       # 人脸检测 (SCRFD)
│   │   ├── face_recognizer.py     # 人脸识别 (ArcFace)
│   │   └── person_detector.py     # 人体检测 (YOLOv8-pose)
│   ├── trackers/             # 跟踪模块
│   │   └── target_tracker.py # 目标跟踪器
│   ├── visualizers/          # 可视化模块
│   │   └── visualizer.py     # 结果绘制
│   ├── tests/                # 单元测试
│   │   ├── test_gesture.py   # 手势检测测试
│   │   ├── test_face.py      # 人脸识别测试
│   │   └── test_person.py    # 人体检测测试
│   └── models/               # 模型文件
│       ├── scrfd_500m_bnkps.onnx   # 人脸检测
│       ├── w600k_r50.onnx          # 人脸识别
│       └── yolov8n-pose.onnx       # 人体姿态
└── <future_examples>/        # 更多示例...
```

## 🎯 Example: Face Detection Workflow

```bash
# 1. Navigate to example directory
cd examples/face_detection

# 2. Download model
uv run python download_model.py

# 3. Run face detection with camera
uv run python main.py

# Controls:
# - Press 'q' to quit
# - Press 's' to save screenshot
```

## 🎯 Example: Target Following Workflow

```bash
# 1. Navigate to example directory
cd examples/target_following

# 2. Run individual tests first (recommended)
uv run python tests/test_gesture.py  # Test gesture detection
uv run python tests/test_face.py     # Test face recognition
uv run python tests/test_person.py   # Test person detection

# 3. Run integrated target following
uv run python main.py

# Gesture Controls:
# - Open Palm (张开手掌): Start tracking - locks current face as target
# - Closed Fist (握拳): Stop tracking - returns to idle state
# - Press 'q' to quit

# State Machine:
# IDLE → (Open Palm) → TRACKING → (Closed Fist) → IDLE
#                    ↓
#               LOST_TARGET (if target lost, waits for re-detection)
```

