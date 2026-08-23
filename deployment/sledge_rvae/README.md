# SLEDGE RVAE 车端离线部署包

本目录把现有 `epoch45.ckpt` 转换为固定形状 ONNX，并提供 CUDA 12.3 + TensorRT 8.6.1 下的 Engine 构建、C++ 推理、三端一致性和性能测试闭环。当前部署对象是 SLEDGE raster-vector autoencoder（RVAE），不是多步 diffusion 采样器。

## 1. 当前交付状态

| 提交物 | 路径 | 状态 |
|---|---|---|
| ONNX 导出与验证 | `python/export_onnx.py` | 已生成并通过 `onnx.checker`/ONNX Runtime |
| 固定 nuPlan 样例与 PyTorch 参考 | `python/prepare_sample.py` | 已从现有 `sledge_raw.gz` 生成 |
| C++ TensorRT 闭环 | `cpp/` | 源码、CMake、配置完成；TRT 8.6.1/CUDA 12.3 编译链接通过 |
| Engine 构建 | `scripts/build_engine.sh` | 固定 profile，FP32/FP16 均支持；待目标 GPU 实际构建 |
| 三端一致性 | `python/validate.py` | PyTorch/ORT 已通过；TensorRT 端待目标 GPU |
| 性能材料 | `scripts/benchmark.sh`、`python/performance_report.py` | 采集链路完成；真实性能数据待目标 GPU |

`.onnx`、`.engine`、固定样例和实测报告属于生成产物，默认写入 `artifacts/` 和 `reports/`。TensorRT Engine 与 GPU 架构、TensorRT/CUDA 版本绑定，最终文件必须在目标 GPU 上重新构建。

## 2. 固定模型契约

### 输入

- 名称：`raster`
- dtype：`float32`
- shape/layout：`[1, 12, 256, 256]`，NCHW，固定 batch 1
- 场景范围：ego 中心附近 `64 m × 64 m`
- 分辨率：`0.25 m/pixel`
- 数值：SLEDGE 预处理输出的 float raster，不再做图像的 `/255`、mean/std 归一化

通道顺序来自项目 `SledgeRasterIndex`：

| 通道 | 内容 |
|---:|---|
| 0–1 | map lines 的方向编码 x/y |
| 2–3 | vehicles 的速度方向编码 x/y，包含 ego box |
| 4–5 | pedestrians 的速度方向编码 x/y |
| 6–7 | static objects 的朝向编码 x/y |
| 8–9 | green traffic-light lines 的方向编码 x/y |
| 10–11 | red traffic-light lines 的方向编码 x/y |

C++ 入口接受上述 NCHW `.npy`；也接受等价 NHWC/HWC float32 `.npy` 并在预处理阶段转置。当前阶段以文件/张量接口交付，nuPlan/车辆 topic 到 12 通道 raster 的接入点保留在调用方。

### 原始输出

所有输出均为 `float32`。`*_logits` 是存在性 logit，后处理执行 `sigmoid(logit) >= 0.3`。

| 名称 | shape | 语义 |
|---|---|---|
| `lines_states` | `[1,50,20,2]` | 50 个 line query，每条 20 个 XY 点，单位 m |
| `lines_logits` | `[1,50]` | line 存在性 logit |
| `vehicles_states` | `[1,50,6]` | x, y, heading, width, length, velocity；m/rad/m/s |
| `vehicles_logits` | `[1,50]` | vehicle 存在性 logit |
| `pedestrians_states` | `[1,20,6]` | 同 agent 状态格式 |
| `pedestrians_logits` | `[1,20]` | pedestrian 存在性 logit |
| `static_objects_states` | `[1,30,5]` | x, y, heading, width, length |
| `static_objects_logits` | `[1,30]` | static object 存在性 logit |
| `green_lights_states` | `[1,20,20,2]` | green light line 点 |
| `green_lights_logits` | `[1,20]` | green light line logit |
| `red_lights_states` | `[1,20,20,2]` | red light line 点 |
| `red_lights_logits` | `[1,20]` | red light line logit |
| `ego_states` | `[1,1]` | 当前 checkpoint 的单标量 ego head 输出 |
| `ego_mask` | `[1,1]` | 常量 1，保留原模型输出契约 |

训练代码通过 `mu + eps*std` 随机采样 latent。部署图固定使用 latent 均值 `mu`，否则重复推理及三端一致性不可复现。包装层只把 decoder 中的原位 slice 赋值改写为等价 `concat`，避免导出 `ScatterND`；数学定义和 checkpoint 权重不变。

## 3. 环境基线

目标机必须满足：

- NVIDIA driver 与 CUDA 12.3 runtime 兼容
- CUDA Toolkit 12.3.x
- TensorRT 8.6.1（C++ headers、`libnvinfer`、`trtexec`）
- CMake >= 3.18，支持 C++17 的编译器
- Python 导出环境包含项目依赖及 `requirements-export.txt`

本项目已在 `deployment/sledge_rvae/.venv` 创建部署虚拟环境。复现方式：先激活原 SLEDGE 环境，再执行：

```bash
conda activate sledge
bash deployment/sledge_rvae/scripts/bootstrap_env.sh
source deployment/sledge_rvae/scripts/activate_deploy.sh
```

该环境通过 `--system-site-packages` 复用与 checkpoint 匹配的 SLEDGE/PyTorch/nuPlan 二进制依赖，并在隔离目录安装 ONNX 1.16.1、ONNX Runtime 1.18.1、TensorRT 8.6.1 Python/核心库和 CUDA runtime 12.3.101。必须 source `activate_deploy.sh`，它会补齐 NVIDIA wheel 动态库路径。`python_environment.json` 记录全部 import 结果。

导出环境里的 PyTorch 只在 CPU 上生成参考结果；车端 C++ 不链接 PyTorch。原 SLEDGE 环境存在历史包元数据冲突，因此以 `check_environment.py`、模型加载和端到端导出验证为准，不建议为消除 `pip check` 警告而升级原模型依赖。

不要用其他 TensorRT 版本加载本包生成的 Engine。CMake 会拒绝非 CUDA 12.3.x，C++ 编译期会拒绝非 TensorRT 8.6.x。

## 4. 一键流程

在仓库根目录执行：

```bash
conda activate sledge
bash deployment/sledge_rvae/scripts/bootstrap_env.sh
bash deployment/sledge_rvae/scripts/run_target_pipeline.sh
```

也可以逐步执行。

### 4.1 导出固定样例与 PyTorch 参考

```bash
python -m deployment.sledge_rvae.python.prepare_sample --device cpu
```

默认样例来自现有 nuPlan feature cache，并保存：

- `input_raster.npy`
- `pytorch_reference.npz`
- `pytorch/*.npy`
- `metadata.json`

可用 `--cache-sample` 指向另一份 nuPlan 导出的 `sledge_raw.gz`，形成多样例集合。

### 4.2 导出 ONNX

```bash
python -m deployment.sledge_rvae.python.export_onnx
```

导出固定 shape、opset 17 的 `artifacts/sledge_rvae.onnx`，随后执行 `onnx.checker`，检查输入输出名称并写 SHA-256 manifest。

### 4.3 构建 Engine

先做 FP32：

```bash
PRECISION=fp32 bash deployment/sledge_rvae/scripts/build_engine.sh
```

FP32 一致性通过后再做 FP16：

```bash
PRECISION=fp16 bash deployment/sledge_rvae/scripts/build_engine.sh
```

脚本记录 GPU、driver、系统、TensorRT 版本、固定 profile、workspace、自定义插件（当前为 none）、完整 `trtexec --verbose` 日志和文件 SHA-256。INT8 未启用：当前没有校准集和 INT8 精度报告。

若目标机没有 `trtexec`，脚本会自动改用 `python/build_engine.py` 的 TensorRT 8.6.1 Python Builder API，构建参数和固定 Profile 保持一致。

### 4.4 编译和运行 C++

```bash
cmake -S deployment/sledge_rvae/cpp \
  -B deployment/sledge_rvae/cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT=/path/to/TensorRT-8.6.1
cmake --build deployment/sledge_rvae/cpp/build --parallel

deployment/sledge_rvae/cpp/build/sledge_rvae_trt \
  --config deployment/sledge_rvae/configs/runtime.ini
```

如果使用本包通过 pip 安装的 CUDA 12.3 runtime，可额外传入
`-DCUDA_RUNTIME_ROOT=deployment/sledge_rvae/.venv/lib/python3.9/site-packages/nvidia/cuda_runtime`；正式车端仍建议使用系统 CUDA Toolkit 12.3。

C++ 流程包含：

1. 读取和验证 float32 `.npy`，NCHW/NHWC 转换、shape/NaN/Inf 检查；
2. Engine 反序列化、固定输入绑定、page-locked host memory、GPU memory、CUDA Stream；
3. 异步 H2D、`enqueueV2`、D2H 和 CUDA event 分段计时；
4. sigmoid + threshold 后处理，原始输出 `.npy` 和紧凑 JSON 落盘；
5. 预热后统计 mean/P50/P95/P99/min/max、吞吐、Engine 加载时间和显存。

路径、shape、精度标签、阈值、预热次数、测试次数和是否保存输出均由 `configs/runtime.ini` 指定。切换 FP16 Engine 时必须同时把 `precision=fp16`，防止报告误标。

### 4.5 三端一致性

```bash
python -m deployment.sledge_rvae.python.validate \
  --trt-runner deployment/sledge_rvae/cpp/build/sledge_rvae_trt \
  --runner-config deployment/sledge_rvae/configs/runtime.ini \
  --precision fp32
```

报告写到 `reports/consistency.json` 和 `.md`。脚本比较：

- 每个原始 tensor 的最大/平均绝对误差；
- 全输出加权平均绝对误差；
- 后处理后的 query 数量、active-query IoU、概率误差、激活 query 上 state 误差。

默认验收阈值：FP32 `max_abs <= 5e-4` 且 `mean_abs <= 5e-5`；FP16 `max_abs <= 5e-2` 且 `mean_abs <= 5e-3`。FP32 阈值对应 32 m 坐标范围约 `1e-5` 的相对量级，并要求后处理 active query 集合保持一致。若失败，先分别检查输入 tensor、ONNX Runtime；PyTorch–ONNX 已失败通常是导出算子问题，只有 TensorRT 失败通常是 TensorRT tactic/精度或 Engine 环境问题，只有后处理失败则检查阈值和 sigmoid。

### 4.6 性能测试

```bash
bash deployment/sledge_rvae/scripts/benchmark.sh
```

输出包含：

- 预处理、H2D、Engine、D2H、后处理、端到端延迟；
- 平均/P50/P95/P99/min/max；
- Engine 加载时间、吞吐率；
- Engine 加载后和运行峰值 CUDA 显存；
- `nvidia-smi` 采样的 GPU 利用率和总显存；
- 完整硬件/系统/driver/CUDA/TensorRT/编译器环境日志。

测试时关闭其他 GPU 任务、固定功耗/时钟策略并至少运行 100 次；报告必须连同 Engine precision 和输入 shape 一起提交。

## 5. 已知限制和实车接入口

- 目前只支持 batch 1 和固定 `1×12×256×256`。
- 当前 checkpoint 的 ego head 只有一个标量，不能宣称为完整 4D ego state；车辆侧使用前需与消费接口确认。
- C++ 当前输入是模型 tensor 文件/内存，真实 topic、地图查询、目标跟踪消息和 SLEDGE rasterization 在实车接入阶段适配。
- Engine 不可跨 TensorRT 版本/GPU 型号保证兼容，目标 GPU 必须重建。
- FP16 仅在独立通过一致性报告后发布；INT8 暂不发布。
- 本包只处理模型离线推理，不包含车辆安全降级、topic 时序、超时、健康监控和控制权限。
