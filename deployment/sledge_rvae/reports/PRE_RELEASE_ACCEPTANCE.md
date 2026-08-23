# 上车部署预验收结果

验收日期：2026-08-19。结论：**离线部署包、ONNX、目标构建代码和验证/性能采集链路已完成；需要 NVIDIA GPU 的最终产物尚未生成。**

| 要求 | 当前结果 | 证据/产物 | 目标 GPU 上剩余动作 |
|---|---|---|---|
| 固定 nuPlan 样例与 PyTorch 参考 | PASS | `artifacts/validation/sample_000/` | 无 |
| ONNX 导出与独立运行 | PASS | `artifacts/sledge_rvae.onnx`、manifest | 无 |
| PyTorch–ONNX Runtime 一致性 | PASS | `reports/consistency.json/.md` | 无 |
| TensorRT 定向静态检查 | PASS | `reports/onnx_compatibility.json/.md` | GPU 上执行原生 parser/build |
| C++ 预处理/推理/后处理源码 | PASS（编译级） | `cpp/`、`reports/cpp_compile_check.md` | 加载真实 Engine 运行 |
| CUDA 12.3 + TensorRT 8.6.1 环境 | PASS（用户态包） | `.venv`、`reports/python_environment.json` | 目标机确认 driver/GPU 与 Toolkit 12.3 |
| FP32 Engine | BLOCKED：本机无 NVIDIA driver/GPU | `scripts/build_engine.sh`、`python/build_engine.py` | 构建并保存 Engine/build log |
| FP16 Engine | BLOCKED：依赖 FP32 GPU 验收 | 同上 | FP32 通过后构建并验证 |
| PyTorch/ORT/TRT 三端一致性 | BLOCKED：没有可运行 Engine | `python/validate.py` | 生成真实 TRT 结果与报告 |
| 延迟/显存/GPU 利用率/吞吐 | BLOCKED：本机无 NVIDIA GPU | `scripts/benchmark.sh`、`python/performance_report.py` | 至少 100 次稳定测试并保存报告 |

## 已验证数值

- ONNX：opset 17，固定输入 `float32 [1,12,256,256]`，14 个固定输出。
- ONNX SHA-256：`6c3d96e04c6af9a1ac641cbf86319f44fce8f0ffb53fddb484deadf841f60dce`。
- PyTorch–ONNX Runtime：全局最大绝对误差 `3.5375357e-4`，全局平均绝对误差 `2.1180532e-5`。
- 六类后处理 active-query IoU 均为 `1.0`。
- C++：TensorRT `8.6.1` 头文件/库 + CUDA runtime `12.3.101` 完整编译链接通过。

## 目标机唯一入口

在 CUDA Toolkit 12.3、TensorRT 8.6.1 且 NVIDIA GPU/driver 正常的机器上，从仓库根目录执行：

```bash
bash deployment/sledge_rvae/scripts/run_target_pipeline.sh
```

脚本先执行严格版本预检，再构建 FP32 Engine、编译 C++、做三端一致性和性能采集。任何一步失败都会非零退出，不会生成伪造的通过结论。
