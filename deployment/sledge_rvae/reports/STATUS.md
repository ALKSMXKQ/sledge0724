# 当前验证状态

本部署代码是在非目标 GPU 环境整理的。当前主机系统 CUDA Toolkit 为 11.5，NVIDIA driver/GPU 不可用；因此没有填写伪造的 Engine、TensorRT 误差或性能数字。独立 `.venv` 已安装 ONNX 1.16.1、ONNX Runtime 1.18.1、TensorRT 8.6.1 Python/核心库和 CUDA runtime 12.3.101，全部 Python import 检查通过。Builder 创建准确停止在 CUDA error 100（没有可用 GPU/driver）。

本机已完成的真实检查：从 nuPlan-derived `sledge_raw.gz` 生成固定 `1×12×256×256` 输入和 PyTorch 参考；部署包装层与原 decoder 的确定性 latent-mean 路径在全部 14 个输出上逐元素相等（max/mean absolute error 均为 0）。

ONNX 已实际生成（opset 17，207,350,191 bytes，SHA-256 `6c3d96e04c6af9a1ac641cbf86319f44fce8f0ffb53fddb484deadf841f60dce`）并通过结构校验。PyTorch–ONNX Runtime：全局 max absolute error `3.5375357e-4`，mean absolute error `2.1180532e-5`，六类后处理 active-query IoU 全部为 `1.0`，FP32 验收通过。ONNX 已消除 `ScatterND`、`CumSum`、控制流和自定义 domain，TensorRT 定向静态预检通过。

完整 C++ 程序已使用 TensorRT 8.6.1 官方头文件、8.6.1 动态库和 CUDA runtime 12.3.101 头文件/库成功编译链接；这证明 C++ API 兼容性，但不代替目标 GPU 运行。

部署虚拟环境有意通过 `--system-site-packages` 复用原 SLEDGE 环境中与 checkpoint 匹配的 PyTorch/nuPlan 依赖；PyTorch 导出和 ONNX Runtime 验证均在 CPU 完成。`pip check` 会显示原工程遗留的包元数据版本冲突，以及 PyTorch CUDA 12.1 wheel 与新增 TensorRT CUDA 12.3 runtime wheel 并存。已验证的 import、模型加载、ONNX 导出与 ONNX Runtime 结果不受影响；目标机 C++/TensorRT 流程只加载 CUDA 12.3 和 TensorRT 8.6.1，不依赖 PyTorch CUDA 库。

提交前必须在 CUDA 12.3 + TensorRT 8.6.1 目标 GPU 上执行 `scripts/run_target_pipeline.sh`，并将以下真实产物随包提交：

- `artifacts/sledge_rvae.onnx` 与 manifest；
- `artifacts/sledge_rvae_fp32.engine`，如通过验证再附 FP16 Engine；
- `artifacts/validation/sample_000/`；
- `reports/build_fp32.log`、`environment.txt`；
- `reports/consistency.json/.md`；
- `reports/performance_report.json/.md` 和 `gpu_samples.csv`。
