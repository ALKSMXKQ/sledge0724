# 遮挡物后行人冲出：统一全链路

本目录集中保存“一句自然语言 → EventFrame → 参数化语义模板 → B1 场景编辑 → R1 RVAE 重建 → B2 扩散重绘 → 闭环仿真与评估”的全部新增代码、配置、测试和说明。它不会替换或复制原有训练数据；运行产物统一写入命令指定的 `--output-root`。

## 1. 设计边界

当前只支持一个明确的危险场景族：**行人被遮挡物遮住，随后横向冲入 ego 路径**。普通无视线遮挡的行人横穿不会静默降级进入本链路。

场景多样性由四个可解释轴构成：

1. 遮挡物：`vehicle / bicycle / generic_object / traffic_cone / barrier / czone_sign`。这些值与模拟器可见的 `TrackedObjectType` 一一对应；`car / van / truck / bus` 仅作为自然语言同义词，统一归一化为 `vehicle`，不再制造不可观察的伪多样性。
2. 行人方向：`left_to_right / right_to_left`
3. 行人速度：当前 RVAE 允许的 `0.5–2.0 m/s`，论文矩阵默认取 `1.2 / 1.6 / 1.9`
4. 原始场景类型：从已有 cache 的 scenario type 中分层轮询取样

不额外随机改变遮挡关系的核心几何约束。遮挡物必须处在 ego—行人视线之间，行人必须朝 ego 车道移动，初始状态不能产生参与者重叠。这样，多样性来自明确控制变量和底图，而不是不可解释的随机扰动。

## 2. 链路结构

```text
自然语言
  ↓ EventFrame 解析、事件顺序构造、校验/修复
EventFrame
  ↓ 严格适配（prompt/override/EventFrame/default 均记录 provenance）
HazardSemanticSpec
  ↓ primitive compiler
可执行操作序列
  ↓ 原 compositional editor / executor
B1 参数化危险场景
  ↓ B0 不变量检查：只能新增 1 个行人和 1 个遮挡物
可仿真 B1 sledge_vector.gz
  ↓ RVAE encoder(mu) / decoder
R1 原始重建候选 + 语义保护重建 sledge_vector.gz
  ↓ 从 B1 latent 启动 half-denoise + 危险语义硬约束
B2 half-denoise 扩散重绘
  ↓ 精确字段、语义指标、对象类型和模拟器读取四重检查
通过样本的 B2 sledge_vector.gz
  ↓ nuPlan/SLEDGE closed-loop simulation
仿真指标与阶段对比
```

这里的遮挡场景生成沿用项目历史版本中已经验证过的 compositional editor：行人横穿位置、速度和方向由模板控制；遮挡物沿 ego 到行人的视线放置，并包含 frame-0 时间偏移补偿、与 ego/行人/其他 agent 的碰撞规避。新代码主要增加 EventFrame 严格适配、统一编排、实验矩阵和 B0/B1/B2 同口径评估，没有另造一套遮挡几何方法。

由于当前 RVAE 在编码/解码新插入的小行人时可能直接丢掉该实体，低噪声解码也可能移动遮挡物或扭曲道路，严格模式会保留模型的原始候选用于审计，再把经过验证的 B1 道路、ego、目标行人和遮挡物精确合成回可仿真输出。背景交通仍可由模型改变，但最终还必须通过完整遮挡关系、冲出方向、速度、到达时序、无重叠和 corridor clearance 检查。这个保证属于“生成 + 语义投影 + 拒绝门控”的系统保证，不代表未经保护的 RVAE/扩散模型天然学会了该语义；原始候选通过率会单独记录。

可变长度 raw cache 在编辑前会把所有原始数据行视为已占用实体，并只在末尾增加临时插入槽。这样可以兼容“states 有数据但原始 mask 全 False”的历史 cache，避免旧逻辑覆盖第一行并在压缩时删除全部原始对象。保存前只删除新建但未激活的槽并重映射索引。

## 3. 目录说明

```text
occluded_pedestrian_pipeline/
├── cli.py                     # 唯一命令入口
├── pipeline.py                # B0/B1/B2 编排与结构化产物
├── experiment_matrix.py       # 可复现实验矩阵和分层底图取样
├── configs/                   # 默认模板与实验配置
├── language/                  # EventFrame → HazardSemanticSpec
├── generation/                # 原 compositional editor 与 primitives
├── evaluation/                # B0/B1/B2 指标、可视化、闭环仿真
└── tests/                     # 单元测试
```

## 4. 建议执行顺序

在仓库根目录执行：

```bash
cd /home16T/home8T_1/leitingting/sledge_workspace/sledge
SLEDGE_PYTHON=/home/leitingting/anaconda3/envs/sledge/bin/python
```

先跑单样本，检查语言结构、几何校验和 B0/B1 对比图：

```bash
$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli single \
  --input-raw ../exp/caches/autoencoder_cache/<log>/<scenario_type>/<token>/sledge_raw.gz \
  --output-root ../exp/occluded_pedestrian_runs/single \
  --prompt "A pedestrian hidden behind a road barrier crosses from right to left at pedestrian speed 1.6 m/s." \
  --occluder barrier --direction right_to_left --speed 1.6
```

然后跑 20 个底图的固定语义 debug。该 profile 只固定遮挡物/方向/速度，专门验证不同原始场景是否都能稳定编辑：

```bash
$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli batch \
  --input-root ../exp/caches/autoencoder_cache \
  --output-root ../exp/occluded_pedestrian_runs/debug20 \
  --profile debug20
```

B1 通过后先保存 RVAE 重建候选和受保护重建，再运行 B2 扩散重绘：

```bash
$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli reconstruct \
  --run-root ../exp/occluded_pedestrian_runs/debug20 \
  --config ../semantic_img2img_cfg.yaml --device cuda --max-scenes 20

$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli refine \
  --run-root ../exp/occluded_pedestrian_runs/debug20 \
  --device cuda --max-refine-scenes 20 --mode semantic_protected

$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli audit-gz \
  --run-root ../exp/occluded_pedestrian_runs/debug20
```

若当前只验收前半链路，可直接运行 100 个不同训练底图的 `pilot100`，随后导出模拟器可读的带类型 gzip cache：

```bash
$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli batch \
  --input-root ../exp/caches/autoencoder_cache \
  --output-root ../exp/occluded_pedestrian_runs/pilot100_nuplan_types_v1 \
  --profile pilot100 \
  --target-accepted 100 \
  --candidate-pool-size 800 \
  --control-mode controlled \
  --accept-defaults

$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli export-b1 \
  --run-root ../exp/occluded_pedestrian_runs/pilot100_nuplan_types_v1 \
  --config ../semantic_img2img_cfg.yaml
```

`sledge_vector.gz` 内嵌 `__sledge_object_type_overrides__`，更新后的 `SledgeScenario` 会据此恢复 `VEHICLE/BICYCLE/GENERIC_OBJECT/TRAFFIC_CONE/BARRIER/CZONE_SIGN`。旧 reader 会忽略该扩展字段，仍可读取基础向量。

最后可单独检查 B2 仿真命令；正式验收应运行三阶段对比，它会分别模拟 B0、B1、B2，并生成轨迹图、代表性 GIF、碰撞对象审计和指标对比图：

```bash
$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli simulate \
  --run-root ../exp/occluded_pedestrian_runs/debug20 --stage b1 --dry-run

$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli simulate \
  --run-root ../exp/occluded_pedestrian_runs/debug20 --stage rvae --limit 20

$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli simulate \
  --run-root ../exp/occluded_pedestrian_runs/debug20 --stage b2 \
  --mode semantic_protected --limit 20

$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli compare-stages \
  --run-root ../exp/occluded_pedestrian_runs/debug20 \
  --config ../semantic_img2img_cfg.yaml --limit 20
```

若希望顺序执行全部阶段，可使用 `all`。它依次执行批量模板编辑、B1 仿真 cache 导出、RVAE 重建、两种扩散对照、gzip 总审计和闭环仿真。调试时可用 `--skip-b1-export`、`--skip-rvae-reconstruction`、`--skip-refine` 或 `--skip-simulation` 设置阶段门控。

```bash
$SLEDGE_PYTHON -m sledge.semantic_control.occluded_pedestrian_pipeline.cli all \
  --input-root ../exp/caches/autoencoder_cache \
  --output-root ../exp/occluded_pedestrian_runs/pilot100_full \
  --profile pilot100 --target-accepted 100 --candidate-pool-size 800 \
  --config ../semantic_img2img_cfg.yaml --device cuda \
  --max-refine-scenes 100 --simulation-limit 100
```

## 5. 正式多样性矩阵

- `debug20`：1 组控制条件 × 20 个分层底图，用于先验证全链路。
- `pilot100`：6 个可见遮挡类别、2 个方向、3 档速度，在 100 个不同训练底图上近似均衡取样。
- `paper18`：3 个可见遮挡类别 × 2 方向 × 3 速度 = 18 个条件，每条件 10 个底图，共 180 个样本。
- `extended24`：4 个可见遮挡类别 × 2 方向 × 3 速度 = 24 个条件，每条件 10 个底图，共 240 个样本。

建议顺序是 `single → debug20 → pilot100 → paper18`。不要在 B1 语义通过率尚不稳定时直接增加更多场景类型或提示词表达。

## 6. 每次运行的结构化产物

```text
<output-root>/
├── b0_original_cache/<sample_id>/sledge_raw.gz
├── b1_edited_cache/<sample_id>/
│   ├── sledge_raw.gz
│   ├── scenario_label.json
│   ├── edit_report.json
│   └── semantic_report.json
├── b1_simulation_cache/log/sudden_pedestrian_crossing/<sample_id>/
│   ├── sledge_vector.gz       # 内嵌模拟器对象类型
│   └── scenario_label.json
├── rvae_reconstruction/
│   ├── candidate_cache/log/sudden_pedestrian_crossing/<sample_id>/sledge_vector.gz
│   ├── generated_cache/log/sudden_pedestrian_crossing/<sample_id>/sledge_vector.gz
│   └── reports/<sample_id>/
├── b2_diffusion/
│   ├── raw_diffusion_baseline/{reports,generated_cache}/...
│   └── semantic_protected/{reports,generated_cache}/...
├── stage_vector_caches/       # B0/B1 的仿真接口适配缓存
├── stage_simulations/{B0,B1,B2}/
│   └── simulation/            # 各阶段独立闭环日志和 parquet 指标
├── visualizations/
│   ├── stage_scenes/          # 每个样本的 B0/B1/B2 三联图
│   ├── b1_typed_cache/        # B1 模拟缓存的逐场景类型预览和六类总览
│   ├── generated_gzip_stages/ # B1/R1/B2 可仿真 gzip 三联图
│   ├── simulation_trajectories/{B0,B1,B2}/
│   └── metric_comparison/     # 三阶段指标 CSV、综合指标图、接触对象分类图
├── artifacts/<sample_id>/
│   ├── 01_language/           # prompt、EventFrame、校验与参数来源
│   ├── 02_specification/      # HazardSemanticSpec、primitive ops
│   ├── 03_editing/            # 编辑报告、严格校验、B0/B1 对比图
│   ├── 04_evaluation/         # B0/B1/B2 同口径指标
│   └── sample_summary.json
└── manifests/
    ├── cases.jsonl
    ├── b1_results.jsonl/.csv
    ├── b1_summary.json
    ├── rvae_reconstruction_summary.json
    ├── b2_{raw_diffusion_baseline,semantic_protected}_summary.json
    ├── generated_gzip_stages.json/.csv
    ├── stage_simulation_comparison.json
    ├── stage_metrics_comparison.json
    └── {b0,b1,b2}_contact_targets.json
```

每个样本的 `adaptation.json` 会标记关键参数来自 `control_override`、`prompt_evidence`、`eventframe` 还是 `deterministic_default`，可用于检查自然语言控制是否真的生效。

## 7. 阶段验收标准

- 语言阶段：EventFrame 和映射 spec 校验通过；遮挡物、方向、速度没有被错误填入 ego 参数。
- B1：先检查 B0 的道路、交通灯和 ego 完全不变，再逐行检查所有原始车辆、行人和静态物体仍存在且数值不变；只允许新增 1 个目标行人和 1 个遮挡物。之后再检查 LOS 遮挡、方向、速度、到达时序、初始无碰撞和 `1.50 m` ego 动态扫掠余量。ego 速度不再为了凑 TTC 被修改；不合适的原场景会被拒绝并自动换底图。
- diffusion input：B1 保存前先检查固定容量（车辆 50、行人 20、静态物体 30）足以容纳 B0 和两个新增实体；固定容量转换后再做同一套逐实体检查。若容量截断会挤掉任何 B0 实体，该原场景会直接被拒绝并换底图。
- R1：保留未经保护的 RVAE decoder 候选作模型能力审计；另将 B1 道路、ego、目标行人和遮挡物恢复到重建结果，重新检查完整语义并通过 `SledgeScenario` 读取后，才写入 `rvae_reconstruction/generated_cache`。
- B2：扩散解码完成后，将上述危险语义元素硬约束为 B1，重新检查 shape/mask/数值、完整遮挡语义、对象类型元数据和模拟器读取；失败文件会从 `semantic_protected/generated_cache` 移除。
- 仿真：B0/B1/B2 使用同一配置分别运行；速度不高于 0.1 m/s 的停放车辆保持静止且保留 `VEHICLE` 类型，避免 IDM 将遮挡车错误吸附到车道并加速。统计碰撞、TTC、可行驶区域、进度和舒适性，同时按 track token 区分“目标行人/遮挡物/其他车辆”的真实接触对象。限速指标因要求生成地图上唯一 lane 匹配而不适用，本链路只禁用该一项。

服务器工作盘为 NFS 时，nuPlan 的 `aiofiles` 可能在 NuBoard/metric 文件上阻塞。统一入口会在 `/tmp` 完成仿真后自动复制到 `<output-root>/stage_simulations/<stage>/simulation`；并且会核对实际指标场景数，子进程返回 0 但结果不完整时仍判为失败。

## 8. 测试

```bash
$SLEDGE_PYTHON -m pytest -q \
  sledge/semantic_control/occluded_pedestrian_pipeline/tests
```

测试覆盖显式参数优先级和来源追踪、RVAE 速度边界、primitive 编译、debug20 重现性与底图分层、18 条件/180 样本矩阵计数。
