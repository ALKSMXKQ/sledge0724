本次主要修改(创建树状参数表)

1. 新增真正的递进式参数树

新增文件：
```text
sledge/semantic_control/language/hierarchical_ontology.py
```
其中 `tree` 是真实嵌套的单路径树，不是原先的并列参数字典。

2. 增加父子节点合法性约束

代码中已经定义了以下约束表：

```text
道路拓扑 → 允许的交通空间
交通空间 → 允许的主体大类
主体大类 → 允许的具体主体
具体主体 → 允许的危险交互
危险交互 → 允许的辅助实体
危险交互 → 允许的来源和目标区域
危险交互 → 允许的触发事件
危险交互 → 允许的自车响应
```

3. 新增层次化参数补全

新增文件：
```text
sledge/semantic_control/language/hierarchical_pipeline.py
```

缺失参数不再只根据单个扁平槽位补全，而是根据已选中的完整父路径补全。

4. 保留旧代码兼容性

原有以下输出全部保留：

```text
semantic_slots
actor_layer
interaction_layer
motion_layer
event_layer
object_layer
road_layer
risk_layer
parameter_layer
```

只是在此基础上新增：

```text
hierarchy_layer
```

因此现有评测和后续生成代码不需要立即全部重写。

新增默认入口：

```python
from sledge.semantic_control import DefaultLanguageUnderstandingPipeline

pipeline = DefaultLanguageUnderstandingPipeline()
frame, spec = pipeline.parse_to_spec(prompt)
```

也可以显式使用：

```python
from sledge.semantic_control.language import HierarchicalEventFramePipeline
```

## 新增运行入口

单条自然语言：

```bash
python -m sledge.script.language.run_hierarchical_language_pipeline \
  --prompt "A child suddenly emerges from behind a parked truck into the ego lane."
```

批量 JSONL：

```bash
python -m sledge.script.language.run_hierarchical_language_pipeline \
  --input-jsonl cases.jsonl \
  --output hierarchical_results.jsonl
```

也保留了兼容入口：

```bash
python -m sledge.script.run_hierarchical_language_pipeline \
  --prompt "A child suddenly emerges from behind a parked truck into the ego lane."
```

## 新增测试

测试文件：

```text
sledge/semantic_control/language/tests/test_hierarchical_pipeline.py
```

包含 7 个测试，覆盖：

1. 遮挡儿童冲出场景的完整树路径；
2. 拒绝“行人加塞”等非法父子关系；
3. 根据完整父路径补全参数；
4. 明确参数不被层次先验覆盖；
5. 左右来源为候选列表时不会报错；
6. 环岛入口不会被误判为普通车辆加塞；
7. 新树结构与旧 HazardSpec 层兼容。

你可以在服务器运行：

```bash
python -m pytest \
  sledge/semantic_control/language/tests/test_hierarchical_pipeline.py \
  -vv
```
这次 PR 一共涉及 **9 个文件**。

## 新增的文件（5个）

```text
sledge/semantic_control/language/hierarchical_ontology.py
sledge/semantic_control/language/hierarchical_pipeline.py
sledge/semantic_control/language/tests/test_hierarchical_pipeline.py
sledge/script/language/run_hierarchical_language_pipeline.py
sledge/script/run_hierarchical_language_pipeline.py
```

作用分别是：

* `hierarchical_ontology.py`：树状参数体系、父子约束、路径解析与合法性验证。
* `hierarchical_pipeline.py`：将现有 EventFrame 流程接入层次树，并完成路径条件参数补全。
* `test_hierarchical_pipeline.py`：层次路径、非法关系、参数继承和兼容性测试。
* `script/language/run_hierarchical_language_pipeline.py`：正式命令行入口。
* `script/run_hierarchical_language_pipeline.py`：兼容旧脚本目录结构的转发入口。

## 修改的文件（4个）

```text
sledge/semantic_control/language/__init__.py
sledge/semantic_control/__init__.py
sledge/semantic_control/language/README.md
sledge/script/language/README.md
```

具体修改：

* `language/__init__.py`

  * 导出 `HierarchicalSceneResolver`
  * 导出 `HierarchicalEventFramePipeline`
  * 导出层次化验证和补全相关类

* `semantic_control/__init__.py`

  * 增加默认语言理解流水线的延迟导入
  * 让外部可通过以下方式调用：

```python
from sledge.semantic_control import DefaultLanguageUnderstandingPipeline
```

完整遮挡行人冲出后续链路共新增 11 个工程文件：

sledge/semantic_control/occluded_pedestrian_pipeline/
├── configs/
│   └── occluded_language_cases.jsonl
├── language/
│   ├── occluded_prompt_matrix.py
│   └── hierarchical_template_validator.py
├── generation/
│   ├── diffusion_modes.py
│   ├── hierarchical_template_sampler.py
│   └── hierarchical_spec_adapter.py
├── evaluation/
│   └── diffusion_semantic_retention.py
└── tests/
    ├── test_occluded_prompt_matrix.py
    ├── test_hierarchical_template_validator.py
    ├── test_hierarchical_template_sampler.py
    └── test_diffusion_semantic_retention.py

主要作用：

* 生成 12 条遮挡行人自然语言测试；
* 自动验收层次参数模板；
* 将参数范围确定性采样为具体控制参数；
* 根据遮挡物侧唯一派生实际执行方向；
* 将新层次模板转换为现有 `HazardSemanticSpec`；
* 分别统计原始扩散和语义保护扩散的危险语义保持率。

修改文件

共提供 9 个完整替换文件：

sledge/semantic_control/occluded_pedestrian_pipeline/
├── cli.py
├── pipeline.py
├── experiment_matrix.py
├── configs/
│   └── experiment_matrix.json
├── language/
│   ├── __init__.py
│   └── eventframe_adapter.py
├── generation/
│   ├── __init__.py
│   └── refinement_runner.py
└── evaluation/
    └── __init__.py



## 本次实现的关键逻辑

语言层只保留相对方向：

```text
motion_direction = occluder_to_ego_path
```

左右不再是两个语义方向，而是遮挡物位置：

```text
occluder_side = left
→ concrete_direction = left_to_right

occluder_side = right
→ concrete_direction = right_to_left
```

行人类别统一为：

```text
primary_actor_type = pedestrian
tracked_object_type = TrackedObjectType.PEDESTRIAN
sledge_collection = pedestrians
```

儿童、跑步者等仅保存在：

```text
language_actor_detail
```

## 两种扩散模式

### 原始扩散基线

```text
raw_diffusion_baseline
```

该模式：

* 不回填行人；
* 不回填遮挡物；
* 不回填道路和 ego 状态；
* 关闭 ROI 保护；
* 只进行一次生成；
* 不通过多次修复筛选语义更好的结果；
* B2 评估时重新搜索对象，不依赖原 slot index。

它用于回答：

> 当前扩散模型本身能否保留遮挡物后行人冲出的危险语义？

### 语义保护对照

```text
semantic_protected
```

该模式保留已有道路、行人、遮挡物和 ego 的精确合成机制，只作为对照实验。


## 第一步：生成和测试 12 条语言输入

```bash
cd /home16T/home8T_1/leitingting/sledge_workspace/sledge

python -m sledge.semantic_control.occluded_pedestrian_pipeline.cli \
  generate-prompts \
  --output /tmp/occluded_language_cases.jsonl
```

生成参数模板并验收：

```bash
python -m sledge.semantic_control.occluded_pedestrian_pipeline.cli \
  validate-prompts \
  --input-jsonl /tmp/occluded_language_cases.jsonl \
  --output-jsonl /tmp/hierarchical_template_results.jsonl
```

## 第二步：生成 24 个 B1 修改场景

每条语言输入绑定两个不同原始场景：

```bash
python -m sledge.semantic_control.occluded_pedestrian_pipeline.cli \
  batch-language \
  --input-root ../exp/caches/autoencoder_cache \
  --output-root ../exp/occluded_pedestrian_runs/language24 \
  --prompt-jsonl /tmp/occluded_language_cases.jsonl \
  --scenes-per-prompt 2
```

参数采样完成时只标记：

```text
concrete_parameter_sample_ready = true
sampled_scene_ready = false
```

只有 B1 编辑和严格几何检查通过后，才会标记：

```text
sampled_scene_ready = true
```

## 第三步：运行原始扩散基线

```bash
python -m sledge.semantic_control.occluded_pedestrian_pipeline.cli \
  refine \
  --run-root ../exp/occluded_pedestrian_runs/language24 \
  --mode raw_diffusion_baseline \
  --device cuda
```

## 第四步：运行语义保护对照

```bash
python -m sledge.semantic_control.occluded_pedestrian_pipeline.cli \
  refine \
  --run-root ../exp/occluded_pedestrian_runs/language24 \
  --mode semantic_protected \
  --device cuda
```

也可以一次运行两种模式：

```bash
python -m sledge.semantic_control.occluded_pedestrian_pipeline.cli \
  refine \
  --run-root ../exp/occluded_pedestrian_runs/language24 \
  --mode both \
  --device cuda
```

## 最终指标

每个扩散结果都会统计：

```text
pedestrian_retained
occluder_retained
occlusion_retained
unique_direction_retained
ego_path_intersection_retained
reveal_event_proxy_retained
interaction_timing_retained
full_hazard_semantics_retained
```

最重要的是：

```text
full_hazard_semantics_retained
```

只有行人、遮挡物、初始遮挡、正确方向、进入自车路径和交互时序同时保留时，该指标才为 `true`。

结果位于：

```text
<run-root>/manifests/
├── b2_raw_diffusion_baseline_retention.jsonl
├── b2_raw_diffusion_baseline_summary.json
├── b2_semantic_protected_retention.jsonl
├── b2_semantic_protected_summary.json
└── diffusion_semantic_retention_comparison.json
```
