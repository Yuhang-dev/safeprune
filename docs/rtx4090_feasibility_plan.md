# 两张 RTX 4090 上的 Agent FFN 动态剪枝可行性计划

本文档面向 2 x RTX 4090。远端配置保持不变，本地和远端尽量使用 smoke 已验证版本：

```text
torch 2.6.0+cu124
transformers 4.44.2
datasets 2.21.0
peft 0.12.0
trl 0.10.1
accelerate 0.34.2
swanlab 0.8.1
```

## 0. 执行结论

当前优先级：

1. 复现 SliceGPT，建立真实结构压缩和效率测量基线。
2. 在本项目中实现 Qwen2.5 的 FFN-only stage mask bank。
3. 验证 stage-aware 和 failure-triggered re-densification 是否优于固定 mask。
4. 再引入 Probe Pruning 作为 hidden-state-only 动态剪枝基线。

旧 SafePrune-DPO 4090 smoke 不再作为主线，但保留为排雷记录：attention head pruning 在当前 setup 下高度不稳定，mask-only 不能声称真实 speedup。

## 1. GPU 分工

| GPU | 任务 |
|---|---|
| GPU 0 | SliceGPT OPT 系列、静态 mask、mask bank calibration。 |
| GPU 1 | Phi-2、Qwen Agent 评测、失败恢复实验。 |

两张 4090 不当作一个 48 GB 统一显存池，优先跑两个独立进程。

## 2. 本地 CPU 验证

本地没有 GPU 时，只跑 CPU 单元测试和 dry-run，不加载 Qwen 模型：

```powershell
conda create -n prune-agent-cpu python=3.10 -y
conda activate prune-agent-cpu
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.44.2 datasets==2.21.0 peft==0.12.0 trl==0.10.1 accelerate==0.34.2 PyYAML swanlab==0.8.1 pytest ruff
pip install -e .

python -m unittest discover -s tests
python scripts/build_stage_mask_bank.py --config configs/agent_qwen2_5_1_5b_4090.yaml --skip-model-load
python scripts/evaluate_agent_masks.py --config configs/agent_qwen2_5_1_5b_4090.yaml --dry-run
```

需要模型加载、activation scoring 和真实 generation 的步骤全部放到远端 4090。

## 3. 远端阶段

### 第 1 阶段：SliceGPT 复现

模型：

```text
facebook/opt-125m
facebook/opt-1.3b
facebook/opt-2.7b
microsoft/phi-2
```

稀疏率：

```text
0%, 10%, 20%, 25%, 30%
```

指标：

```text
WikiText-2 PPL
PIQA/ARC-Easy
model size
peak GPU memory
tokens/s
```

命令见 `docs/slicegpt_repro_commands.md`。

### 第 2 阶段：Qwen FFN mask MVP

生成受控 Agent 任务：

```bash
python scripts/prepare_agent_tasks.py \
  --output data/agent/controlled_tasks.jsonl \
  --count 1000
```

构建 1.5B stage mask bank：

```bash
python scripts/build_stage_mask_bank.py \
  --config configs/agent_qwen2_5_1_5b_4090.yaml \
  --max-calibration-batches 256
```

本阶段只报告 active FFN ratio 和任务指标，不报告真实加速。

### 第 3 阶段：Agent 动态路由

比较：

| 方法 | 说明 |
|---|---|
| Dense | 不剪枝。 |
| Static Global Mask | 全轨迹同一个 FFN mask。 |
| Stage Mask | 按 `plan/act/observe/reflect/answer` 切换。 |
| Failure Re-densification | 失败事件后临时恢复到 10% sparsity。 |

主指标：

```text
task success rate
tool-call validity
recovery success rate
active FFN ratio
latency
cost per successful task
```

### 第 4 阶段：扩展到 3B 和动态基线

在 Qwen2.5-3B 上重复第 2-3 阶段。随后跑 Probe Pruning 官方代码，建立 hidden-state-only 动态剪枝对照。

## 4. 止损线

| 风险 | 处理 |
|---|---|
| 1.5B dense Agent 成功率太低 | 切到 3B 作为主模型。 |
| stage mask 不优于 static mask | 转向 failure-conditioned / tool-risk-conditioned 路由。 |
| re-densification 频繁抖动 | 使用 step-level 切换、滞回和最短保持窗口。 |
| mask-only 无真实加速 | 第一篇只写算法 frontier；系统加速留给 compact FFN 或 kernel。 |

## 5. 产物

每次远端运行记录：

```text
resolved config
commit hash
GPU 型号和显存
依赖版本
任务数量
mask bank 路径
metrics JSON
SwanLab run URL
失败日志
```
