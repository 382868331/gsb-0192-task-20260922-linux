# mis_envlight — 离散环境光的多策略重要性估计

离线半球积分器：对至多 64 个离散小片的环境光场景，用多重重要性采样
（MIS, balance heuristic）估计

```
S = sum_i  L_i * f_i * cos_i * w_i
```

- **Python 版本**：3.14.7（仅用标准库：`bisect` / `math` / `random` /
  `dataclasses` / `typing`；测试用 `unittest` / `fractions`）。
- **额外依赖**：无。安装 Python 后即可运行，全部离线，无网络、账号、
  数据库、Docker 等要求。
- **平台**：Linux 原生（任何支持 CPython 的平台均可）。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试（自动发现两个测试文件）
python demo.py                            # 固定输入演示（约 0.1 秒）
```

## 接口

```python
from mis_envlight import (
    Patch, Scene, estimate, direct_sum, proposal_pdfs,
    enumerate_contributions,
    BatchSpec, BatchedEstimator, estimate_batched,
    ValidationError, DegenerateProposalError, MisEnvLightError,
)
```

- `Patch(direction, radiance, weight)` — 单个小片（frozen dataclass）。
  - `direction`：单位 3 向量 `(x, y, z)`，`z >= 0`（上半球）；
    模长容差 `NORM_TOLERANCE = 1e-9`，不做静默归一化。
  - `radiance`：有限实数，`>= 0`。
  - `weight`：有限实数，`> 0`（立体角权重）。
- `Scene(patches)` — 1..64 个 `Patch`；构造时全部校验，失败不留下
  部分构造的对象。`cos_i` 取 `direction` 的 z 分量。
- `direct_sum(scene, f) -> float` — 精确目标值（`math.fsum` 累加）。
- `estimate(scene, f, n_radiance, n_cosine, seed) -> EstimateResult` —
  MIS 估计。`f` 为长度等于小片数的有限非负数组（不是回调）；
  `n_radiance` / `n_cosine` 为两策略各自样本数（`int >= 0`，
  合计至少 1）；`seed` 为 `int`，同一种子结果完全可复现。
- `proposal_pdfs(scene)` — 两提议分布的逐小片概率。
- `enumerate_contributions(scene, f, n_radiance, n_cosine)` —
  确定性的逐（策略， 小片）贡献表，用于穷举验证期望。

`EstimateResult` 字段：`estimate`、`variance`（估计量的估计方差，
见下文）、`stddev`、`exact`、`n_radiance`、`n_cosine`、
`records`（逐样本 `SampleRecord`：策略、小片索引、两策略 pdf、
balance-heuristic 权重 `mis_weight`、目标项、贡献）、
`diagnostics`；`result.explain(limit=None)` 给出逐样本权重解释文本。

## 估计量定义

两提议分布分别为 `p_i ∝ L_i·w_i`（radiance 策略）与
`q_i ∝ cos_i·w_i`（cosine 策略）。场景契约是等立体角小片，此时与
"正比于 L / 正比于 cos" 完全等价；保留 `w_i` 使非均匀权重也正确。
balance heuristic 计入每策略样本数：小片 `i` 处单次抽样的贡献为

```
contrib_i = g_i / (n_radiance * p_i + n_cosine * q_i),   g_i = L_i f_i cos_i w_i
```

估计量为全部样本贡献之和，其期望恒等于 `S`（测试中用 `fractions.Fraction`
精确验证该恒等式，并穷举一次抽样的全部组合核对）。

## 分批自适应有限混合采样（本次新增）

每一批从**两个已有策略的有限混合**中抽样，批次 b 的条件分布为

```
pdf_b(i) = α_b · p_i + (1 − α_b) · q_i
```

- 每批结束后**允许更新下一批的混合权重** `α_b`（radiance 策略占比）；
  本库不内置学习规则，权重由调用方按上一批结果自行决定。
- **正概率下界**：`α_b ∈ [mix_floor, 1 − mix_floor]`，
  `mix_floor = DEFAULT_MIX_FLOOR = 0.05`（可自定义，必须 `0 < floor < 0.5`）。
  两个组分策略在每一批都保留严格正权重，不能被关死；`α=0`/`α=1` 被拒绝。
- **冻结实际抽样时的条件 PDF**：每条样本记录生成它那一批的 `pdf_b(i)`
  （以及 `α_b`、两策略 pdf），最终估计只读取这些冻结值，**绝不用最终
  分布回算旧样本的 pdf**。
- N 个样本的无偏估计为

```
S_hat = (1/N) · Σ_samples g_i / pdf_b(i)
```

  给定抽样历史（从而 `α_b` 已固定）时每条样本的条件期望为 `S`，因此
  自适应改权重不破坏无偏性；测试用 `Fraction` 对"两批分布不同"精确验证。

### 入口

- 一次性：`estimate_batched(scene, f, batches, seed, mix_floor=0.05)`
  - `batches`：非空序列，每项为 `(n_samples, alpha)` 或 `BatchSpec`；
    所有规格**先整体校验再抽样**，非法计划在第一条随机数之前抛错。
  - 返回 `BatchedEstimateResult`。
- 有状态（批间反馈循环）：

```python
runner = BatchedEstimator(scene, f, seed=1)
b0 = runner.run_batch(256, 0.5)          # 第一批
a1 = rule_from(b0.records)               # 调用方自定，受下界约束
runner.run_batch(256, a1)                # 分布变化的第二批
result = runner.result()
```

  失败的 `run_batch` 在改动任何状态、消耗任何随机数之前完成校验，
  因此被拒批次**不留半更新状态**，也不扰动可复现的随机流（续跑结果与
  一路合法的 runner 完全一致，有直接测试）。

### 返回结构

- `BatchSpec(n_samples, alpha)`：批次计划（frozen dataclass）。
- `BatchSampleRecord`：逐样本 `batch_index`、`draw`、`patch_index`、
  `alpha`、`pdf_radiance`、`pdf_cosine`、**`pdf_mixture`（冻结的条件 pdf）**、
  `target_value`、`contribution = g_i / pdf_mixture`。
- `BatchResult` / `BatchSummary`：`batch_index`、`n_samples`、`alpha`、
  整批冻结 `pdf`、`contribution_sum`、`mean`。
- `BatchedEstimateResult`：`estimate`、`variance`（全局均值的分层方差，
  任一批仅 1 个样本时为 `None`）、`stddev`、`exact`、`total_samples`、
  `n_batches`、`mix_floor`、`records`、`batches`、`diagnostics`；
  `result.explain(limit=None)` 输出含批次/α/冻结 pdf 的逐样本解释。

### 单组分零质量

某组分总质量为 0 时（零能量场景的 radiance，或全掠射场景的 cosine），
该组分不承载权重，混合在**仍有定义的组分上重新归一化**（此时 pdf 与 α
无关），并发 `ZERO_MASS_COMPONENT` 诊断；两组分都为零（暗且全掠射）
则构造时抛 `DegenerateProposalError`。

## 错误语义

- 数值输入拒绝 `NaN` / `±Infinity`；整数参数（样本数、种子）拒绝
  `bool`；`f` 拒绝负数、错误长度、非序列。
- 非法输入抛 `ValidationError`（同时是 `ValueError`），消息定位到字段，
  如 `patches[1].radiance: must be >= 0, got -2.0`；不做任何静默纠正。
- 某策略被请求样本但总提议质量为 0（如零能量场景请求 radiance 策略）
  抛 `DegenerateProposalError`。
- 所有校验在抽样之前完成：调用失败不会留下未说明的部分状态
  （`estimate` 本身无外部副作用）。

## 诊断（diagnostics，不中断计算）

- `ZERO_PDF_TERM`：某小片在一个活跃策略下概率为零（如 `L=0` 的暗片、
  `cos=0` 的掠射片）。目标项为零或另一策略覆盖时无害，仅作提示。
- `UNREACHABLE_TERM`：目标项非零但所有活跃策略概率均为零——估计量
  有偏。在本契约的合法输入下数学上不可达（`g_i>0` 蕴含 `L_i>0` 且
  `cos_i>0`），作为防御性检查保留并有直接测试。
- `VARIANCE_NOT_ESTIMABLE`：某活跃策略恰好 1 个样本，其方差贡献不可
  估计，此时 `variance`/`stddev` 为 `None`。
- `UNUSED_ZERO_MASS_STRATEGY`：未使用的策略总质量为零（信息性）。
- `ZERO_MASS_COMPONENT`（分批混合）：某组分策略总质量为零，混合已在
  另一组分上重归一化，`α` 此时不起作用（信息性）。

分批模式下 `ZERO_PDF_TERM` 针对的是**各批冻结的混合 pdf**：某小片在某
几批 pdf 为 0 但非每批都为 0 时仅提示；`UNREACHABLE_TERM` 表示该小片
在**每一批的冻结 pdf 下都为 0** 且目标项非零——有偏。

## 误差口径（浮点）

- `direct_sum` 与估计量累加均用 `math.fsum`（补偿求和），64 项规模下
  累加误差在 1 ulp 量级；除此外每样本贡献只有一次除法与若干次乘法。
- 期望恒等式验证：精确有理数（`Fraction`）断言严格相等；浮点穷举
  断言相对容差 `1e-12`。
- 方向模长校验容差 `1e-9`（唯一的近似型输入校验，已在上文标明）。
- 随机估计的判定不使用宽容差掩盖错误：单亮片测试用估计标准差的
  3σ 界，单片场景断言严格相等且方差为 0。

## 设计取舍

- 提议分布按 `L·w` / `cos·w` 归一（等立体角下即 "∝L / ∝cos"）。
- `f` 只接受有限数组而非回调：契约可静态校验、可穷举、可复现。
- 抽样用 `random.Random(seed)` + 累积分布二分，跨运行确定性复现。
- 方差按策略分层估计：`var = Σ_s n_s · s_s²`（`s_s²` 为该策略样本贡献
  的无偏样本方差）；任一活跃策略只有 1 个样本时整体报告 `None`，
  不给出虚假精度。
- 分批模式的方差按批分层估计全局均值：
  `var = (1/N²) · Σ_b n_b · s_b²`（各批用其**冻结 pdf** 下的贡献）；
  任一批只有 1 个样本时整体报告 `None`。
- 失败即抛异常与诊断列表分离：致命契约违例（非法输入、退化提议）
  抛异常；采样可继续的异常结构（零概率项等）进入 `diagnostics`。

## 已知限制

- 仅上半球（`z >= 0`），`cos` 取方向 z 分量，不支持自定义法线。
- `L`、`f` 均要求非负；负值被积函数不在本契约内。
- 小片数上限 64；无分层/低差异序列等更高级采样；方差在单样本
  策略/批次下不可估计（报 `None`）。
- 分批混合只允许**已有的两个策略**作为组分（有限混合），不新增策略、
  不内置权重学习规则；`mix_floor` 固定两组分的正概率下界，
  不支持把任一组分权重降到 0。
- 只实现本题契约的数学子集，不宣称兼容任何渲染工业标准。
