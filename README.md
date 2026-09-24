# mis_envlight — 离散环境光的多策略重要性估计

离线半球积分器：对至多 64 个离散小片的环境光场景，用多重重要性采样
（MIS, balance heuristic）估计

```
S = sum_i  L_i * f_i * cos_i * w_i
```

支持两种用法：单轮 `estimate`，以及**多批自适应** `estimate_adaptive`——
允许每批结束后更新下一批两策略的采样配比，每个样本按抽样时的条件
PDF 计分。

- **Python 版本**：3.14.7（仅用标准库：`bisect` / `math` / `random` /
  `dataclasses` / `typing`；测试用 `unittest` / `fractions`）。
- **额外依赖**：无。安装 Python 后即可运行，全部离线，无网络、账号、
  数据库、Docker 等要求。
- **平台**：Linux 原生（任何支持 CPython 的平台均可）。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 全部测试（旧 25 + 新 33 = 58）
python demo.py                            # 固定输入演示（约 0.1 秒）
```

## 接口

```python
from mis_envlight import (
    Patch, Scene, estimate, direct_sum, proposal_pdfs,
    enumerate_contributions,
    estimate_adaptive, variance_minimizing_updater,
    BatchRecord, AdaptiveResult,
    MIN_STRATEGY_WEIGHT, MAX_BATCHES,
    DIAG_WEIGHTS_UPDATED, DIAG_WEIGHTS_CLAMPED,
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

## 多批自适应估计 `estimate_adaptive`

```python
estimate_adaptive(scene, f, n_batches, samples_per_batch, seed, *,
                  initial_weights=(0.5, 0.5),
                  floor=MIN_STRATEGY_WEIGHT,   # 1e-3
                  updater=None) -> AdaptiveResult
```

- 把 `n_batches` 个**独立批次**的估计取无加权平均；每批是一次普通的
  两策略 balance-heuristic MIS，各抽 `samples_per_batch >= 2` 个样本；
  `n_batches` 范围 `[1, MAX_BATCHES]`（`MAX_BATCHES = 1024`）。每批的
  期望都是 `S`，所以批间无论怎样改配比，`(1/B)·Σ_b E_b` 恒无偏。
- **采样权重只允许在两个已有策略间分配**：`initial_weights` 是一对
  正比值 `(w_radiance, w_cosine)`，内部归一化为份额；每批按份额把
  整数样本数分给两策略（四舍五入，各自至少 1 个）。
- **正概率下界 `floor`**（`(0, 0.5]`，默认 `MIN_STRATEGY_WEIGHT=1e-3`）：
  归一化后每个策略份额必须 `>= floor`，因此两策略每批都活跃。初始
  权重违例直接报错；updater 返回违例但形式合法的值会被**夹到
  `[floor, 1-floor]`** 并产生 `WEIGHTS_CLAMPED` 诊断（不是静默纠正）。
  由于两策略每批都活跃，任一提议总质量为零的场景在抽样前就被拒绝。
- `updater(BatchRecord) -> (w_radiance, w_cosine)` 在每批结束后
  （最后一批除外）被调用，决定**下一批**的配比。返回畸形值
  （非序列、长度不对、NaN/Inf、负、双零）抛 `ValidationError` 并中止
  整次运行，不返回任何半更新结果。`updater=None` 表示固定分布。
- 内置 `variance_minimizing_updater()`：按上一批两策略单样本贡献的
  经验标准差之比 `w_s ∝ σ_s` 分配下一批；单样本层或零方差时保持原
  配比。它是启发式，不宣称全局最优。

**样本保存抽样时的条件 PDF，绝不用最终分布回算旧样本。**
每个 `SampleRecord` 在所属批次抽样时冻结该批的 `n_self/n_other` 与
两策略 PDF（PDF 本身由场景固定，配比按批冻结），聚合只读取这些冻结
值。测试用 `Fraction` 给出一个精确反例：两片场景，批 1 用 `(1,3)`
抽样、批 2 用 `(3,1)`；若把批 1 的样本用最终配比 `(3,1)` 重新评分，
平均估计量变成 `72/35`（真值 `S=2`，偏差 `2/35`），证明回算必有偏；
实现采用的逐批条件评分严格等于 `S`。

`AdaptiveResult` 字段：`estimate`、`variance` / `stddev`（批均值的
估计方差 `(1/B²)·Σ_b Var(E_b)`，任一批出现单样本层时为 `None`）、
`exact`、`n_batches`、`samples_per_batch`、`floor`、`initial_weights`、
`final_weights`、`batch_records`（逐批 `BatchRecord`：冻结的配比、份额、
批估计、批方差与该批 `records`）、扁平的 `records`（每个 `SampleRecord`
带 `batch` 与 `n_other`，并有 `mixture_pdf()` 给出抽样时的条件混合
PDF）、`diagnostics`；`result.explain(limit=None)` 返回逐批分配表。

兼容性：`n_batches=1` 且 50/50 初始权重时，抽样顺序与逐样本贡献和旧
`estimate` 逐位一致（有严格相等测试）。

## 错误语义

- 数值输入拒绝 `NaN` / `±Infinity`；整数参数（样本数、种子）拒绝
  `bool`；`f` 拒绝负数、错误长度、非序列。
- 非法输入抛 `ValidationError`（同时是 `ValueError`），消息定位到字段，
  如 `patches[1].radiance: must be >= 0, got -2.0`；不做任何静默纠正。
- 某策略被请求样本但总提议质量为 0（如零能量场景请求 radiance 策略）
  抛 `DegenerateProposalError`。
- 所有校验在抽样之前完成：调用失败不会留下未说明的部分状态
  （`estimate` 本身无外部副作用）。
- `estimate_adaptive` 同样**先全部校验再抽样**：批次数、每批样本数、
  `floor`、`initial_weights`、`seed`、`updater` 可调用性、以及两策略
  退化检查都在第一个样本产生前完成；运行中 updater 返回非法值时整次
  运行抛错中止，已抽的批次作为函数内局部状态随异常丢弃，不返回半更新
  结果（`Scene` 本身不可变，也无其他外部状态变更）。

## 诊断（diagnostics，不中断计算）

- `ZERO_PDF_TERM`：某小片在一个活跃策略下概率为零（如 `L=0` 的暗片、
  `cos=0` 的掠射片）。目标项为零或另一策略覆盖时无害，仅作提示。
- `UNREACHABLE_TERM`：目标项非零但所有活跃策略概率均为零——估计量
  有偏。在本契约的合法输入下数学上不可达（`g_i>0` 蕴含 `L_i>0` 且
  `cos_i>0`），作为防御性检查保留并有直接测试。
- `VARIANCE_NOT_ESTIMABLE`：某活跃策略恰好 1 个样本，其方差贡献不可
  估计，此时 `variance`/`stddev` 为 `None`。
- `UNUSED_ZERO_MASS_STRATEGY`：未使用的策略总质量为零（信息性）。
- `WEIGHTS_UPDATED`：某批之后配比被更新（逐次记录下一批的最终份额）。
- `WEIGHTS_CLAMPED`：updater 请求的份额越界（形式合法但低于 `floor`
  或高于 `1-floor`），已被夹到下界并明示。

## 误差口径（浮点）

- `direct_sum` 与估计量累加均用 `math.fsum`（补偿求和），64 项规模下
  累加误差在 1 ulp 量级；除此外每样本贡献只有一次除法与若干次乘法。
- 多批聚合是 `fsum(批估计)/批数` 一次除法；每批内部与单轮口径一致。
- 期望恒等式验证：精确有理数（`Fraction`）断言严格相等；浮点逐批条件
  期望断言相对容差 `1e-12`；256 个独立种子的蒙特卡洛无偏性用
  `4·SE` 界（固定分布与逐批变化两种 regime）。
- 方向模长校验容差 `1e-9`（唯一的近似型输入校验，已在上文标明）。
- 随机估计的判定不使用宽容差掩盖错误：单亮片测试用估计标准差的
  3σ 界，单片场景断言严格相等且方差为 0。

## 设计取舍

- 提议分布按 `L·w` / `cos·w` 归一（等立体角下即 "∝L / ∝cos"）。
- `f` 只接受有限数组而非回调：契约可静态校验、可穷举、可复现。
- 抽样用 `random.Random(seed)` + 累积分布二分，整条自适应链路由单个
  `Random` 实例按批顺序喂数，同种子完整复现（包括更新决策的效果）。
- 方差按策略分层估计：`var = Σ_s n_s · s_s²`（`s_s²` 为该策略样本贡献
  的无偏样本方差）；多批结果的方差是独立批次均值的方差
  `(1/B²)Σ_b Var(E_b)`；任一批某策略层只有 1 个样本时整体报 `None`，
  不给出虚假精度。
- 失败即抛异常与诊断列表分离：致命契约违例（非法输入、退化提议、
  updater 非法返回）抛异常；形式合法但越界的权重请求走"夹紧+诊断"，
  因为正概率下界是可自动维持的保证而非契约违例。
- 权重按比值解释、内部归一化，因此接受 `(1,3)` 这类输入；但不接受
  零、负、非有限值。整数拆分保证每策略至少 `ceil(m·floor)` 个样本，
  当该保证在给定 `samples_per_batch` 下不可实现时提前拒绝。

## 已知限制

- 仅上半球（`z >= 0`），`cos` 取方向 z 分量，不支持自定义法线。
- `L`、`f` 均要求非负；负值被积函数不在本契约内。
- 小片数上限 64；批次数上限 1024；无分层/低差异序列等更高级采样；
  方差在单样本策略层下不可估计（报 `None`）。
- 多批模式下两策略每批必须都活跃（由 `floor>0` 强制），因此不支持
  完全关闭某个策略；需要单策略估计请继续使用旧 `estimate`。
- `variance_minimizing_updater` 是单批经验扩展的启发式：每样本价值
  本身依赖配比，故不保证全局最优；它只在两个已有策略间重新分配，
  不引入新策略。
- 只实现本题契约的数学子集，不宣称兼容任何渲染工业标准。
