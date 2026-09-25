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
./.venv/bin/python -m unittest discover -s tests -v   # 必要测试
./.venv/bin/python demo.py                            # 固定输入演示（约 0.1 秒）
```

## 接口

```python
from mis_envlight import (
    Patch, Scene, estimate, estimate_with_control, direct_sum,
    proposal_pdfs, enumerate_contributions,
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
- `estimate_with_control(scene, f, g, G, n_radiance, n_cosine,
  n_pilot_radiance=0, n_pilot_cosine=0, seed=0) -> ControlEstimateResult` —
  带控制变量的两阶段 MIS 估计（见下节）。`g` 为长度等于小片数的
  有限控制数组，**可正可负**；`G` 为调用方给出的控制精确离散积分
  `sum_i L_i*g_i*cos_i*w_i`（有限实数，库信任该值，不重算；
  `result.control_sum` 是库自算的同一和，可用于交叉核对）。
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

## 控制变量两阶段估计（estimate_with_control）

调用方提供控制数组 `g`（逐小片，**可正可负**）及其精确离散积分
`G = sum_i L_i·g_i·cos_i·w_i`。记目标项 `T_i = L_i f_i cos_i w_i`、
控制项 `C_i = L_i g_i cos_i w_i`、正式阶段 MIS 分母
`D_i = n_radiance·p_i + n_cosine·q_i`（按真实策略样本数计算），
单样本商 `X = T_i/D_i`、`Y = C_i/D_i`。估计量为

```
estimate = [正式样本的 Σ (X - beta·Y)] + beta·G
```

对任意固定 `beta` 都是 `S` 的无偏估计（残差估计 `S - beta·ΣC`，而
`ΣC = G`）。残差保持符号，不取绝对值；提议分布与 MIS 分母不变。

- **pilot 阶段**：`n_pilot_radiance + n_pilot_cosine` 个样本，来自
  **独立随机流**（`seed` 的固定整数混合，见 `_pilot_seed`），估计
  固定系数 `beta = Σ_s n_s·cov_s(X,Y) / Σ_s n_s·var_s(Y)`（`n_s` 为
  **正式**样本数，`cov_s/var_s` 为该策略 pilot 样本协方差/方差；
  pilot 样本少于 2 个的策略不参与）。
- **正式阶段**：`beta` 固定不变，正式样本不再影响它；主随机流为
  `random.Random(seed)`，与 `estimate()` 完全相同。
- **pilot 方差为零**（或没有任何策略有足够的 pilot 样本）时
  `beta = 0.0`，并给出 `CONTROL_ZERO_PILOT_VARIANCE` 诊断解释原因；
  此时结果退化为普通 MIS。
- **预算包含 pilot**：`result.total_samples` 为两阶段样本数之和。
- **兼容性**：`beta = 0` 时正式阶段逐样本与
  `estimate(..., 同 seed)` 完全一致（有测试逐条比对）；`estimate()`
  本身未改动，旧种子结果不变。
- `ControlEstimateResult` 字段：`estimate`、`variance`/`stddev`
  （口径同 `estimate()`，对残差贡献按策略分层）、`exact`、`beta`、
  `control_exact`（调用方的 `G`）、`control_sum`（库自算的控制项和）、
  `residual_estimate`、两阶段各自样本数、`total_samples`、
  `records`（正式）、`pilot_records`（pilot）、`diagnostics`；
  `explain()` 给出逐样本解释。记录中的 `mis_weight` 两阶段都按
  **正式**样本数计算（pilot 的作用只是定 `beta`）。
- 控制变量恒等式只保证无偏，**不保证方差必然下降**；demo 与测试中的
  方差对比只是特定场景/控制/种子集上的观测事实。

## 错误语义

- 数值输入拒绝 `NaN` / `±Infinity`；整数参数（样本数、种子）拒绝
  `bool`；`f` 拒绝负数、错误长度、非序列；`g` 允许负数但拒绝
  非有限值与错误长度；`G` 拒绝非有限值与非数值。
- 非法输入抛 `ValidationError`（同时是 `ValueError`），消息定位到字段，
  如 `patches[1].radiance: must be >= 0, got -2.0`；不做任何静默纠正。
- 某策略被请求样本但总提议质量为 0（如零能量场景请求 radiance 策略）
  抛 `DegenerateProposalError`。
- 所有校验在抽样之前完成：调用失败不会留下未说明的部分状态
  （`estimate` 本身无外部副作用）。

## 诊断（diagnostics，不中断计算）

- `ZERO_PDF_TERM`：某小片在一个活跃策略下概率为零（如 `L=0` 的暗片、
  `cos=0` 的掠射片）。目标项为零或另一策略覆盖时无害，仅作提示；
  启用控制变量时消息同时给出该小片的控制项。
- `UNREACHABLE_TERM`：目标项非零**或控制项非零**但所有活跃策略概率
  均为零——估计量有偏。在本契约的合法输入下数学上不可达（非零项蕴含
  `L_i>0` 且 `cos_i>0`），作为防御性检查保留并有直接测试。
- `VARIANCE_NOT_ESTIMABLE`：某活跃策略恰好 1 个样本，其方差贡献不可
  估计，此时 `variance`/`stddev` 为 `None`。
- `UNUSED_ZERO_MASS_STRATEGY`：未使用的策略总质量为零（信息性）。
- `CONTROL_ZERO_PILOT_VARIANCE`：pilot 无法估计 `beta`（无任何策略有
  ≥2 个 pilot 样本，或 pilot 控制方差为零），`beta` 取 0.0，控制变量
  不起作用；诊断消息说明原因。

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
  不给出虚假精度。控制变量版本对**残差贡献** `X - beta·Y` 用同一
  口径，`beta` 来自 pilot 且正式阶段固定。
- 失败即抛异常与诊断列表分离：致命契约违例（非法输入、退化提议）
  抛异常；采样可继续的异常结构（零概率项等）进入 `diagnostics`。

## 已知限制

- 仅上半球（`z >= 0`），`cos` 取方向 z 分量，不支持自定义法线。
- `L`、`f` 均要求非负；负值被积函数不在本契约内（控制数组 `g`
  可正可负，但它只是控制变量，不改变目标）。
- 控制变量不保证方差下降：恒等式只保证对固定 `beta` 无偏；`G` 由
  调用方负责其精确性，库信任该值（`control_sum` 可交叉核对）。
- 小片数上限 64；无分层/低差异序列等更高级采样；方差在单样本
  策略下不可估计（报 `None`）。
- 只实现本题契约的数学子集，不宣称兼容任何渲染工业标准。
