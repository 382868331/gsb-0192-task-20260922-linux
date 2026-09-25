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
python -m unittest discover -s tests -v   # 必要测试
python demo.py                            # 固定输入演示（约 0.1 秒）
```

（仓库内所有命令等价地走 `./.venv/bin/python`，只需标准库。）

## 接口

```python
from mis_envlight import (
    Patch, Scene, estimate, estimate_with_control, direct_sum, proposal_pdfs,
    enumerate_contributions,
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
- `estimate_with_control(scene, f, control, control_sum,
  n_pilot_radiance, n_pilot_cosine, n_radiance, n_cosine, seed)`
  `-> ControlEstimateResult` — 两阶段控制变量估计，见下节。
- `proposal_pdfs(scene)` — 两提议分布的逐小片概率。
- `enumerate_contributions(scene, f, n_radiance, n_cosine)` —
  确定性的逐（策略， 小片）贡献表，用于穷举验证期望。

`EstimateResult` 字段：`estimate`、`variance`（估计量的估计方差，
见下文）、`stddev`、`exact`、`n_radiance`、`n_cosine`、
`records`（逐样本 `SampleRecord`：策略、小片索引、两策略 pdf、
balance-heuristic 权重 `mis_weight`、目标项、贡献）、
`diagnostics`；`result.explain(limit=None)` 给出逐样本权重解释文本。

## 控制变量估计（estimate_with_control）

调用方提供**控制数组** `control`（即逐小片控制项 `g_i`，与目标求和项
`t_i = L_i·f_i·cos_i·w_i` 同单位，**可正可负**，只要求有限）及其
**精确离散积分** `control_sum`（`G = sum_i g_i`，由调用方保证精确，
实现不回算、不校验其数值正确性）。估计量为

```
estimate = MIS( t_i - beta * g_i ) + beta * G
```

两阶段结构：

1. **pilot 阶段**：用 `n_pilot_radiance + n_pilot_cosine` 个样本（独立
   随机流，见下）估计一个**固定**系数
   `beta = (Σ_s n_s·Cov_s(x, y)) / (Σ_s n_s·Var_s(y))`，其中 `x`、`y`
   分别为逐样本目标贡献与控制贡献，`s` 遍历两策略（每策略内样本
   i.i.d.，该式使残差估计量的总方差最小）。pilot 样本只用于估计
   beta，**计入总预算**（`n_total` 包含 pilot）。
2. **主阶段**：beta 冻结，不再随主阶段样本变化；用
   `random.Random(seed)` 对残差项 `t_i - beta·g_i` 做普通 MIS
   （提议分布与 balance-heuristic 分母不变，仍按真实策略样本数
   计算；残差保持带符号，**不取绝对值**），最后加回精确的
   `beta * G`。

随机数流：pilot 用 `random.Random(_pilot_seed(seed))`（由调用种子经
固定 64 位整数混合派生，跨进程确定），主阶段用 `random.Random(seed)`
——两条流分开，且主阶段流与不启用控制变量的 `estimate()` 完全相同：
`control` 全零（此时 pilot 方差为零、beta 回退为 0）时主阶段逐样本
复现旧路径（有测试断言）。

`ControlEstimateResult` 字段：`estimate`、`variance`、`stddev`、
`exact`、`beta`、`residual_estimate`（加回 `beta*G` 之前的残差
MIS 估计）、`control_sum`、`n_pilot_radiance`、`n_pilot_cosine`、
`n_radiance`、`n_cosine`、`n_total`（含 pilot 的总预算）、
`records`（主阶段逐样本记录，`target_value` 为残差项，
`control_value` 为该小片控制项）、`pilot_records`（pilot 阶段记录，
`target_value` 为原始目标项）、`diagnostics`；`explain()` 输出两阶段
摘要与逐样本权重表。

**方差口径（控制变量路径）**：`variance = Σ_s n_s·s_s²`，取**主阶段**
残差贡献的逐策略无偏样本方差；以 pilot 估计的 beta 为固定常数
（条件方差），精确的 `beta*G` 加回不贡献方差，pilot 抽样的不确定性
不向内传播。任一活跃主策略恰好 1 个样本时报 `None` 并给出
`VARIANCE_NOT_ESTIMABLE` 诊断。

**pilot 方差为零的回退**：pilot 控制贡献方差为零（或没有任何策略有
≥2 个 pilot 样本，包括 pilot 数为 0）时无法估计 beta，此时
`beta = 0.0` 并给出 `PILOT_ZERO_VARIANCE` 诊断说明——估计量退化为
普通 MIS，pilot 预算被花费但不起作用。

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
  `cos=0` 的掠射片）。目标项为零或另一策略覆盖时无害，仅作提示；
  控制变量路径下消息同时列出该小片的控制项。
- `UNREACHABLE_TERM`：目标项非零**或控制项非零**但所有活跃策略概率
  均为零——估计量有偏。目标项情形在本契约的合法输入下数学上不可达
  （`t_i>0` 蕴含 `L_i>0` 且 `cos_i>0`），作为防御性检查保留并有直接
  测试；控制项情形真实可达（`g_i` 由调用方任意给定，可在 `L_i=0` 或
  `cos_i=0` 的小片上非零），有专门测试。
- `VARIANCE_NOT_ESTIMABLE`：某活跃策略恰好 1 个样本，其方差贡献不可
  估计，此时 `variance`/`stddev` 为 `None`。
- `UNUSED_ZERO_MASS_STRATEGY`：未使用的策略总质量为零（信息性）。
- `PILOT_ZERO_VARIANCE`：pilot 阶段无法估计 beta（控制贡献方差为零
  或 pilot 样本不足），beta 回退为 0.0，见上节。

## 误差口径（浮点）

- `direct_sum` 与估计量累加均用 `math.fsum`（补偿求和），64 项规模下
  累加误差在 1 ulp 量级；除此外每样本贡献只有一次除法与若干次乘法。
- 期望恒等式验证：精确有理数（`Fraction`）断言严格相等；浮点穷举
  断言相对容差 `1e-12`。控制变量路径另验证：固定 beta 下
  `E[残差 MIS] + beta*G == S` 的精确有理数恒等式，以及控制项等于
  目标项时估计严格等于 `G`（beta 恰为 1.0、残差恰为 0）。
- 方向模长校验容差 `1e-9`（唯一的近似型输入校验，已在上文标明）。
- 随机估计的判定不使用宽容差掩盖错误：单亮片测试用估计标准差的
  3σ 界，单片场景断言严格相等且方差为 0；"目标 = 控制 + 常量"场景
  用 3σ 界核对精确和；种子扫描只比较聚合 MSE。

## 设计取舍

- 提议分布按 `L·w` / `cos·w` 归一（等立体角下即 "∝L / ∝cos"）。
- `f` 只接受有限数组而非回调：契约可静态校验、可穷举、可复现。
- 抽样用 `random.Random(seed)` + 累积分布二分，跨运行确定性复现。
- 方差按策略分层估计：`var = Σ_s n_s · s_s²`（`s_s²` 为该策略样本贡献
  的无偏样本方差）；任一活跃策略只有 1 个样本时整体报告 `None`，
  不给出虚假精度。
- 控制变量：控制项按"逐小片求和项"建模（而非 f 同型数组），因此
  控制可以在 `L=0` / `cos=0` 的小片上非零，零支持诊断必须覆盖它；
  beta 用独立 pilot 流一次性估计后冻结，避免 beta 追着主阶段样本
  变化引入偏差；主阶段随机流与旧路径一致，保证零控制时逐样本兼容。
- 失败即抛异常与诊断列表分离：致命契约违例（非法输入、退化提议）
  抛异常；采样可继续的异常结构（零概率项等）进入 `diagnostics`。

## 已知限制

- 仅上半球（`z >= 0`），`cos` 取方向 z 分量，不支持自定义法线。
- `L`、`f` 均要求非负；负值被积函数不在本契约内（控制数组 `g` 与
  残差可正可负，这是控制变量路径的正常状态，不是输入放松）。
- 小片数上限 64；无分层/低差异序列等更高级采样；方差在单样本
  策略下不可估计（报 `None`）。
- 控制变量路径的方差不传播 pilot 阶段估计 beta 的不确定性（beta
  视为固定常数的条件方差）；pilot 样本只用于 beta，不并入主估计。
- 方差下降**不是普遍保证**：只有当控制与目标足够相关且 pilot 成本
  占比合理时才有收益；测试与 demo 只断言有限种子集合上的聚合 MSE
  下降，不断言单次运行或任意场景。
- `control_sum` 由调用方保证精确，实现不回算；给错 G 会按
  `beta * (G_错 - G_真)` 引入偏差。
- 只实现本题契约的数学子集，不宣称兼容任何渲染工业标准。
