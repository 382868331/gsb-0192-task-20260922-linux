# mis_envlight — 离散环境光的多策略重要性估计

离线半球积分器：对至多 64 个离散环境小片（方向可在整个单位球上），
用多重重要性采样
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

## 接口

```python
from mis_envlight import (
    Patch, Scene, estimate, direct_sum, proposal_pdfs,
    enumerate_contributions,
    ValidationError, DegenerateProposalError, MisEnvLightError,
)
```

- `Patch(direction, radiance, weight)` — 单个小片（frozen dataclass）。
  - `direction`：单位 3 向量 `(x, y, z)`，可位于单位球任意位置；
    模长容差 `NORM_TOLERANCE = 1e-9`，不做静默归一化。
  - `radiance`：有限实数，`>= 0`。
  - `weight`：有限实数，`> 0`（立体角权重）。
- `Scene(patches, normal=(0, 0, 1))` — 1..64 个 `Patch` 与接收表面的
  单位法线 `normal`（默认 +z）；构造时全部校验，失败不留下部分构造
  的对象。`cos_i = max(direction_i · normal, 0)`：法线背面的小片
  余弦被夹为 0，既不产生正面贡献，也不进入 cosine 提议分布。
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

两提议分布分别为 `p_i ∝ L_i·w_i`（radiance 策略，覆盖全球的亮片，
含背面片，但其目标项为 0，被抽中时贡献恰好为 0）与
`q_i ∝ max(d_i·n, 0)·w_i`（cosine 策略，只覆盖正面半球）。场景契约
是等立体角小片，此时与 "正比于 L / 正比于 cos" 完全等价；保留
`w_i` 使非均匀权重也正确。法线改变时，cosine、`q_i` 的归一化常数与
MIS 分母一起重新计算，不能只旋转旧结果而沿用旧权重。
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
  相对当前法线 `cos=0` 的掠射片）。目标项为零或另一策略覆盖时无害，
  仅作提示。
- `BACK_FACING_PATCH`：某小片方向落在法线背面（`d·n < 0`）。其 cosine
  被夹为 0，正面目标项为 0，cosine 策略概率为 0；即使 radiance 策略
  抽中它，贡献也恰好为 0，不会泄漏为正面贡献。
- `UNREACHABLE_TERM`：目标项非零但所有活跃策略概率均为零——估计量
  有偏。在本契约的合法输入下数学上不可达（`g_i>0` 蕴含 `L_i>0` 且
  `cos_i>0`），作为防御性检查保留并有直接测试。
- `VARIANCE_NOT_ESTIMABLE`：某活跃策略恰好 1 个样本，其方差贡献不可
  估计，此时 `variance`/`stddev` 为 `None`。
- `UNUSED_ZERO_MASS_STRATEGY`：未使用的策略总质量为零（信息性）。

## 误差口径（浮点）

- `direct_sum` 与估计量累加均用 `math.fsum`（补偿求和），64 项规模下
  累加误差在 1 ulp 量级；除此外每样本贡献只有一次除法与若干次乘法。
- 期望恒等式验证：精确有理数（`Fraction`）断言严格相等；浮点穷举
  断言相对容差 `1e-12`。
- 方向模长校验容差 `1e-9`（唯一的近似型输入校验，已在上文标明）。
- 随机估计的判定不使用宽容差掩盖错误：单亮片测试用估计标准差的
  3σ 界，单片场景断言严格相等且方差为 0。

## 设计取舍

- 提议分布按 `L·w` / `max(d·n,0)·w` 归一（等立体角下即
  "∝L / ∝cos"）；法线改变时 cosine、pdf 与 MIS 分母一并重算。
- 旋转一致性：对所有小片方向与法线施加同一旋转 R，`cos_i`、两组
  pdf、抽样序列（同种子）与估计量逐位不变；测试用绕 y 轴 90°
  旋转验证。
- `f` 只接受有限数组而非回调：契约可静态校验、可穷举、可复现。
- 抽样用 `random.Random(seed)` + 累积分布二分，跨运行确定性复现。
- 方差按策略分层估计：`var = Σ_s n_s · s_s²`（`s_s²` 为该策略样本贡献
  的无偏样本方差）；任一活跃策略只有 1 个样本时整体报告 `None`，
  不给出虚假精度。
- 失败即抛异常与诊断列表分离：致命契约违例（非法输入、退化提议）
  抛异常；采样可继续的异常结构（零概率项等）进入 `diagnostics`。

## 已知限制

- 方向允许位于整个单位球；是否接收光由 `normal` 决定，背面方向恒为
  零贡献（不区分双面材质）。
- `L`、`f` 均要求非负；负值被积函数不在本契约内。
- 小片数上限 64；无分层/低差异序列等更高级采样；方差在单样本
  策略下不可估计（报 `None`）。
- 只实现本题契约的数学子集，不宣称兼容任何渲染工业标准。
