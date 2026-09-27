# mis_envlight — 离散环境光的多策略重要性估计

离线半球积分器：对至多 64 个离散小片的环境光场景，用多重重要性采样
（MIS, balance heuristic）估计

```
S = sum_i  L_i * f_i * cos_i * w_i,   cos_i = max(dot(d_i, n), 0)
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
    DEFAULT_NORMAL,
    ValidationError, DegenerateProposalError, MisEnvLightError,
)
```

- `Patch(direction, radiance, weight)` — 单个小片（frozen dataclass）。
  - `direction`：单位 3 向量 `(x, y, z)`，可取球面上任意方向（该方向
    相对表面是正面还是背面取决于查询时给定的法线，见下）；模长容差
    `NORM_TOLERANCE = 1e-9`，不做静默归一化。
  - `radiance`：有限实数，`>= 0`。
  - `weight`：有限实数，`> 0`（立体角权重）。
- `Scene(patches)` — 1..64 个 `Patch`；构造时全部校验，失败不留下
  部分构造的对象。表面方向不由场景固定：查询函数接受 `normal` 参数
  （单位 3 向量，默认 `DEFAULT_NORMAL = (0, 0, 1)`），
  `cos_i = max(dot(direction_i, normal), 0)`；背面方向（点积 ≤ 0）
  对正面贡献为 0，也不获得 cosine 策略的采样概率。
- `direct_sum(scene, f, normal=DEFAULT_NORMAL) -> float` — 精确目标值
  （`math.fsum` 累加）。
- `estimate(scene, f, n_radiance, n_cosine, seed, normal=DEFAULT_NORMAL)
  -> EstimateResult` — MIS 估计。`f` 为长度等于小片数的有限非负数组
  （不是回调）；`n_radiance` / `n_cosine` 为两策略各自样本数
  （`int >= 0`，合计至少 1）；`seed` 为 `int`，同一种子结果完全可
  复现；`normal` 为单位表面法线。
- `proposal_pdfs(scene, normal=DEFAULT_NORMAL)` — 两提议分布的逐小片
  概率；cosine 分布随 `normal` 一起改变（采样方向与采样概率同源）。
- `enumerate_contributions(scene, f, n_radiance, n_cosine, normal=...)`
  — 确定性的逐（策略， 小片）贡献表，用于穷举验证期望。

`EstimateResult` 字段：`estimate`、`variance`（估计量的估计方差，
见下文）、`stddev`、`exact`、`n_radiance`、`n_cosine`、
`normal`（本次查询使用的单位法线）、
`records`（逐样本 `SampleRecord`：策略、小片索引、两策略 pdf、
balance-heuristic 权重 `mis_weight`、目标项、贡献）、
`diagnostics`；`result.explain(limit=None)` 给出逐样本权重解释文本。

## 估计量定义

两提议分布分别为 `p_i ∝ L_i·w_i`（radiance 策略）与
`q_i ∝ cos_i·w_i`（cosine 策略）。场景契约是等立体角小片，此时与
"正比于 L / 正比于 cos" 完全等价；保留 `w_i` 使非均匀权重也正确。
`cos_i = max(dot(d_i, n), 0)` 同时决定目标项与 cosine 提议概率：
改变法线时，采样方向与采样概率一起调整，绝不会只旋转结果而沿用旧
权重。落在表面背面的方向 `cos_i = 0`，既不产生正面贡献，也不被
cosine 策略采样；多策略混用时每样本仍按其实际采用策略的概率
（`p_i` / `q_i`）记入 MIS 分母。
balance heuristic 计入每策略样本数：小片 `i` 处单次抽样的贡献为

```
contrib_i = g_i / (n_radiance * p_i + n_cosine * q_i),   g_i = L_i f_i cos_i w_i
```

估计量为全部样本贡献之和，其期望恒等于 `S`（测试中用 `fractions.Fraction`
精确验证该恒等式，并穷举一次抽样的全部组合核对）。

## 错误语义

- 数值输入拒绝 `NaN` / `±Infinity`；整数参数（样本数、种子）拒绝
  `bool`；`f` 拒绝负数、错误长度、非序列；`normal` 必须是 3 分量
  单位向量（容差 `NORM_TOLERANCE`），任何朝向均可。
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
- 法线是查询参数而非场景参数：同一场景可对不同朝向复用；radiance
  提议与法线无关，cosine 提议、目标项、退化检查、诊断全部由同一组
  `cos_i` 派生，保证概率与方向一致。整体旋转场景并同步旋转法线后，
  所有逐小片概率不变，同种子估计结果逐样本一致（有直接测试）。
- `f` 只接受有限数组而非回调：契约可静态校验、可穷举、可复现。
- 抽样用 `random.Random(seed)` + 累积分布二分，跨运行确定性复现。
- 方差按策略分层估计：`var = Σ_s n_s · s_s²`（`s_s²` 为该策略样本贡献
  的无偏样本方差）；任一活跃策略只有 1 个样本时整体报告 `None`，
  不给出虚假精度。
- 失败即抛异常与诊断列表分离：致命契约违例（非法输入、退化提议）
  抛异常；采样可继续的异常结构（零概率项等）进入 `diagnostics`。

## 已知限制

- 法线默认为 `(0, 0, 1)`；方向可取整个球面，但仅 `dot(d, n) > 0`
  的小片对正面积分有贡献（背面片在 radiance 策略下仍可被采到，其
  贡献按实际概率正确记账为 0）。
- `L`、`f` 均要求非负；负值被积函数不在本契约内。
- 小片数上限 64；无分层/低差异序列等更高级采样；方差在单样本
  策略下不可估计（报 `None`）。
- 只实现本题契约的数学子集，不宣称兼容任何渲染工业标准。
