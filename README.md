# mis_envlight — 离散环境光的多策略重要性估计

离线半球积分器：场景为至多 64 个等立体角小片（方向、非负辐亮度、立体角权重），
目标积分为离散和

```
I = sum_i  L_i * f_i * cos_i * w_i ,  cos_i = max(0, dot(normal, dir_i))
```

两个提议分布分别正比于 `L` 与 `cos`，用 balance heuristic 做多重重要性采样
（MIS），权重中计入每策略样本数；提供确定种子抽样、估计方差与逐样本权重解释。
`f` 以有限数组逐片给出，不接受任意回调。

## 环境

- Python 3.14.7（仅标准库，无第三方依赖，无需网络/账号/硬件/Docker）。
- Linux 原生运行；安装 Python 后无需安装任何其他依赖。

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 演示：估计 vs 精确值 + 两个实际触发的拒绝边界
```

## 接口（`mis_envlight` 包）

| 名称 | 说明 |
|---|---|
| `Patch(direction, radiance, weight)` | 小片数据。`direction` 为单位 3 向量；`radiance`、`weight` 为非负有限实数。 |
| `build_scene(patches, normal) -> Scene` | 校验并构建场景。1..64 片；法线须为单位向量。 |
| `exact_integral(scene, f_values) -> float` | 直接求和基准（`math.fsum`）。 |
| `strategy_probabilities(scene) -> (p_L, p_cos)` | 两个提议分布；退化的策略为 `None`。 |
| `mis_contributions(scene, f_values, n_radiance, n_cosine) -> ContributionTable` | 逐片单样本贡献表（穷举验证与抽样共用）。 |
| `estimate(scene, f_values, n_radiance, n_cosine, seed) -> MisEstimate` | 运行 MIS 估计。 |
| `diagnose(scene, f_values) -> list[Diagnostic]` | 退化策略 / 零概率项 / 不可达项诊断。 |
| `SampleRecord.explain()` | 逐样本权重的人类可读解释。 |

`MisEstimate` 字段：`estimate`、`variance`、`std_error`、`n_radiance`、`n_cosine`、
`samples`（逐样本 `SampleRecord`）、`diagnostics`。

### 估计量定义

策略 `s` 一次抽样命中小片 `i` 的贡献为

```
contrib_s(i) = t_i / (n_radiance*p_L[i] + n_cosine*p_C[i])
             = w_s(i) * t_i / (n_s * p_s(i)) ,  w_s(i) = n_s*p_s(i) / den_i   (balance heuristic)
```

估计量为全部 `n_radiance + n_cosine` 个样本贡献之和，对任意每策略样本数均无偏
（测试中以穷举所有一次抽样组合验证期望等于直接求和，相对容差 1e-12）。

## 错误语义

所有错误为 `ValueError` 子类，消息含参数名与小片下标，可定位；不静默纠正：

- `SceneValidationError`：NaN/Infinity 数值、负数辐亮度/权重、非单位向量
  （模长偏差 > 1e-6，不静默归一化）、`f_values` 长度不符、小片数越界、
  **整数参数（`n_radiance`/`n_cosine`/`seed`）拒绝 bool 与非 int**、
  两策略样本数全为 0。
- `DegenerateStrategyError`：请求抽样的策略概率质量全零（如总辐亮度为 0
  却请求 `n_radiance > 0`）。该策略样本数取 0 时合法。
- `UnestimableTargetError`：某片目标非零但两策略概率均为零（在本契约的
  目标形式 `L*f*cos*w` 下不可达，作为防御性检查保留）。

`estimate` 先完成全部校验再抽样：失败时抛异常，不返回部分结果，不留残留状态。

## 输入上限

- 小片数：1..64（`MAX_PATCHES`）。
- `n_radiance`、`n_cosine`：非负 int，合计 ≥ 1。
- `seed`：任意 int（拒绝 bool/非 int），同一种子结果完全确定。
- 方向与法线：单位 3 向量（容差 1e-6）。

## 误差口径（浮点）

- `exact_integral` 用 `math.fsum` 求和，是模块内的高精度基准。
- 估计量为 float64 累加（`math.fsum`）；MIS 估计在精确算术下无偏，
  穷举测试验证期望与直接求和一致（相对/绝对容差 1e-12，仅覆盖浮点舍入）。
- `variance` 为估计量方差：各策略组内无偏样本方差乘该组样本数后求和。
  **任一被使用的策略样本数 < 2 时方差无法无偏估计，`variance`/`std_error`
  为 `None`**（不是 0）。

## 设计取舍

- 提议分布严格按题面正比于 `L` 与 `cos`（不含立体角权重 `w`）；由于目标支撑
  要求 `L>0` 且 `cos>0`，两策略合起来对目标支撑覆盖完备，MIS 仍无偏。
- 先校验后抽样的两阶段设计，保证失败无部分变更。
- 抽样顺序固定（先 radiance 全部、再 cosine 全部），配合确定种子保证可复现。
- 方差按策略分组估计而非全体混合，避免不同策略均值不同带来的偏差。

## 已知限制

- 仅实现本题契约的离散子集：单法线、clamped cosine、逐片常量 `f`；
  不宣称完整渲染行业标准兼容（无 BSDF 回调、无连续半球采样、无光谱）。
- `f_values` 允许任意有限实数（含负数），非负性由调用方负责。
- 半球外（cos ≤ 0）的小片目标恒为 0，不拒绝，仅在诊断中可能体现为退化策略。
