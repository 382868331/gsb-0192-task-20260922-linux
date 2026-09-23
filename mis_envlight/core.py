"""离散环境光的多策略重要性估计（balance-heuristic MIS）。

场景：至多 ``MAX_PATCHES`` 个等立体角小片，每片给定单位方向、非负辐亮度
与立体角权重。目标积分为离散和 ``sum_i L_i * f_i * cos_i * w_i``，其中
``cos_i = max(0, dot(normal, dir_i))``，``f`` 以有限数组逐片给出。

两个提议分布分别正比于 ``L`` 与 ``cos``；按 balance heuristic 做多重重要
性采样，权重中计入每策略样本数。抽样由 ``random.Random(seed)`` 驱动，完全
确定。

公开契约与错误语义见 README.md；所有校验失败抛出 ``SceneValidationError``
（``ValueError`` 子类），消息中带参数名与小片下标，可定位。
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass

__all__ = [
    "MAX_PATCHES",
    "UNIT_VECTOR_TOLERANCE",
    "RADIANCE",
    "COSINE",
    "STRATEGIES",
    "MisEnvLightError",
    "SceneValidationError",
    "DegenerateStrategyError",
    "UnestimableTargetError",
    "Patch",
    "Scene",
    "Diagnostic",
    "ContributionTable",
    "SampleRecord",
    "MisEstimate",
    "build_scene",
    "exact_integral",
    "strategy_probabilities",
    "mis_contributions",
    "diagnose",
    "estimate",
]

MAX_PATCHES = 64
"""接口边界：小片数上限（题面给定）。"""

UNIT_VECTOR_TOLERANCE = 1e-6
"""单位向量模长允许偏差；超出即拒绝，不做静默归一化。"""

RADIANCE = "radiance"
COSINE = "cosine"
STRATEGIES = (RADIANCE, COSINE)


class MisEnvLightError(ValueError):
    """本模块所有输入/状态错误的基类（ValueError 子类）。"""


class SceneValidationError(MisEnvLightError):
    """输入违反契约（NaN/Inf、bool 充当 int、非单位向量、越界数量等）。"""


class DegenerateStrategyError(MisEnvLightError):
    """请求抽样的策略概率质量全零（如总辐亮度为 0 却请求 radiance 抽样）。"""


class UnestimableTargetError(MisEnvLightError):
    """存在目标非零的小片，但两个策略对它的概率都为零，MIS 无法覆盖。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Patch:
    """单个等立体角小片。字段在 build_scene 中统一校验。"""

    direction: tuple  # 单位向量 (x, y, z)，半球外方向
    radiance: float  # 非负有限辐亮度 L
    weight: float  # 非负有限立体角权重 w


@dataclass(frozen=True)
class Scene:
    """校验后的场景：小片、法线、逐片 clamped cos 及策略归一化常数。"""

    patches: tuple
    normal: tuple
    cosines: tuple  # cos_i = max(0, dot(normal, dir_i))
    sum_radiance: float
    sum_cosine: float

    @property
    def size(self) -> int:
        return len(self.patches)


@dataclass(frozen=True)
class Diagnostic:
    """诊断条目。code 取值：

    - ``strategy-degenerate-radiance`` / ``strategy-degenerate-cosine``：
      对应策略总质量为零，该策略不可用；
    - ``zero-probability-term``：某小片在某一策略下概率为零（另一策略仍
      覆盖，MIS 合法，但值得报告）；
    - ``term-unreachable``：目标非零但两策略概率均为零（estimate 会拒绝）。
    """

    code: str
    patch_index: object  # int 或 None（场景级诊断）
    message: str


@dataclass(frozen=True)
class ContributionTable:
    """逐片、逐策略的单样本贡献表（MIS 核心量，供估计与穷举验证共用）。

    ``radiance_contribution[i]`` 表示 radiance 策略一次抽样命中小片 i 时对
    估计量的贡献 ``t_i / (n_radiance*p_L[i] + n_cosine*p_C[i])``，即
    balance-heuristic 权重 ``n_s*p_s/den`` 乘以 ``t_i/(n_s*p_s[i])``；
    该策略概率为零的位置为 None（永不被抽到）。
    """

    n_radiance: int
    n_cosine: int
    targets: tuple  # t_i = L_i * f_i * cos_i * w_i
    p_radiance: object  # tuple 或 None（策略退化）
    p_cosine: object  # tuple 或 None（策略退化）
    radiance_contribution: tuple
    cosine_contribution: tuple


@dataclass(frozen=True)
class SampleRecord:
    """一次抽样的完整权重解释。"""

    draw: int  # 第几次抽样（0 起）
    strategy: str  # RADIANCE 或 COSINE
    patch_index: int
    target: float  # 该片的 t_i
    pdf_radiance: object  # float 或 None（策略退化）
    pdf_cosine: object  # float 或 None（策略退化）
    mis_weight: float  # balance heuristic 权重 n_s*p_s / sum_t n_t*p_t
    contribution: float  # 本次抽样对估计量的贡献

    def explain(self) -> str:
        def fmt(v):
            return "n/a(degenerate)" if v is None else f"{v:.6g}"

        return (
            f"draw {self.draw}: strategy={self.strategy}, patch={self.patch_index}, "
            f"target=L*f*cos*w={self.target:.6g}, "
            f"p_radiance={fmt(self.pdf_radiance)}, p_cosine={fmt(self.pdf_cosine)}, "
            f"mis_weight=n_strategy*p_strategy/(n_rad*p_rad+n_cos*p_cos)="
            f"{self.mis_weight:.6g}, contribution=mis_weight*target/"
            f"(n_strategy*p_strategy)={self.contribution:.6g}"
        )


@dataclass(frozen=True)
class MisEstimate:
    """估计结果。

    ``variance``/``std_error`` 为估计量方差/标准误；当某个被使用的策略抽样
    数 < 2 时方差无法无偏估计，二者为 None（契约见 README）。
    """

    estimate: float
    variance: object
    std_error: object
    n_radiance: int
    n_cosine: int
    samples: tuple  # tuple[SampleRecord, ...]
    diagnostics: tuple  # tuple[Diagnostic, ...]

    @property
    def n_samples(self) -> int:
        return self.n_radiance + self.n_cosine


# ---------------------------------------------------------------------------
# 输入校验（拒绝 NaN/Inf；整数参数拒绝 bool；错误消息可定位）
# ---------------------------------------------------------------------------


def _check_real(value, name, *, allow_negative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneValidationError(
            f"{name}: expected a real number (int/float, not bool), "
            f"got {type(value).__name__}"
        )
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise SceneValidationError(f"{name}: must be finite (NaN/Infinity rejected), got {v!r}")
    if not allow_negative and v < 0.0:
        raise SceneValidationError(f"{name}: must be non-negative, got {v!r}")
    return v


def _check_int(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise SceneValidationError(
            f"{name}: expected int (bool rejected), got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise SceneValidationError(f"{name}: must be >= {minimum}, got {value}")
    return value


def _check_unit_vector(vec, name):
    if isinstance(vec, str) or not isinstance(vec, (tuple, list)) or len(vec) != 3:
        raise SceneValidationError(
            f"{name}: expected a 3-component sequence, got {vec!r}"
        )
    comps = tuple(
        _check_real(c, f"{name}[{k}]", allow_negative=True)
        for k, c in enumerate(vec)
    )
    norm = math.sqrt(math.fsum(c * c for c in comps))
    if abs(norm - 1.0) > UNIT_VECTOR_TOLERANCE:
        raise SceneValidationError(
            f"{name}: expected a unit vector (|v|-1 <= {UNIT_VECTOR_TOLERANCE}), "
            f"got |v|={norm!r}; normalize it yourself, no silent correction"
        )
    return comps


def _check_f_values(f_values, size):
    if isinstance(f_values, str) or not isinstance(f_values, (tuple, list)):
        raise SceneValidationError(
            f"f_values: expected a sequence of {size} finite reals, "
            f"got {type(f_values).__name__}"
        )
    if len(f_values) != size:
        raise SceneValidationError(
            f"f_values: expected {size} entries (one per patch), got {len(f_values)}"
        )
    return tuple(
        _check_real(v, f"f_values[{i}]", allow_negative=True)
        for i, v in enumerate(f_values)
    )


# ---------------------------------------------------------------------------
# 场景构建与精确积分
# ---------------------------------------------------------------------------


def build_scene(patches, normal) -> Scene:
    """校验并构建场景。任何非法输入在此抛出可定位的 SceneValidationError。"""
    if isinstance(patches, str) or not isinstance(patches, (tuple, list)):
        raise SceneValidationError(
            f"patches: expected a sequence of Patch, got {type(patches).__name__}"
        )
    if not 1 <= len(patches) <= MAX_PATCHES:
        raise SceneValidationError(
            f"patches: expected 1..{MAX_PATCHES} patches, got {len(patches)}"
        )
    checked = []
    for i, p in enumerate(patches):
        if not isinstance(p, Patch):
            raise SceneValidationError(
                f"patches[{i}]: expected a Patch instance, got {type(p).__name__}"
            )
        direction = _check_unit_vector(p.direction, f"patches[{i}].direction")
        radiance = _check_real(p.radiance, f"patches[{i}].radiance")
        weight = _check_real(p.weight, f"patches[{i}].weight")
        checked.append(Patch(direction, radiance, weight))
    nrm = _check_unit_vector(normal, "normal")
    cosines = tuple(
        max(0.0, math.fsum(a * b for a, b in zip(nrm, p.direction)))
        for p in checked
    )
    return Scene(
        patches=tuple(checked),
        normal=nrm,
        cosines=cosines,
        sum_radiance=math.fsum(p.radiance for p in checked),
        sum_cosine=math.fsum(cosines),
    )


def _targets(scene: Scene, f):
    return tuple(
        p.radiance * fi * c * p.weight
        for p, fi, c in zip(scene.patches, f, scene.cosines)
    )


def exact_integral(scene: Scene, f_values) -> float:
    """直接求和 sum L*f*cos*weight（math.fsum，高精度基准）。"""
    if not isinstance(scene, Scene):
        raise SceneValidationError(
            f"scene: expected a Scene from build_scene, got {type(scene).__name__}"
        )
    f = _check_f_values(f_values, scene.size)
    return math.fsum(_targets(scene, f))


def strategy_probabilities(scene: Scene):
    """返回 (p_radiance, p_cosine)；某策略总质量为零时对应项为 None。"""
    if not isinstance(scene, Scene):
        raise SceneValidationError(
            f"scene: expected a Scene from build_scene, got {type(scene).__name__}"
        )
    if scene.sum_radiance > 0.0:
        p_rad = tuple(p.radiance / scene.sum_radiance for p in scene.patches)
    else:
        p_rad = None
    if scene.sum_cosine > 0.0:
        p_cos = tuple(c / scene.sum_cosine for c in scene.cosines)
    else:
        p_cos = None
    return p_rad, p_cos


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------


def diagnose(scene: Scene, f_values) -> list:
    """返回诊断列表：退化策略、零概率项、以及两策略均零但目标非零的项。"""
    if not isinstance(scene, Scene):
        raise SceneValidationError(
            f"scene: expected a Scene from build_scene, got {type(scene).__name__}"
        )
    f = _check_f_values(f_values, scene.size)
    targets = _targets(scene, f)
    p_rad, p_cos = strategy_probabilities(scene)
    out = []
    if p_rad is None:
        out.append(
            Diagnostic(
                "strategy-degenerate-radiance",
                None,
                "total radiance is zero; the radiance-proportional strategy "
                "cannot be sampled",
            )
        )
    if p_cos is None:
        out.append(
            Diagnostic(
                "strategy-degenerate-cosine",
                None,
                "total clamped cosine is zero; the cosine-proportional strategy "
                "cannot be sampled",
            )
        )
    for i, t in enumerate(targets):
        pr = 0.0 if p_rad is None else p_rad[i]
        pc = 0.0 if p_cos is None else p_cos[i]
        if pr == 0.0 and pc == 0.0:
            if t != 0.0:
                out.append(
                    Diagnostic(
                        "term-unreachable",
                        i,
                        f"patch {i}: target t={t!r} is non-zero but both "
                        f"strategies assign zero probability; MIS cannot "
                        f"estimate this term",
                    )
                )
        elif pr == 0.0 or pc == 0.0:
            which = "radiance" if pr == 0.0 else "cosine"
            out.append(
                Diagnostic(
                    "zero-probability-term",
                    i,
                    f"patch {i}: {which} strategy assigns zero probability "
                    f"(target t={t!r}); the other strategy still covers it",
                )
            )
    return out


# ---------------------------------------------------------------------------
# MIS 贡献表（balance heuristic，计入每策略样本数）
# ---------------------------------------------------------------------------


def mis_contributions(scene: Scene, f_values, n_radiance: int, n_cosine: int) -> ContributionTable:
    """计算逐片单样本贡献表。

    对策略 s 命中小片 i 的一次抽样，贡献为
    ``t_i / (n_radiance*p_L[i] + n_cosine*p_C[i])``，
    等价于 balance-heuristic 权重 ``n_s*p_s[i]/den_i`` 乘 ``t_i/(n_s*p_s[i])``
    （即 Veach/PBRT 形式 ``(1/n_s) * w_s * t_i/p_s[i]``），
    估计量为所有样本贡献之和，对任意每策略样本数均无偏。
    """
    if not isinstance(scene, Scene):
        raise SceneValidationError(
            f"scene: expected a Scene from build_scene, got {type(scene).__name__}"
        )
    n_rad = _check_int(n_radiance, "n_radiance", minimum=0)
    n_cos = _check_int(n_cosine, "n_cosine", minimum=0)
    if n_rad + n_cos < 1:
        raise SceneValidationError(
            "n_radiance + n_cosine: must be >= 1 (no samples requested)"
        )
    f = _check_f_values(f_values, scene.size)
    targets = _targets(scene, f)
    p_rad, p_cos = strategy_probabilities(scene)

    if n_rad > 0 and p_rad is None:
        raise DegenerateStrategyError(
            "n_radiance > 0 but total radiance is zero: the "
            "radiance-proportional strategy has no probability mass"
        )
    if n_cos > 0 and p_cos is None:
        raise DegenerateStrategyError(
            "n_cosine > 0 but total clamped cosine is zero: the "
            "cosine-proportional strategy has no probability mass"
        )

    def denom(i):
        d = 0.0
        if p_rad is not None:
            d += n_rad * p_rad[i]
        if p_cos is not None:
            d += n_cos * p_cos[i]
        return d

    for i, t in enumerate(targets):
        if t != 0.0 and denom(i) == 0.0:
            raise UnestimableTargetError(
                f"patch {i}: target t={t!r} is non-zero but both strategies "
                f"assign zero probability; refusing to silently drop the term"
            )

    def contributions(p, n):
        if p is None or n == 0:
            return tuple(None for _ in targets)
        return tuple(
            (targets[i] / denom(i)) if p[i] > 0.0 else None
            for i in range(len(targets))
        )

    return ContributionTable(
        n_radiance=n_rad,
        n_cosine=n_cos,
        targets=targets,
        p_radiance=p_rad,
        p_cosine=p_cos,
        radiance_contribution=contributions(p_rad, n_rad),
        cosine_contribution=contributions(p_cos, n_cos),
    )


# ---------------------------------------------------------------------------
# 确定种子抽样
# ---------------------------------------------------------------------------


def _cumulative(p):
    cum = []
    s = 0.0
    for x in p:
        s += x
        cum.append(s)
    cum[-1] = 1.0  # 消除浮点舍入尾差；末尾零概率项因 bisect_right 仍不可达
    return cum


def _draw_index(rng, cum):
    u = rng.random()  # [0.0, 1.0)
    idx = bisect.bisect_right(cum, u)
    if idx >= len(cum):  # 理论不可达（cum[-1]==1.0 且 u<1），防御性钳制
        idx = len(cum) - 1
    return idx


def estimate(scene: Scene, f_values, n_radiance: int, n_cosine: int, seed: int) -> MisEstimate:
    """运行 balance-heuristic MIS 估计。

    先全部校验再抽样：任何失败都发生在产生结果之前，不留下部分结果。
    抽样顺序固定（先 radiance 策略全部抽样，再 cosine 策略），同一种子
    结果完全确定。
    """
    seed = _check_int(seed, "seed")
    table = mis_contributions(scene, f_values, n_radiance, n_cosine)
    diagnostics = diagnose(scene, f_values)

    rng = random.Random(seed)
    records = []
    draw = 0
    for strategy, count, p, contrib in (
        (RADIANCE, table.n_radiance, table.p_radiance, table.radiance_contribution),
        (COSINE, table.n_cosine, table.p_cosine, table.cosine_contribution),
    ):
        if count == 0:
            continue
        cum = _cumulative(p)
        for _ in range(count):
            idx = _draw_index(rng, cum)
            den = 0.0
            if table.p_radiance is not None:
                den += table.n_radiance * table.p_radiance[idx]
            if table.p_cosine is not None:
                den += table.n_cosine * table.p_cosine[idx]
            weight = count * p[idx] / den
            records.append(
                SampleRecord(
                    draw=draw,
                    strategy=strategy,
                    patch_index=idx,
                    target=table.targets[idx],
                    pdf_radiance=None if table.p_radiance is None else table.p_radiance[idx],
                    pdf_cosine=None if table.p_cosine is None else table.p_cosine[idx],
                    mis_weight=weight,
                    contribution=contrib[idx],
                )
            )
            draw += 1

    value = math.fsum(r.contribution for r in records)

    # 估计量方差：各策略组内无偏样本方差乘该组样本数后求和。
    # 任一被使用策略样本数 < 2 时无法无偏估计，记 None。
    groups = {}
    for r in records:
        groups.setdefault(r.strategy, []).append(r.contribution)
    variance = 0.0
    for contribs in groups.values():
        n = len(contribs)
        if n < 2:
            variance = None
            break
        mean = math.fsum(contribs) / n
        var_s = math.fsum((c - mean) ** 2 for c in contribs) / (n - 1)
        variance += n * var_s

    return MisEstimate(
        estimate=value,
        variance=variance,
        std_error=None if variance is None else math.sqrt(variance),
        n_radiance=table.n_radiance,
        n_cosine=table.n_cosine,
        samples=tuple(records),
        diagnostics=tuple(diagnostics),
    )
