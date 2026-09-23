"""固定输入演示：离散环境光 balance-heuristic MIS 估计 vs 精确求和。

真实计算，无 sleep、无预录结果。运行：python demo.py
"""

import math

from mis_envlight import (
    DegenerateStrategyError,
    Patch,
    SceneValidationError,
    build_scene,
    diagnose,
    estimate,
    exact_integral,
)


def make_demo_scene():
    """确定性生成 24 片半球场景（Fibonacci 半球 + 两个亮斑）。"""
    n = 24
    normal = (0.0, 0.0, 1.0)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    patches = []
    for i in range(n):
        z = 1.0 - (i + 0.5) / n  # (0,1] 等面积分层
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = i * golden
        direction = (r * math.cos(phi), r * math.sin(phi), z)
        radiance = 0.3
        if i == 5:
            radiance = 6.0  # 主亮斑
        elif i == 17:
            radiance = 2.5  # 次亮斑
        weight = 2.0 * math.pi / n  # 等立体角
        patches.append(Patch(direction, radiance, weight))
    f_values = [0.2 + 0.8 * ((i * 0.6180339887498949) % 1.0) for i in range(n)]
    return build_scene(patches, normal), f_values


def main():
    scene, f_values = make_demo_scene()
    exact = exact_integral(scene, f_values)
    print(f"场景: {scene.size} 片等立体角小片, 法线 (0,0,1)")
    print(f"精确值 exact = sum L*f*cos*w = {exact:.12f}")

    result = estimate(scene, f_values, n_radiance=128, n_cosine=128, seed=20260922)
    err = result.estimate - exact
    print(
        f"MIS 估计 = {result.estimate:.12f} "
        f"(n_radiance={result.n_radiance}, n_cosine={result.n_cosine}, seed=20260922)"
    )
    print(f"标准误 std_error = {result.std_error:.6g}, "
          f"绝对误差 = {err:.6g}, 相对误差 = {abs(err) / exact:.3%}")

    diags = diagnose(scene, f_values)
    print(f"诊断: {len(diags)} 条")
    for d in diags[:3]:
        print(f"  [{d.code}] {d.message}")

    print("逐样本权重解释（前 3 次抽样）:")
    for rec in result.samples[:3]:
        print("  " + rec.explain())

    # 实际触发的拒绝边界 1：NaN 辐亮度 → 可定位的校验错误。
    try:
        bad = [Patch((0.0, 0.0, 1.0), 1.0, 0.5),
               Patch((0.0, 0.6, 0.8), float("nan"), 0.5)]
        build_scene(bad, (0.0, 0.0, 1.0))
    except SceneValidationError as exc:
        print(f"拒绝（非法输入）: {type(exc).__name__}: {exc}")

    # 实际触发的拒绝边界 2：零能量场景请求 radiance 策略 → 退化策略错误。
    try:
        dark = build_scene(
            [Patch((0.0, 0.0, 1.0), 0.0, 0.5),
             Patch((0.0, 0.6, 0.8), 0.0, 0.5)],
            (0.0, 0.0, 1.0),
        )
        estimate(dark, [1.0, 1.0], n_radiance=1, n_cosine=1, seed=0)
    except DegenerateStrategyError as exc:
        print(f"拒绝（退化策略）: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
