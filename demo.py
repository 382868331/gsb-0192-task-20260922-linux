"""Demo for mis_envlight: estimate vs. exact, plus triggered rejections.

Deterministic inputs, real computation, no network. Runs in well under 8s.
"""

import math

from mis_envlight import (
    Patch,
    Scene,
    estimate,
    direct_sum,
    ValidationError,
    DegenerateProposalError,
)


def build_scene():
    """Deterministic 16-patch hemisphere scene (golden-angle spiral)."""
    n = 16
    golden = math.pi * (3.0 - math.sqrt(5.0))
    patches = []
    for k in range(n):
        z = (k + 0.5) / n
        r = math.sqrt(1.0 - z * z)
        phi = k * golden
        direction = (r * math.cos(phi), r * math.sin(phi), z)
        # Two bright "lights" (k=3 high up, k=11 near the horizon) on a dim
        # deterministic background.
        radiance = 0.2 + 0.05 * ((k * 7) % 5)
        if k == 3:
            radiance += 8.0
        if k == 11:
            radiance += 4.0
        patches.append(Patch(direction=direction, radiance=radiance,
                             weight=2.0 * math.pi / n))
    # Deterministic per-patch f array (finite array, not a callback).
    f = [0.3 + 0.1 * ((k * 5 + 1) % 7) for k in range(n)]
    return Scene(patches), f


def main():
    scene, f = build_scene()
    n_radiance, n_cosine, seed = 64, 64, 20260922

    exact = direct_sum(scene, f)
    res = estimate(scene, f, n_radiance, n_cosine, seed)

    print("=== MIS estimate vs. exact (deterministic seed %d) ===" % seed)
    print(f"patches={len(scene)}  n_radiance={n_radiance}  n_cosine={n_cosine}")
    print(f"exact     = {exact:.10f}")
    print(f"estimate  = {res.estimate:.10f}")
    print(f"abs error = {abs(res.estimate - exact):.3e} "
          f"(rel {abs(res.estimate - exact) / exact:.3e})")
    print(f"estimated stddev of estimator = {res.stddev:.6f}")
    print(f"diagnostics: {len(res.diagnostics)}")
    for d in res.diagnostics[:4]:
        print(f"  [{d.code}] {d.message}")
    print()
    print("=== Per-sample weight explanation (first 4 samples) ===")
    print(res.explain(limit=4))
    print()

    print("=== Diagnostics demo: grazing patch (cos = 0) ===")
    # Patch 1 lies exactly on the horizon: the cosine strategy assigns it
    # zero probability, which the estimator reports as a diagnostic.
    gscene = Scene([
        Patch(direction=(0.0, 0.0, 1.0), radiance=1.0, weight=1.0),
        Patch(direction=(1.0, 0.0, 0.0), radiance=2.0, weight=1.0),
        Patch(direction=(0.0, 0.6, 0.8), radiance=1.0, weight=1.0),
    ])
    gres = estimate(gscene, [1.0, 1.0, 1.0], 8, 8, seed=5)
    for d in gres.diagnostics:
        print(f"  [{d.code}] {d.message}")
    print()

    print("=== Triggered rejection boundaries ===")
    # 1. NaN radiance is rejected with a locatable error.
    try:
        Scene([Patch(direction=(0.0, 0.0, 1.0), radiance=math.nan, weight=1.0)])
    except ValidationError as exc:
        print(f"ValidationError (NaN radiance): {exc}")
    # 2. bool is rejected where an integer count is required.
    try:
        estimate(scene, f, True, 4, seed=1)
    except ValidationError as exc:
        print(f"ValidationError (bool count):   {exc}")
    # 3. A zero-energy scene cannot serve the radiance strategy.
    dark = Scene([Patch(direction=(0.0, 0.0, 1.0), radiance=0.0, weight=1.0)])
    try:
        estimate(dark, [1.0], 2, 0, seed=1)
    except DegenerateProposalError as exc:
        print(f"DegenerateProposalError:        {exc}")


if __name__ == "__main__":
    main()
