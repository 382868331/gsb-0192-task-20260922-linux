"""Demo for mis_envlight: estimate vs. exact, plus triggered rejections.

Deterministic inputs, real computation, no network. Runs in well under 8s.
"""

import math

from mis_envlight import (
    Patch,
    Scene,
    estimate,
    estimate_with_control,
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

    print("=== Control variate: two-phase estimator (same total budget) ===")
    # Correlated control: neighbor-smoothed f. The caller knows its exact
    # discrete integral G (computed here once, exactly, as the caller would).
    g = [0.5 * (f[k] + f[(k + 1) % len(f)]) for k in range(len(f))]
    G = direct_sum(scene, g)
    # Budget: pilot 16+16 + main 48+48 = 128 = the plain run's 64+64.
    cv = estimate_with_control(scene, f, g, G, 48, 48, 16, 16, seed=seed)
    print(f"pilot: n_radiance={cv.n_pilot_radiance} "
          f"n_cosine={cv.n_pilot_cosine} | main: n_radiance={cv.n_radiance} "
          f"n_cosine={cv.n_cosine} | total={cv.total_samples}")
    print(f"beta (fixed by pilot) = {cv.beta:.6f}")
    print(f"controlled estimate = {cv.estimate:.10f} "
          f"(residual {cv.residual_estimate:.10f} + beta*G)")
    print(f"abs error = {abs(cv.estimate - exact):.3e}   "
          f"estimated stddev = {cv.stddev:.6f}")
    print(f"plain run stddev    = {res.stddev:.6f} (same 128-sample budget)")
    # Finite-seed sweep at equal budget: observed spread of both estimators.
    # This is a property of this scene/control/seed set, not a guarantee.
    def spread(estimates):
        m = math.fsum(estimates) / len(estimates)
        return math.fsum((e - m) ** 2 for e in estimates) / (len(estimates) - 1)
    plain_runs, ctrl_runs = [], []
    for s in range(12):
        plain_runs.append(estimate(scene, f, 64, 64, seed=1000 + s).estimate)
        ctrl_runs.append(estimate_with_control(scene, f, g, G, 48, 48, 16, 16,
                                               seed=1000 + s).estimate)
    print(f"12-seed spread: plain {spread(plain_runs):.3e} vs "
          f"controlled {spread(ctrl_runs):.3e} (scenario-specific)")
    print()

    print("=== Diagnostics demo: grazing patch (cos = 0) ===")    # Patch 1 lies exactly on the horizon: the cosine strategy assigns it
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
