"""Demo for mis_envlight: estimate vs. exact, plus triggered rejections.

Deterministic inputs, real computation, no network. Runs in well under 8s.
"""

import math

from mis_envlight import (
    Patch,
    Scene,
    estimate,
    estimate_adaptive,
    variance_minimizing_updater,
    direct_sum,
    ValidationError,
    DegenerateProposalError,
    DIAG_WEIGHTS_CLAMPED,
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

    print()
    print("=== Multi-batch adaptive MIS ===")
    # Small hand-built scene: 3 patches, all quantities explicit so the
    # exact value can be checked by hand:
    #   g = L*f*cos*w = 1*1*1, 2*1*0.5, 1*2*0.25 = 1.0, 1.0, 0.5  ->  S = 2.5
    ascene = Scene([
        Patch(direction=(0.0, 0.0, 1.0), radiance=1.0, weight=1.0),
        Patch(direction=(math.sqrt(0.75), 0.0, 0.5), radiance=2.0, weight=1.0),
        Patch(direction=(math.sqrt(1.0 - 0.25 ** 2), 0.0, 0.25),
              radiance=1.0, weight=1.0),
    ])
    af = [1.0, 1.0, 2.0]
    print(f"hand-built 3-patch scene, exact S = {direct_sum(ascene, af):.3f} "
          f"(hand check: 1.0 + 1.0 + 0.5 = 2.5)")

    # (a) Fixed distribution: no updater, the split never moves.
    fix = estimate_adaptive(ascene, af, n_batches=4, samples_per_batch=8,
                            seed=20260924)
    print(f"[fixed]   estimate={fix.estimate:.6f}  stddev={fix.stddev:.6f}  "
          f"splits={[(b.n_radiance, b.n_cosine) for b in fix.batch_records]}")

    # (b) Distribution changes between batches: a deterministic updater
    # alternates the allocation 20/80 <-> 80/20 after every batch. The
    # estimator stays close to S because every batch is unbiased on its own
    # conditional mixture; old samples keep the pdfs they were drawn with.
    def alternate(brec):
        return (0.8, 0.2) if brec.batch % 2 == 0 else (0.2, 0.8)

    adapt = estimate_adaptive(ascene, af, n_batches=4, samples_per_batch=64,
                              seed=20260924, updater=alternate, floor=0.05)
    print(f"[adaptive] estimate={adapt.estimate:.6f}  "
          f"stddev={adapt.stddev:.6f}  (exact 2.500000)")
    print(adapt.explain())
    print("weights initial -> final: "
          f"({adapt.initial_weights[0]:.3f},{adapt.initial_weights[1]:.3f}) -> "
          f"({adapt.final_weights[0]:.3f},{adapt.final_weights[1]:.3f})")
    # Conditional scoring: batch 0 keeps its original split even though the
    # final batch uses a different one.
    b0, bL = adapt.batch_records[0], adapt.batch_records[-1]
    print(f"conditional scoring: batch 0 scored at "
          f"({b0.n_radiance},{b0.n_cosine}), final batch at "
          f"({bL.n_radiance},{bL.n_cosine}); old records are not rescored")

    # (c) The built-in variance-minimizing updater on the 16-patch scene:
    # it nudges the split toward whichever strategy's single-draw
    # contributions varied more in the preceding batch.
    var = estimate_adaptive(scene, f, n_batches=4, samples_per_batch=64,
                            seed=20260922,
                            updater=variance_minimizing_updater(),
                            floor=0.05)
    splits = [(b.n_radiance, b.n_cosine) for b in var.batch_records]
    print(f"[variance updater] estimate={var.estimate:.6f} "
          f"(exact {direct_sum(scene, f):.6f}) splits={splits}")

    print()
    print("=== Adaptive boundary: positive-probability floor enforced ===")
    # Updater asks for (1.0, 0.0) — zero probability for cosine. It is not
    # applied: the request is clamped to the floor and reported explicitly.
    clamped = estimate_adaptive(
        ascene, af, n_batches=2, samples_per_batch=10, seed=1, floor=0.1,
        updater=lambda br: (1.0, 0.0))
    for b in clamped.batch_records:
        print(f"  batch {b.batch}: realized split "
              f"({b.n_radiance},{b.n_cosine}) "
              f"shares=({b.weight_radiance:.2f},{b.weight_cosine:.2f})")
    for d in clamped.diagnostics:
        if d.code == DIAG_WEIGHTS_CLAMPED:
            print(f"  [{d.code}] {d.message}")

    print()
    print("=== Adaptive rejections ===")
    # 1. A malformed updater return (NaN) aborts the whole run — no
    # half-updated result is returned.
    try:
        estimate_adaptive(ascene, af, 3, 8, seed=1,
                          updater=lambda br: (0.5, float("nan")))
    except ValidationError as exc:
        print(f"ValidationError (bad updater): {exc}")
    # 2. floor=0.5 cannot be realized with 3 draws per batch.
    try:
        estimate_adaptive(ascene, af, 2, 3, seed=1, floor=0.5)
    except ValidationError as exc:
        print(f"ValidationError (unrealizable floor): {exc}")
    # 3. Adaptive runs keep both strategies active every batch, so a
    # zero-energy scene's degenerate radiance proposal is rejected up front.
    try:
        estimate_adaptive(dark, [1.0], 2, 8, seed=1)
    except DegenerateProposalError as exc:
        print(f"DegenerateProposalError (adaptive): {exc}")


if __name__ == "__main__":
    main()
