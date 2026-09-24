"""Demo for mis_envlight: estimate vs. exact, plus triggered rejections.

Deterministic inputs, real computation, no network. Runs in well under 8s.
"""

import math

from mis_envlight import (
    DEFAULT_MIX_FLOOR,
    Patch,
    Scene,
    BatchedEstimator,
    estimate,
    estimate_batched,
    direct_sum,
    proposal_pdfs,
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

    print("=== Batched adaptive finite-mixture sampling ===")
    # Two batches with *different* mixture weights. Each sample freezes the
    # conditional pdf that was actually in force when it was drawn; the
    # second batch changing alpha never touches the first batch's records.
    bplan = [(256, 0.10), (256, 0.90)]
    bres = estimate_batched(scene, f, bplan, seed=20260924)
    print(f"plan: {[(n, a) for n, a in bplan]} "
          f"(mix_floor={DEFAULT_MIX_FLOOR})")
    print(f"exact          = {exact:.10f}")
    print(f"batched est.   = {bres.estimate:.10f}")
    print(f"abs error      = {abs(bres.estimate - exact):.3e} "
          f"(rel {abs(bres.estimate - exact) / exact:.3e})")
    print(f"stddev of mean = {bres.stddev:.6f}")
    for b in bres.batches:
        print(f"  batch {b.batch_index}: n={b.n_samples:<4} alpha={b.alpha:<5} "
              f"batch_mean={b.mean:.6f}")
    # Demonstrate that old records kept their own frozen conditional pdf:
    # the same patch sampled in both batches carries different pdfs.
    seen = {}
    for r in bres.records:
        seen.setdefault((r.batch_index, r.patch_index), r.pdf_mixture)
    example = next((i for i in range(len(scene))
                    if (0, i) in seen and (1, i) in seen
                    and seen[(0, i)] != seen[(1, i)]), None)
    if example is not None:
        print(f"frozen-pdf check on patch {example}: "
              f"batch0 pdf={seen[(0, example)]:.6f}  "
              f"batch1 pdf={seen[(1, example)]:.6f}  (not back-recomputed)")
    print()

    print("=== Stateful runner: weight chosen from the previous batch ===")
    runner = BatchedEstimator(scene, f, seed=20260924)
    first = runner.run_batch(n_samples=256, alpha=0.5)
    # A deterministic, fully local update rule: steer alpha toward the
    # proposal whose batch-averaged integrand mass looks larger. This is
    # just an example adaptation; the library imposes no learning rule.
    pL, pC = proposal_pdfs(scene)

    def avg_second_moment(batch, own_pdf):
        vals = [r.target_value ** 2 / (own_pdf[r.patch_index] * r.pdf_mixture)
                for r in batch.records if r.target_value > 0.0]
        return sum(vals) / len(vals) if vals else 0.0

    mp, mc = avg_second_moment(first, pL), avg_second_moment(first, pC)
    alpha1 = max(DEFAULT_MIX_FLOOR,
                 min(1.0 - DEFAULT_MIX_FLOOR,
                     math.sqrt(mp) / (math.sqrt(mp) + math.sqrt(mc))))
    runner.run_batch(n_samples=256, alpha=alpha1)
    adapted = runner.result()
    print(f"batch0 alpha=0.5 -> adapted batch1 alpha={alpha1:.4f}")
    print(f"adaptive estimate = {adapted.estimate:.10f} "
          f"(stddev {adapted.stddev:.6f})")
    print()

    print("=== Fixed-distribution boundary (alpha constant across batches) ===")
    fixed = estimate_batched(scene, f, [(200, 0.5), (200, 0.5)], seed=7)
    one = estimate_batched(scene, f, [(400, 0.5)], seed=7)
    print(f"two fixed batches: est={fixed.estimate:.10f}; "
          f"one batch same alpha: est={one.estimate:.10f}")
    print(f"identical draw sequence: "
          f"{[r.patch_index for r in fixed.records] == [r.patch_index for r in one.records]}")
    # Zero-energy scene: radiance component has no mass; the mixture is
    # renormalized onto cosine and flagged, regardless of alpha.
    dark = Scene([Patch(direction=(0.0, 0.0, 1.0), radiance=0.0, weight=1.0),
                  Patch(direction=(0.0, 0.6, 0.8), radiance=0.0, weight=1.0)])
    dres = estimate_batched(dark, [1.0, 1.0], [(64, 0.2), (64, 0.8)], seed=2)
    zc = [d for d in dres.diagnostics if d.code == "ZERO_MASS_COMPONENT"]
    print(f"zero-energy scene: estimate={dres.estimate}; "
          f"{len(zc)} ZERO_MASS_COMPONENT diagnostic; "
          f"batch pdfs identical: {dres.batches[0].pdf == dres.batches[1].pdf}")
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
    onepatch = Scene([Patch(direction=(0.0, 0.0, 1.0), radiance=0.0,
                            weight=1.0)])
    try:
        estimate(onepatch, [1.0], 2, 0, seed=1)
    except DegenerateProposalError as exc:
        print(f"DegenerateProposalError:        {exc}")
    # 4. Batched mixture: alpha must keep a positive floor on both existing
    #    strategies; 0.0 (switching the cosine strategy off) is rejected,
    #    and the message locates the offending batch.
    try:
        estimate_batched(scene, f, [(16, 0.5), (16, 0.0)], seed=1)
    except ValidationError as exc:
        print(f"ValidationError (mix floor):    {exc}")
    # 5. A stateful runner rejects a bad batch atomically: batch count and
    #    the random stream are untouched.
    runner2 = BatchedEstimator(scene, f, seed=9)
    runner2.run_batch(16, 0.5)
    try:
        runner2.run_batch(16, 1.0)  # violates the floor
    except ValidationError as exc:
        print(f"ValidationError (stateful):     {exc}")
    print(f"runner still reports n_batches={runner2.n_batches}, "
          f"total_samples={runner2.total_samples} (failed batch not committed)")
    # 6. Both proposal components zero-mass: no mixture exists at all.
    flatdark = Scene([Patch(direction=(1.0, 0.0, 0.0), radiance=0.0,
                            weight=1.0)])
    try:
        estimate_batched(flatdark, [1.0], [(8, 0.5)], seed=1)
    except DegenerateProposalError as exc:
        print(f"DegenerateProposalError (mix):  {exc}")


if __name__ == "__main__":
    main()
