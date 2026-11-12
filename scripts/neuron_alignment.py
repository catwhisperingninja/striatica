"""Reproduce A-NAP-001 static-weight measurements; never execute the host model.

Run with the repository's existing Poetry environment. Inputs are verified,
locally cached dictionary NPZ and host safetensors shard, not remote APIs.
Outputs contain no model weights,
semantic labels, activations, or credentials. This is an analysis deliverable,
not a production pipeline or a replacement for the comparability memo.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from threadpoolctl import threadpool_limits

DICT_REV = "50eec2f25c60545a9a74c1c3a26a0afdd0b4b872"
HOST_REV = "c5ebcd40d208330abc697524c919956e692655cf"
DICT_SHA = "bef8a26a3b18e333bafa2c35fe863a2b3dfe26258450e70a5b7480afa8c35ae9"
HOST_SHA = "96c111d3dcdbde9271595e463b5d9f7fc4810ad8b79e736309c0a1833e6c0d35"
CONFIG_SHA = "56ee4e528fedfe70b2fabb0704eec9f2ef94461de936928220a3d6e7fac5ef98"


def normalized(x):
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(x).all() or np.any(norms == 0):
        raise ValueError("Nonfinite or zero input direction")
    return x / norms[:, None]


def summarize(x):
    return dict(
        zip(
            ["min", "q01", "q25", "median", "q75", "q99", "max"],
            map(float, np.quantile(x, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])),
        )
    )


def alignment(dec, enc, host, raw_down=None, block=256):
    """Keep local S1–S5 separate from the paper's shared sign-candidate score."""
    down, gate, up = host
    dn, en = normalized(dec), normalized(enc)
    ng, nu = np.linalg.norm(gate, axis=1), np.linalg.norm(up, axis=1)
    np_, nm = np.linalg.norm(up + gate, axis=1), np.linalg.norm(up - gate, axis=1)
    out = {
        k: np.empty(len(dec), dtype=np.float64)
        for k in [
            "A_dec_frame_matched",
            "A_gate",
            "A_up",
            "S5_abs_cos_pr",
            "S5_squared_cos_pr",
            "paper_joint_score",
        ]
    }
    for k in ["argmax_dec", "argmax_gate", "argmax_up", "paper_neuron", "paper_sign"]:
        out[k] = np.empty(len(dec), dtype=np.int64)
    if raw_down is not None:
        out["A_dec_raw_diagnostic"] = np.empty(len(dec))
        out["argmax_dec_raw_diagnostic"] = np.empty(len(dec), dtype=np.int64)
    for start in range(0, len(dec), block):
        stop = min(start + block, len(dec))
        sl = slice(start, stop)
        d = dn[sl] @ down.T
        # Retain unnormalized host dot products for exact effective-encoder sums.
        g = en[sl] @ gate.T
        u = en[sl] @ up.T
        plus = np.minimum(d, (u + g) / np_)
        minus = np.minimum(-d, (u - g) / nm)
        pi, mi = plus.argmax(axis=1), minus.argmax(axis=1)
        rows = np.arange(stop - start)
        ps, ms = plus[rows, pi], minus[rows, mi]
        positive = ps >= ms
        out["paper_joint_score"][sl] = np.maximum(ps, ms)
        out["paper_neuron"][sl] = np.where(positive, pi, mi)
        out["paper_sign"][sl] = np.where(positive, 1, -1)
        for scores, name, index in [
            (d, "A_dec_frame_matched", "argmax_dec"),
            (g / ng, "A_gate", "argmax_gate"),
            (u / nu, "A_up", "argmax_up"),
        ]:
            ids = scores.argmax(axis=1)
            out[index][sl] = ids
            out[name][sl] = scores[rows, ids]
        mass, energy = np.abs(d), np.square(d)
        out["S5_abs_cos_pr"][sl] = mass.sum(axis=1) ** 2 / energy.sum(axis=1)
        out["S5_squared_cos_pr"][sl] = energy.sum(axis=1) ** 2 / np.square(energy).sum(
            axis=1
        )
        if raw_down is not None:
            raw = dn[sl] @ raw_down.T
            ids = raw.argmax(axis=1)
            out["A_dec_raw_diagnostic"][sl] = raw[rows, ids]
            out["argmax_dec_raw_diagnostic"][sl] = ids
    out["S4_dec_gate_agree"] = out["argmax_dec"] == out["argmax_gate"]
    out["S4_dec_up_agree"] = out["argmax_dec"] == out["argmax_up"]
    return out


def prefix_pr(gram, order, row_normalize):
    """Exact Gram-trace PR of every prefix, with and without subset centering.

    For K=X Xᵀ and H=I−11ᵀ/n, tr((HKH)²) is
    ||K||²_F − 2||K1||²/n + (1ᵀK1)²/n². Maintain all three
    terms as one row/column is added. No eigensolver, sampling, or UMAP.
    """
    n_total = len(order)
    values = np.full((2, n_total + 1), np.nan)
    row_sums = np.zeros(n_total)
    scale = 1 / np.sqrt(np.diag(gram)) if row_normalize else np.ones(n_total)
    trace = square = row_square = total = 0.0
    for i, j in enumerate(order):
        v = gram[j, order[:i]] * scale[j] * scale[order[:i]]
        a = gram[j, j] * scale[j] ** 2
        s = float(v.sum())
        v2 = float(np.einsum("i,i->", v, v))
        cross = float(np.einsum("i,i->", row_sums[:i], v))
        row_square += 2 * cross + v2 + (s + a) ** 2
        row_sums[:i] += v
        row_sums[i] = s + a
        trace += a
        square += 2 * v2 + a * a
        total += 2 * s + a
        n = i + 1
        values[0, n] = trace * trace / square
        if n > 1:
            ct = trace - total / n
            cq = square - 2 * row_square / n + total * total / (n * n)
            if ct > np.finfo(float).eps * trace * 32 and cq > 0:
                values[1, n] = ct * ct / cq
    return values


def curves(dec, enc, scores, metrics, neuron_count):
    """Uniform-weight decoder/encoder PR on both sides of declared cutoff grids."""
    result = []
    for family, x in [("decoder", dec), ("encoder", enc)]:
        print(f"  {family}: full float64 Gram", flush=True)
        gram = x @ x.T
        for metric in metrics:
            a = scores[metric]
            order = np.argsort(a, kind="stable")
            is_sparsity = metric.startswith("S5")
            # Cover the statistic's entire mathematical range, not a chosen split.
            grid = (
                np.geomspace(1, neuron_count, 81)
                if is_sparsity
                else np.linspace(-1, 1, 201)
            )
            for row_normalize in [False, True]:
                for side in ["at_or_below", "above"]:
                    seq = order if side == "at_or_below" else order[::-1]
                    with threadpool_limits(limits=1):
                        vals = prefix_pr(gram, seq, row_normalize)
                    for tau in grid:
                        nlow = int(np.searchsorted(a[order], tau, side="right"))
                        n = nlow if side == "at_or_below" else len(a) - nlow
                        for center in [False, True]:
                            pr = vals[int(center), n]
                            result.append(
                                {
                                    "vector_family": family,
                                    "partition_metric": metric,
                                    "tau": float(tau),
                                    "side": side,
                                    "n": n,
                                    "center": center,
                                    "row_normalize": row_normalize,
                                    "weights": "uniform_per_latent",
                                    "pr": float(pr) if np.isfinite(pr) else None,
                                    "unit": "effective_dimension",
                                }
                            )
            print(f"    {metric}: prefixes complete", flush=True)
        del gram
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--host-shard", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--null-runs", type=int, default=3)
    args = parser.parse_args()
    if args.null_runs < 1:
        parser.error("--null-runs must be at least 1; N3 is a required control")
    if args.output.exists():
        parser.error(
            "--output must be a new directory; existing results are not overwritten"
        )
    started = time.monotonic()
    for path, expected in (
        (args.dictionary, DICT_SHA),
        (args.host_shard, HOST_SHA),
        (args.config, CONFIG_SHA),
    ):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            parser.error(f"Pinned input SHA-256 mismatch: {path.name}")
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    d = config["hidden_size"]
    nff = config["intermediate_size"]
    assert d == 2304
    with np.load(args.dictionary) as data:
        dec, enc = data["W_dec"].astype(np.float64), data["W_enc"].T.astype(np.float64)
    assert dec.shape == enc.shape == (16384, d)
    dn, en = np.linalg.norm(dec, axis=1), np.linalg.norm(enc, axis=1)
    prefix = "model.layers.12."
    with safe_open(args.host_shard, framework="numpy") as host:
        pre = 1 + host.get_tensor(prefix + "pre_feedforward_layernorm.weight").astype(
            np.float64
        )
        post = 1 + host.get_tensor(prefix + "post_feedforward_layernorm.weight").astype(
            np.float64
        )
        down = host.get_tensor(prefix + "mlp.down_proj.weight").T.astype(np.float64)
        gate = host.get_tensor(prefix + "mlp.gate_proj.weight").astype(np.float64) * pre
        up = host.get_tensor(prefix + "mlp.up_proj.weight").astype(np.float64) * pre
    assert down.shape == gate.shape == up.shape == (nff, d)
    assert np.all(pre > 0) and np.all(post > 0)
    directions = (normalized(down * post), gate, up)
    physical = min(1, nff / len(dec))
    sign_capacity = min(1, 2 * nff / len(dec))
    print(
        f"N4 distinct-assignment capacity: physical={physical}; sign-split={sign_capacity}. Not an aligned-fraction bound.",
        flush=True,
    )
    metadata = {
        "status": "PROVISIONAL_PENDING_COMPARABILITY_AND_VALIDATION",
        "dictionary_revision": DICT_REV,
        "host_revision": HOST_REV,
        "dictionary_sha256": DICT_SHA,
        "host_shard_sha256": HOST_SHA,
        "host_config_sha256": CONFIG_SHA,
        "d_model": d,
        "d_ff": nff,
        "d_sae": len(dec),
        "float_dtype": "float64",
        "N1_signed_cos_mean": 0,
        "N1_cos_rms": float(1 / np.sqrt(d)),
        "N2_positive_max_approx": float(np.sqrt(2 * np.log(nff) / d)),
        "N4_physical_injective_capacity": physical,
        "N4_sign_candidate_injective_capacity": sign_capacity,
        "N4_interpretation": "Distinct assignments only; many-to-one aligned fraction is not bounded by physical capacity.",
        "N3": {
            "runs": args.null_runs,
            "seed_start": 20261111,
            "rng": "numpy.default_rng/PCG64",
            "distribution": "independent isotropic Gaussian decoder and encoder rows, each rescaled to the real row's Euclidean norm",
            "preserved": [
                "shape",
                "per-latent decoder norm",
                "per-latent encoder norm",
            ],
            "not_preserved": [
                "real dictionary geometry",
                "encoder-decoder within-latent dependence",
                "activation distribution",
            ],
            "exchangeable_run_p_resolution": 1 / (args.null_runs + 1),
        },
        "coordinate_frames": {
            "decoder_primary": "released W_dec vs normalized down_proj.T * (1 + post_feedforward_layernorm.weight)",
            "encoder_primary": "released W_enc.T vs gate/up rows * (1 + pre_feedforward_layernorm.weight)",
            "raw_decoder_diagnostic": "released W_dec vs raw down_proj.T; intentionally mismatched output frame",
            "paper_plus": "encoder up+gate; decoder +down, after frame matching",
            "paper_minus": "encoder up-gate; decoder -down, literal paper section 2.1",
            "paper_score": "max over shared neuron/sign candidates of min(encoder_cos, decoder_cos)",
            "spectral_pr": "released dictionary row geometry, without host-gain reweighting; center separately within each partition",
        },
        "pr_curve_controls": {
            "matched_null_metrics": ["A_dec_frame_matched", "S5_squared_cos_pr"],
            "real_only_uncontrolled_metrics": [
                "A_dec_raw_diagnostic",
                "S5_abs_cos_pr",
                "paper_joint_score",
            ],
            "interpretation": "Real-only PR curves have no matched-null PR baseline; they are descriptive, not controlled comparisons. Per-latent score availability does not imply a PR null curve exists.",
        },
        "S5": {
            "abs_cos_pr": "sum(abs(cos))**2 / sum(cos**2)",
            "squared_cos_pr": "sum(cos**2)**2 / sum(cos**4)",
            "interpretation": "Nonorthogonal-frame response concentration, not unique sparse reconstruction coefficients.",
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "hub_removed_control": "NOT COMPUTED: historical row-removal recipe/IDs not recovered; no invariance claim",
        "minimum_resolvable_pr_difference": None,
        "resolution_limitation": "No preregistered power/alpha or sufficient independent null runs for a formal PR detection limit; cosine spread is not PR uncertainty.",
    }
    (args.output / "manifest-PROVISIONAL.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    for run in range(args.null_runs + 1):
        name = "real" if run == 0 else f"null-{run}"
        if run:
            rng = np.random.default_rng(20261110 + run)
            xd = normalized(rng.standard_normal(dec.shape)) * dn[:, None]
            xe = normalized(rng.standard_normal(enc.shape)) * en[:, None]
        else:
            xd, xe = dec, enc
        print(
            f"{name}: alignment start; elapsed {time.monotonic() - started:.1f}s",
            flush=True,
        )
        a = alignment(xd, xe, directions, normalized(down) if run == 0 else None)
        np.savez_compressed(
            args.output / f"{name}-per-latent.npz",
            **a,
            dictionary_revision=DICT_REV,
            host_revision=HOST_REV,
        )
        stats = {k: summarize(v) for k, v in a.items() if v.dtype == np.float64}
        stats.update(
            {
                k: {"count": int(a[k].sum()), "n": len(a[k])}
                for k in ["S4_dec_gate_agree", "S4_dec_up_agree"]
            }
        )
        (args.output / f"{name}-summary.json").write_text(
            json.dumps(stats, indent=2) + "\n"
        )
        controls = metadata["pr_curve_controls"]
        metrics = controls["matched_null_metrics"] + (
            controls["real_only_uncontrolled_metrics"] if run == 0 else []
        )
        print(
            f"{name}: alignment saved; elapsed {time.monotonic() - started:.1f}s",
            flush=True,
        )
        points = curves(xd, xe, a, metrics, nff)
        (args.output / f"{name}-curves-PROVISIONAL.json").write_text(
            json.dumps(points, separators=(",", ":")) + "\n"
        )
        print(
            f"{name}: {len(points)} curve points saved; elapsed {time.monotonic() - started:.1f}s",
            flush=True,
        )
    metadata["elapsed_seconds"] = time.monotonic() - started
    (args.output / "manifest-PROVISIONAL.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        "COMPLETE — provisional outputs; do not infer functional consequences.",
        flush=True,
    )


if __name__ == "__main__":
    main()
