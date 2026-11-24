#!/usr/bin/env python3
"""Reproduce the QXU6028 G08 L3 spatial inverse analysis.

Implements the Practical Methods Guide exactly: 10x10 first-difference
regularisation, projected zero-mean Tikhonov inversion, eta sensitivity,
feature localisation, central/off-centre point-spread resolution, 500-vector
noise detection limit, and the announced 0.030 pass threshold.

Install: python -m pip install numpy matplotlib
Run from any directory: python reproduce_L3.py
Inputs default to data/regular_G08 beside this script. Generated files go to
outputs/ beside this script. Use --data-dir and --output-dir to override them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NPIX = 10
N = NPIX * NPIX
NOISE_REPLICATES = 500
NOISE_SEED = 6028
SPEC_THRESHOLD = 0.030
ETA_FACTORS = np.array([0.01, 0.1, 1.0, 10.0, 100.0])


def gradient_matrix(npix: int = NPIX) -> np.ndarray:
    rows: list[np.ndarray] = []
    n = npix * npix
    for i in range(npix):
        for j in range(npix):
            k = i * npix + j
            if j < npix - 1:
                row = np.zeros(n)
                row[k], row[k + 1] = -1.0, 1.0
                rows.append(row)
            if i < npix - 1:
                row = np.zeros(n)
                row[k], row[k + npix] = -1.0, 1.0
                rows.append(row)
    return np.asarray(rows)


def projector(n: int = N) -> np.ndarray:
    one = np.ones((n, 1)) / np.sqrt(n)
    return np.eye(n) - one @ one.T


def prepare_solver(A: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    P = projector(A.shape[1])
    eta0 = 1e-2 * np.linalg.norm(A, 2) ** 2 / np.linalg.norm(L, 2) ** 2
    return P, L, eta0


def make_inverse(A: np.ndarray, L: np.ndarray, P: np.ndarray, eta: float) -> np.ndarray:
    n = A.shape[1]
    M = P @ (A.T @ A + eta * (L.T @ L)) @ P + 1e-9 * np.eye(n)
    inverse = np.linalg.lstsq(M, P @ A.T, rcond=None)[0]
    return P @ inverse


def reconstruct(dvec: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    m = inverse @ dvec
    return m - m.mean()


def map_coordinates(npix: int = NPIX) -> np.ndarray:
    return np.linspace(-2.7, 2.7, npix)


def extrema_features(m: np.ndarray) -> dict[str, object]:
    grid = m.reshape(NPIX, NPIX)
    coords = map_coordinates()
    max_position = np.unravel_index(int(np.argmax(grid)), grid.shape)
    min_position = np.unravel_index(int(np.argmin(grid)), grid.shape)
    return {
        "largest_positive": {
            "row": int(max_position[0]),
            "column": int(max_position[1]),
            "x_probe_pitch": float(coords[max_position[1]]),
            "y_probe_pitch": float(coords[max_position[0]]),
            "amplitude": float(grid[max_position]),
        },
        "largest_negative": {
            "row": int(min_position[0]),
            "column": int(min_position[1]),
            "x_probe_pitch": float(coords[min_position[1]]),
            "y_probe_pitch": float(coords[min_position[0]]),
            "amplitude": float(grid[min_position]),
        },
    }


def weighted_fwhm(map_vector: np.ndarray, centre: tuple[int, int]) -> float:
    grid = map_vector.reshape(NPIX, NPIX)
    weights = np.maximum(grid, 0.0)
    coords = map_coordinates()
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    r2 = (xx - coords[centre[1]]) ** 2 + (yy - coords[centre[0]]) ** 2
    denominator = 2.0 * np.sum(weights)
    if denominator <= 0:
        return float("nan")
    sigma2 = float(np.sum(weights * r2) / denominator)
    return float(2.355 * np.sqrt(max(sigma2, 0.0)))


def resolution_test(A: np.ndarray, inverse: np.ndarray) -> dict[str, object]:
    centres = {"central": (NPIX // 2, NPIX // 2), "off_centre": (2, 7)}
    results: dict[str, object] = {}
    for label, centre in centres.items():
        q = np.zeros(N)
        q[centre[0] * NPIX + centre[1]] = 1.0
        q -= q.mean()
        recovered = reconstruct(A @ q, inverse)
        recovered_grid = recovered.reshape(NPIX, NPIX)
        peak = float(recovered_grid[centre])
        half_width_discrete = float("nan")
        if peak > 0:
            # The weighted estimator below is the guide's recommended fallback
            # when a 10x10 discrete half-maximum contour is coarse.
            half_width_discrete = weighted_fwhm(recovered, centre)
        results[label] = {
            "injected_row": centre[0],
            "injected_column": centre[1],
            "peak_recovered": peak,
            "fwhm_probe_pitch": half_width_discrete,
            "map": recovered.tolist(),
        }
    results["conservative_resolution_probe_pitch"] = float(
        max(results["central"]["fwhm_probe_pitch"], results["off_centre"]["fwhm_probe_pitch"])
    )
    return results


def save_map_figure(
    m: np.ndarray,
    features: dict[str, object],
    threshold: float,
    target: Path,
) -> None:
    grid = m.reshape(NPIX, NPIX)
    coords = map_coordinates()
    vmax = float(np.max(np.abs(grid)))
    fig, ax = plt.subplots(figsize=(7.5, 6.8), constrained_layout=True)
    image = ax.imshow(
        grid,
        origin="lower",
        extent=[coords[0] - 0.3, coords[-1] + 0.3, coords[0] - 0.3, coords[-1] + 0.3],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(coords)
    ax.set_yticks(coords)
    ax.grid(color="white", alpha=0.35, lw=0.65)
    positive = features["largest_positive"]
    negative = features["largest_negative"]
    ax.scatter([positive["x_probe_pitch"]], [positive["y_probe_pitch"]], marker="+", s=180,
               color="#8B1E3F", lw=2.2, label=f"positive {positive['amplitude']:.4f}")
    ax.scatter([negative["x_probe_pitch"]], [negative["y_probe_pitch"]], marker="_", s=220,
               color="#123B5D", lw=2.6, label=f"negative {negative['amplitude']:.4f}")
    ax.annotate("largest +", (positive["x_probe_pitch"], positive["y_probe_pitch"]), xytext=(6, 8),
                textcoords="offset points", fontsize=9)
    ax.annotate("largest −", (negative["x_probe_pitch"], negative["y_probe_pitch"]), xytext=(6, -16),
                textcoords="offset points", fontsize=9)
    ax.set_xlabel("x (probe pitches)")
    ax.set_ylabel("y (probe pitches)")
    ax.set_title(f"G08 L3 zero-mean reconstructed map | max|m|={vmax:.5f}")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    colourbar = fig.colorbar(image, ax=ax, label="Reconstructed non-uniformity")
    colourbar.ax.axhline(threshold, color="#8B1E3F", lw=1.0)
    colourbar.ax.axhline(-threshold, color="#123B5D", lw=1.0)
    fig.savefig(target, dpi=240)
    plt.close(fig)


def save_method_figure(
    eta_factors: np.ndarray,
    residuals: np.ndarray,
    roughness: np.ndarray,
    resolution: dict[str, object],
    delta95: float,
    threshold: float,
    target: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), constrained_layout=True)
    ax = axes[0]
    ax.loglog(roughness, residuals, "o-", color="#284F73", lw=1.8)
    for factor, x, y in zip(eta_factors, roughness, residuals):
        ax.annotate(f"×{factor:g}", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ref = int(np.flatnonzero(eta_factors == 1.0)[0])
    ax.scatter([roughness[ref]], [residuals[ref]], s=110, color="#1A8A70", zorder=3, label="eta0 reference")
    ax.set_xlabel(r"Roughness $\|Lm\|_2$")
    ax.set_ylabel(r"Data residual $\|Am-\bar d\|_2$")
    ax.set_title("Regularisation sensitivity")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False)

    ax = axes[1]
    labels = ["central", "off-centre"]
    resolution_keys = ["central", "off_centre"]
    widths = [resolution[key]["fwhm_probe_pitch"] for key in resolution_keys]
    bars = ax.bar(labels, widths, color=["#C96C5B", "#D48120"], alpha=0.88)
    for bar, width in zip(bars, widths):
        ax.text(bar.get_x() + bar.get_width() / 2, width + 0.03, f"{width:.3f}", ha="center", fontsize=9)
    ax.axhline(max(widths), color="#284F73", ls="--", lw=1.2, label=f"conservative resolution = {max(widths):.3f}")
    ax.set_ylabel("FWHM (probe pitches)")
    ax.set_title(f"Point-spread resolution and noise limit\n95% noise limit = {delta95:.6f}; threshold = {threshold:.3f}")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("G08 L3 method evidence", fontsize=15)
    fig.savefig(target, dpi=240)
    plt.close(fig)


def save_noise_figure(noise_max: np.ndarray, signal_max: float, delta95: float,
                      threshold: float, outcome: str, target: Path) -> None:
    # The noise scale (~5e-6) and the specification scale (0.030) differ by
    # roughly four orders of magnitude. A single linear axis hides the entire
    # histogram at the origin, so show a noise-scale zoom beside the full
    # decision scale.
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    zoom = axes[0]
    zoom_max = max(float(np.max(noise_max) * 1.08), float(delta95 * 1.20))
    zoom.hist(noise_max, bins=28, range=(0.0, zoom_max), color="#8EB8D8", edgecolor="white", alpha=0.92)
    zoom.axvline(delta95, color="#C96C5B", lw=2.0, label=f"delta95 = {delta95:.3e}")
    zoom.set_xlim(0.0, zoom_max)
    zoom.set_xlabel(r"Noise-only max $\max |m_{noise}|$")
    zoom.set_ylabel("Count (500 simulations)")
    zoom.set_title("Noise distribution (zoomed)")
    zoom.text(0.03, 0.96, "All simulated noise maxima\nare visible on this scale.",
              transform=zoom.transAxes, va="top", fontsize=8.5,
              bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"})
    zoom.grid(axis="y", alpha=0.22)
    zoom.legend(frameon=False, fontsize=8)

    decision = axes[1]
    decision.axvspan(0.0, delta95, color="#C96C5B", alpha=0.18, label="noise allowance")
    decision.axvline(delta95, color="#C96C5B", lw=2.0, label=f"delta95 = {delta95:.3e}")
    decision.axvline(signal_max, color="#1A8A70", lw=2.0, label=f"map max|m| = {signal_max:.5f}")
    decision.axvline(threshold, color="#284F73", lw=1.4, ls="--", label=f"threshold = {threshold:.3f}")
    decision.set_xlim(0.0, threshold * 1.05)
    decision.set_ylim(0.0, 1.0)
    decision.set_yticks([])
    decision.set_xlabel("Full decision scale")
    decision.set_title("Pass/fail comparison")
    decision.text(
        0.03,
        0.08,
        f"a + delta95 = {signal_max + delta95:.5f}\n"
        f"threshold = {threshold:.3f}\n"
        f"{outcome.upper() if outcome == 'borderline' else 'clear ' + outcome.upper()}",
        transform=decision.transAxes,
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    decision.grid(axis="x", alpha=0.22)
    decision.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("G08 L3 noise-linked detection limit", fontsize=15)
    fig.savefig(target, dpi=240)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Reproduce G08 L3 from the bundled data")
    parser.add_argument("--data-dir", type=Path, default=script_dir / "data" / "regular_G08",
                        help="input folder (default: data/regular_G08 beside this script)")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs",
                        help="output folder (default: outputs beside this script)")
    args = parser.parse_args()

    for name in ("L3_meta.json", "L3_A.npy", "L3_d.csv"):
        path = args.data_dir / name
        if not path.is_file():
            parser.error(f"required L3 input missing: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.data_dir / "L3_meta.json").read_text(encoding="utf-8"))
    group = str(meta["group"])
    npix = int(meta["npix"])
    if npix != NPIX:
        raise ValueError(f"this implementation expects npix=10, got {npix}")
    A = np.load(args.data_dir / "L3_A.npy")
    d_reps = np.loadtxt(args.data_dir / "L3_d.csv", delimiter=",")
    if A.shape != (128, 100) or d_reps.shape != (3, 128):
        raise ValueError(f"unexpected shapes: A={A.shape}, d={d_reps.shape}")
    if not np.isfinite(A).all() or not np.isfinite(d_reps).all():
        raise ValueError("L3 inputs contain non-finite values")
    row_sum_max = float(np.max(np.abs(A.sum(axis=1))))
    if row_sum_max >= 1e-11:
        raise ValueError(f"row-sum gate failed: {row_sum_max}")
    dbar = d_reps.mean(axis=0)
    L = gradient_matrix(NPIX)
    P, L, eta0 = prepare_solver(A, L)
    eta_values = eta0 * ETA_FACTORS

    maps: list[np.ndarray] = []
    residuals: list[float] = []
    roughness: list[float] = []
    inverses: list[np.ndarray] = []
    for eta in eta_values:
        inverse = make_inverse(A, L, P, float(eta))
        m = reconstruct(dbar, inverse)
        inverses.append(inverse)
        maps.append(m)
        residuals.append(float(np.linalg.norm(A @ m - dbar)))
        roughness.append(float(np.linalg.norm(L @ m)))
    reference_index = int(np.flatnonzero(ETA_FACTORS == 1.0)[0])
    inverse = inverses[reference_index]
    mhat = maps[reference_index]
    features = extrema_features(mhat)

    resolution = resolution_test(A, inverse)
    sigma_channel = d_reps.std(axis=0, ddof=1)
    rng = np.random.default_rng(NOISE_SEED)
    noise_vectors = rng.normal(0.0, sigma_channel, size=(NOISE_REPLICATES, A.shape[0]))
    noise_maps = np.asarray([reconstruct(vector, inverse) for vector in noise_vectors])
    noise_max = np.max(np.abs(noise_maps), axis=1)
    delta95 = float(np.percentile(noise_max, 95.0))
    signal_max = float(np.max(np.abs(mhat)))
    upper = signal_max + delta95
    lower = signal_max - delta95
    if upper < SPEC_THRESHOLD:
        decision, confidence = "pass", "clear pass: max|m| + delta95 is below 0.030"
    elif lower > SPEC_THRESHOLD:
        decision, confidence = "fail", "clear fail: max|m| - delta95 is above 0.030"
    else:
        decision, confidence = "borderline", "borderline: uncertainty interval overlaps the 0.030 threshold"

    map_path = args.output_dir / f"{group}_L3_map_figure.png"
    method_path = args.output_dir / f"{group}_L3_method_figure.png"
    noise_path = args.output_dir / f"{group}_L3_noise_figure.png"
    save_map_figure(mhat, features, SPEC_THRESHOLD, map_path)
    save_method_figure(ETA_FACTORS, np.asarray(residuals), np.asarray(roughness), resolution,
                       delta95, SPEC_THRESHOLD, method_path)
    save_noise_figure(noise_max, signal_max, delta95, SPEC_THRESHOLD, decision, noise_path)

    diagnostics = {
        "group": group,
        "npix": NPIX,
        "A_shape": list(A.shape),
        "d_shape": list(d_reps.shape),
        "row_sum_max_abs": row_sum_max,
        "matrix_rank": int(np.linalg.matrix_rank(A)),
        "singular_values": np.linalg.svd(A, compute_uv=False).tolist(),
        "gradient_shape": list(L.shape),
        "eta0": float(eta0),
        "eta_factors": ETA_FACTORS.tolist(),
        "eta_values": eta_values.tolist(),
        "eta_sensitivity": [
            {"factor": float(factor), "eta": float(eta), "residual_norm": float(res), "roughness_norm": float(rough)}
            for factor, eta, res, rough in zip(ETA_FACTORS, eta_values, residuals, roughness)
        ],
        "reference_eta_factor": 1.0,
        "mean_map": mhat.tolist(),
        "map_mean": float(np.mean(mhat)),
        "map_max_abs": signal_max,
        "features": features,
        "resolution": resolution,
        "noise": {
            "seed": NOISE_SEED,
            "replicates": NOISE_REPLICATES,
            "channel_sigma_min": float(sigma_channel.min()),
            "channel_sigma_median": float(np.median(sigma_channel)),
            "channel_sigma_max": float(sigma_channel.max()),
            "delta95": delta95,
            "noise_max_median": float(np.median(noise_max)),
            "noise_max_p99": float(np.percentile(noise_max, 99)),
        },
        "specification": {
            "threshold": SPEC_THRESHOLD,
            "a_max_abs_m": signal_max,
            "a_plus_delta95": upper,
            "a_minus_delta95": lower,
            "decision": decision,
            "confidence": confidence,
        },
    }
    (args.output_dir / f"{group}_L3_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
