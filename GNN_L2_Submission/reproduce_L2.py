#!/usr/bin/env python3
"""Reproduce the QXU6028 G08 L2 zero-dimensional inverse analysis.

The implementation follows the 2026-27 Practical Methods Guide:
paired repeat sampling, separate channel normalisation, a 4,000-point energy
grid, a 400-point Fermi-level grid, nearest-curve-point fitting, the stated
15% hard-gap selection rule, and 200 paired bootstrap replicates.

Install: python -m pip install "numpy>=2.0" pandas matplotlib
Run from any directory: python reproduce_L2.py
By default, inputs are read beside this script from data/regular_G08 and
generated results are written beside this script to outputs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


KB_EV_K = 8.617333e-5
NA_CM2 = 3.816e15
E_MAX_EV = 2.5
N_ENERGY = 4_000
N_EF = 400
GAP_GRID_KT = np.arange(0.50, 8.00 + 0.001, 0.25)
# Exploratory signed-gap diagnostic. Negative E_g denotes band overlap and is
# deliberately kept separate from the assessed hard-gap library above.
SIGNED_GAP_GRID_KT = np.unique(
    np.round(
        np.concatenate(
            [
                np.arange(-8.0, -0.49, 0.50),
                np.array([-0.25, -0.20, -0.15, -0.125, -0.10, -0.075, -0.066, -0.05, -0.025]),
                np.array([0.0]),
                GAP_GRID_KT,
            ]
        ),
        6,
    )
)
BOOTSTRAP_REPLICATES = 200
BOOTSTRAP_SEED = 6028


def fermi(E: np.ndarray, EF: np.ndarray, temperature_K: float) -> np.ndarray:
    x = np.clip((E - EF) / (KB_EV_K * temperature_K), -60.0, 60.0)
    return 1.0 / (np.exp(x) + 1.0)


def density_of_states(E: np.ndarray, model: str, Eg_eV: float) -> np.ndarray:
    if model == "cone":
        return np.abs(E)
    if model == "signed_gap":
        # A signed gap is the sum of conduction and valence branches:
        # max(E-Eg/2, 0) + max(-E-Eg/2, 0).  For Eg<0 the branches
        # overlap and produce a finite constant DOS plateau around E=0.
        return np.maximum(E - Eg_eV / 2.0, 0.0) + np.maximum(-E - Eg_eV / 2.0, 0.0)
    half_gap = Eg_eV / 2.0
    return np.where(np.abs(E) > half_gap, np.abs(E) - half_gap, 0.0)


def model_curve(
    E: np.ndarray,
    EF_grid: np.ndarray,
    temperature_K: float,
    model: str,
    Eg_eV: float,
) -> np.ndarray:
    """Return independently normalised [R_H, R_s] model pairs."""
    ef = EF_grid[:, None]
    energy = E[None, :]
    f = fermi(energy, ef, temperature_K)
    g = density_of_states(E, model, Eg_eV)[None, :]
    n = NA_CM2 * np.trapezoid(np.where(energy >= ef, g * f, 0.0), E, axis=1)
    p = NA_CM2 * np.trapezoid(np.where(energy <= ef, g * (1.0 - f), 0.0), E, axis=1)
    total = p + n
    hall = (p - n) / total**2
    sheet = 1.0 / total
    hall_norm = hall / np.max(np.abs(hall))
    sheet_norm = sheet / np.max(sheet)
    return np.column_stack([hall_norm, sheet_norm])


def normalise_pairs(pairs: np.ndarray) -> tuple[np.ndarray, float, float]:
    hall_scale = float(np.max(np.abs(pairs[:, 0])))
    sheet_scale = float(np.max(pairs[:, 1]))
    if hall_scale <= 0 or sheet_scale <= 0:
        raise ValueError("normalisation scale must be positive")
    normalised = np.column_stack([pairs[:, 0] / hall_scale, pairs[:, 1] / sheet_scale])
    return normalised, hall_scale, sheet_scale


def nearest_curve_fit(data_xy: np.ndarray, curve_xy: np.ndarray) -> tuple[float, np.ndarray]:
    d2 = np.sum((data_xy[:, None, :] - curve_xy[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1)
    residual = float(np.sqrt(np.mean(np.min(d2, axis=1))))
    return residual, nearest


def fit_candidate_library(
    data_xy: np.ndarray,
    library: dict[str, dict[str, object]],
) -> dict[str, object]:
    cone_residual, cone_nearest = nearest_curve_fit(data_xy, library["cone"]["curve"])

    gap_fits: list[tuple[float, float, np.ndarray, str]] = []
    for key, candidate in library.items():
        if key == "cone":
            continue
        residual, nearest = nearest_curve_fit(data_xy, candidate["curve"])
        gap_fits.append((residual, float(candidate["Eg_kT"]), nearest, key))
    gap_residual, gap_kT, gap_nearest, gap_key = min(gap_fits, key=lambda item: item[0])

    hard_gap_selected = gap_residual < 0.85 * cone_residual and gap_kT >= 1.0
    if hard_gap_selected:
        return {
            "model": "hard_gap",
            "Eg_kT": gap_kT,
            "residual": gap_residual,
            "nearest": gap_nearest,
            "library_key": gap_key,
            "cone_residual": cone_residual,
            "best_gap_residual": gap_residual,
            "best_gap_kT": gap_kT,
        }
    return {
        "model": "cone",
        "Eg_kT": 0.0,
        "residual": cone_residual,
        "nearest": cone_nearest,
        "library_key": "cone",
        "cone_residual": cone_residual,
        "best_gap_residual": gap_residual,
        "best_gap_kT": gap_kT,
    }


def build_library(temperature_K: float) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    kBT = KB_EV_K * temperature_K
    E = np.linspace(-E_MAX_EV, E_MAX_EV, N_ENERGY)
    EF_grid = np.linspace(-8.0 * kBT, 9.0 * kBT, N_EF)
    library: dict[str, dict[str, object]] = {
        "cone": {
            "model": "cone",
            "Eg_kT": 0.0,
            "curve": model_curve(E, EF_grid, temperature_K, "cone", 0.0),
        }
    }
    for gap_kT in GAP_GRID_KT:
        key = f"gap_{gap_kT:.2f}"
        library[key] = {
            "model": "hard_gap",
            "Eg_kT": float(gap_kT),
            "curve": model_curve(E, EF_grid, temperature_K, "hard_gap", gap_kT * kBT),
        }
    return EF_grid, library


def load_paired_repeats(data_file: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = pd.read_csv(data_file)
    required = {"measurement_index", "repeat", "R_H_raw", "R_xx_raw"}
    if set(data.columns) != required:
        raise ValueError(f"unexpected columns: {list(data.columns)}")
    indices = np.sort(data["measurement_index"].unique())
    repeats: list[np.ndarray] = []
    for index in indices:
        block = data.loc[data["measurement_index"] == index].sort_values("repeat")
        if len(block) != 3 or block["repeat"].nunique() != 3:
            raise ValueError(f"measurement_index {index} does not contain three distinct repeats")
        repeats.append(block[["R_H_raw", "R_xx_raw"]].to_numpy(dtype=float))
    return data, indices, np.stack(repeats, axis=0)


def save_gate_figure(
    data: pd.DataFrame,
    indices: np.ndarray,
    means: np.ndarray,
    sign_bracket: tuple[int, int],
    target: Path,
) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, constrained_layout=True)
    colours = ["#315D8A", "#2E8B78", "#C1732B"]
    for repeat, colour in zip(sorted(data["repeat"].unique()), colours):
        block = data.loc[data["repeat"] == repeat].sort_values("measurement_index")
        axes[0].plot(block["measurement_index"], block["R_H_raw"] * 1e12, "o-", ms=3, lw=0.8,
                     alpha=0.72, color=colour, label=f"repeat {repeat}")
        axes[1].plot(block["measurement_index"], block["R_xx_raw"] * 1e12, "o-", ms=3, lw=0.8,
                     alpha=0.72, color=colour, label=f"repeat {repeat}")
    axes[0].plot(indices, means[:, 0] * 1e12, "k-", lw=2.0, label="repeat mean")
    axes[1].plot(indices, means[:, 1] * 1e12, "k-", lw=2.0, label="repeat mean")
    for axis in axes:
        axis.axvspan(sign_bracket[0], sign_bracket[1], color="#E8B04B", alpha=0.24)
        axis.grid(alpha=0.24)
    axes[0].axhline(0, color="0.25", lw=0.9)
    axes[0].set_ylabel(r"$R_H$ raw ($10^{-12}$, arb. units)")
    axes[1].set_ylabel(r"$R_{xx}$ raw ($10^{-12}$, arb. units)")
    axes[1].set_xlabel("Measurement index")
    axes[0].legend(ncol=4, frameon=False, fontsize=8)
    axes[0].set_title(
        f"G08 L2 gate: Hall sign change is bracketed by indices {sign_bracket[0]} and {sign_bracket[1]}"
    )
    fig.savefig(target, dpi=240)
    plt.close(fig)


def save_model_figure(
    data_xy: np.ndarray,
    indices: np.ndarray,
    selected_curve: np.ndarray,
    central: dict[str, object],
    sign_bracket: tuple[int, int],
    target: Path,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8.4, 6.8), constrained_layout=True)
    ax.plot(selected_curve[:, 0], selected_curve[:, 1], color="#284F73", lw=2.3,
            label=f"selected {str(central['model']).replace('_', '-')} curve")
    nearest_points = selected_curve[np.asarray(central["nearest"], dtype=int)]
    for row, (data_point, nearest_point) in enumerate(zip(data_xy, nearest_points)):
        ax.plot(
            [data_point[0], nearest_point[0]],
            [data_point[1], nearest_point[1]],
            color="#C96C5B",
            lw=0.55,
            alpha=0.42,
            zorder=1,
            label="data-to-curve residual" if row == 0 else None,
        )
    ax.plot(data_xy[:, 0], data_xy[:, 1], color="0.55", lw=0.8, alpha=0.75, zorder=2)
    scatter = ax.scatter(data_xy[:, 0], data_xy[:, 1], c=indices, cmap="viridis", s=36,
                         edgecolor="white", linewidth=0.45, zorder=3, label="repeat means")
    for index in sign_bracket:
        row = int(np.flatnonzero(indices == index)[0])
        ax.scatter(data_xy[row, 0], data_xy[row, 1], s=105, facecolors="none",
                   edgecolors="#D48120", linewidths=1.8, zorder=4)
        ax.annotate(str(index), data_xy[row], xytext=(6, 5), textcoords="offset points", fontsize=8)
    improvement = 100.0 * (1.0 - central["best_gap_residual"] / central["cone_residual"])
    note = (
        f"cone RMS = {central['cone_residual']:.4f}\n"
        f"best gap RMS = {central['best_gap_residual']:.4f}\n"
        f"best gap = {central['best_gap_kT']:.2f} $k_BT$\n"
        f"residual improvement = {improvement:.1f}%"
    )
    ax.text(0.025, 0.025, note, transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"})
    fig.colorbar(scatter, ax=ax, label="Measurement index")
    ax.set_xlabel(r"Normalised Hall coefficient, $\widetilde{R}_H$")
    ax.set_ylabel(r"Normalised sheet resistance, $\widetilde{R}_s$")
    ax.set_title("G08 L2: measured pairs and selected model curve")
    ax.grid(alpha=0.23)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(target, dpi=240)
    plt.close(fig)


def scan_signed_gap(
    data_xy: np.ndarray,
    temperature_K: float,
    EF_grid: np.ndarray,
    library: dict[str, dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Scan signed Eg, adding an exploratory overlap branch for Eg<0."""
    E = np.linspace(-E_MAX_EV, E_MAX_EV, N_ENERGY)
    sse: list[float] = []
    for gap_kT in SIGNED_GAP_GRID_KT:
        if abs(gap_kT) < 1e-12:
            curve = library["cone"]["curve"]
        elif gap_kT >= 0.50 - 1e-12 and abs((gap_kT / 0.25) - round(gap_kT / 0.25)) < 1e-9:
            curve = library[f"gap_{gap_kT:.2f}"]["curve"]
        else:
            curve = model_curve(E, EF_grid, temperature_K, "signed_gap", float(gap_kT) * KB_EV_K * temperature_K)
        residual, _ = nearest_curve_fit(data_xy, curve)
        sse.append(float(len(data_xy) * residual**2))
    return SIGNED_GAP_GRID_KT.copy(), np.asarray(sse)


def save_sse_vs_gap_figure(
    gap_kT: np.ndarray,
    sse: np.ndarray,
    temperature_K: float,
    cone_sse: float,
    best_positive_gap: float,
    best_positive_sse: float,
    target: Path,
) -> dict[str, object]:
    best_index = int(np.argmin(sse))
    best_gap = float(gap_kT[best_index])
    best_sse = float(sse[best_index])
    if plt is None:
        return {
            "best_signed_gap_kT": best_gap,
            "best_signed_gap_sse": best_sse,
            "cone_sse": float(cone_sse),
            "best_formal_positive_gap_kT": float(best_positive_gap),
            "best_formal_positive_gap_sse": float(best_positive_sse),
            "signed_scan_improvement_fraction": float(1.0 - best_sse / cone_sse),
            "signed_gap_grid_count": int(len(gap_kT)),
        }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.65, 1.0]})
    ax = axes[0]
    ax.plot(gap_kT, sse, color="#284F73", lw=1.8, marker="o", ms=2.4,
            label=r"nearest-point SSE $\sum_i d_i^2$")
    ax.axvspan(gap_kT.min(), 0.0, color="#E8B04B", alpha=0.14,
               label=r"exploratory overlap: $E_g<0$")
    ax.axvspan(0.50, gap_kT.max(), color="#79A9D1", alpha=0.10,
               label=r"assessed hard-gap grid: $E_g\geq0.5k_BT$")
    ax.scatter([0], [cone_sse], s=100, color="#1A8A70", zorder=4, label="cone, $E_g=0$")
    ax.scatter([best_gap], [best_sse], s=84, color="#6B4E9B", zorder=4,
               label=f"signed-scan minimum ({best_gap:.3f} $k_BT$)")
    ax.scatter([best_positive_gap], [best_positive_sse], s=72, color="#C96C5B", zorder=4,
               label=f"best formal hard gap ({best_positive_gap:.2f} $k_BT$)")
    ax.set_xlabel(r"Signed gap parameter $E_g/k_BT$")
    ax.set_ylabel(r"Residual sum of squares, $\mathrm{SSE}=\sum_i d_i^2$")
    ax.set_title("L2 residual sum of squares versus signed gap")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.text(
        0.02,
        0.03,
        f"cone SSE = {cone_sse:.6f}\n"
        f"signed minimum = {best_sse:.6f}\n"
        f"exploratory improvement = {100*(1-best_sse/cone_sse):.2f}%",
        transform=ax.transAxes,
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    zoom = axes[1]
    mask = (gap_kT >= -0.25) & (gap_kT <= 0.25)
    zoom.plot(gap_kT[mask], sse[mask], color="#284F73", lw=1.9, marker="o", ms=3.0)
    zoom.axvline(0.0, color="#1A8A70", lw=1.0, ls="--")
    zoom.scatter([best_gap], [best_sse], s=85, color="#6B4E9B", zorder=4)
    zoom.scatter([0], [cone_sse], s=70, color="#1A8A70", zorder=4)
    zoom.set_xlabel(r"Zoomed $E_g/k_BT$")
    zoom.set_ylabel("SSE")
    zoom.set_title("Near-zero overlap/cone region")
    zoom.grid(alpha=0.22)
    fig.suptitle(
        f"G08 L2: signed-gap sensitivity at {temperature_K:.0f} K "
        "(negative values are an exploratory overlap extension)",
        fontsize=14,
    )
    fig.savefig(target, dpi=240)
    plt.close(fig)
    return {
        "best_signed_gap_kT": best_gap,
        "best_signed_gap_sse": best_sse,
        "cone_sse": float(cone_sse),
        "best_formal_positive_gap_kT": float(best_positive_gap),
        "best_formal_positive_gap_sse": float(best_positive_sse),
        "signed_scan_improvement_fraction": float(1.0 - best_sse / cone_sse),
        "signed_gap_grid_count": int(len(gap_kT)),
    }


def save_trajectory_figure(
    indices: np.ndarray,
    central_ef: np.ndarray,
    ef_median: np.ndarray,
    ef_p16: np.ndarray,
    ef_p84: np.ndarray,
    temperature_K: float,
    hard_gap_fraction: float,
    sign_bracket: tuple[int, int],
    target: Path,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    ax.fill_between(indices, ef_p16, ef_p84, color="#79A9D1", alpha=0.34,
                    label="bootstrap 16th-84th percentile")
    ax.plot(indices, ef_median, color="#356D9B", lw=1.5, ls="--", label="bootstrap median")
    ax.plot(indices, central_ef, color="#162F46", lw=2.0, marker="o", ms=3.2,
            label="fit to repeat means")
    ax.axhline(0, color="0.25", lw=0.9)
    ax.axvspan(sign_bracket[0], sign_bracket[1], color="#E8B04B", alpha=0.22,
               label="Hall sign-change bracket")
    ax.set_xlabel("Measurement index")
    ax.set_ylabel(r"Assigned Fermi level $E_F$ (eV)")
    ax.set_title(f"G08 L2 Fermi-level trajectory (hard-gap selection in {hard_gap_fraction:.1%} of bootstraps)")
    ax.grid(alpha=0.23)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    kBT = KB_EV_K * temperature_K
    secondary = ax.secondary_yaxis("right", functions=(lambda value: value / kBT, lambda value: value * kBT))
    secondary.set_ylabel(r"$E_F/k_BT$")
    fig.savefig(target, dpi=240)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Reproduce G08 L2 from the bundled data")
    parser.add_argument("--data-dir", type=Path, default=script_dir / "data" / "regular_G08",
                        help="input folder (default: data/regular_G08 beside this script)")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs",
                        help="output folder (default: outputs beside this script)")
    args = parser.parse_args()

    for name in ("L2_meta.json", "L2_data.csv", "results_template.json"):
        path = args.data_dir / name
        if not path.is_file():
            parser.error(f"required L2 input missing: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.data_dir / "L2_meta.json").read_text(encoding="utf-8"))
    temperature_K = float(meta["T_K"])
    group = str(meta["group"])
    kBT = KB_EV_K * temperature_K
    data, indices, repeats = load_paired_repeats(args.data_dir / "L2_data.csv")
    means = repeats.mean(axis=1)
    stds = repeats.std(axis=1, ddof=1)
    data_xy, hall_scale, sheet_scale = normalise_pairs(means)

    sign_changes = np.flatnonzero(np.signbit(means[:-1, 0]) != np.signbit(means[1:, 0]))
    if len(sign_changes) != 1:
        raise ValueError(f"expected one Hall sign-change bracket, found {len(sign_changes)}")
    sign_bracket = (int(indices[sign_changes[0]]), int(indices[sign_changes[0] + 1]))

    EF_grid, library = build_library(temperature_K)
    central = fit_candidate_library(data_xy, library)
    central_ef = EF_grid[central["nearest"]]

    selected_curve = library[str(central["library_key"])]["curve"]
    selected_d2 = np.sum((data_xy - selected_curve[central["nearest"]]) ** 2, axis=1)
    neighbour_indices_left = np.maximum(np.asarray(central["nearest"]) - 1, 0)
    neighbour_indices_right = np.minimum(np.asarray(central["nearest"]) + 1, len(EF_grid) - 1)
    neighbour_d2_left = np.sum((data_xy - selected_curve[neighbour_indices_left]) ** 2, axis=1)
    neighbour_d2_right = np.sum((data_xy - selected_curve[neighbour_indices_right]) ** 2, axis=1)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_gap = np.empty(BOOTSTRAP_REPLICATES)
    bootstrap_ef = np.empty((BOOTSTRAP_REPLICATES, len(indices)))
    bootstrap_model: list[str] = []
    for b in range(BOOTSTRAP_REPLICATES):
        choices = rng.integers(0, repeats.shape[1], size=len(indices))
        sampled_pairs = repeats[np.arange(len(indices)), choices, :]
        sampled_xy, _, _ = normalise_pairs(sampled_pairs)
        fit = fit_candidate_library(sampled_xy, library)
        bootstrap_model.append(str(fit["model"]))
        bootstrap_gap[b] = float(fit["Eg_kT"])
        bootstrap_ef[b] = EF_grid[fit["nearest"]]

    gap_p16, gap_median, gap_p84 = np.percentile(bootstrap_gap, [16, 50, 84])
    gap_half_width = 0.5 * (gap_p84 - gap_p16)
    ef_p16, ef_median, ef_p84 = np.percentile(bootstrap_ef, [16, 50, 84], axis=0)
    hard_gap_fraction = float(np.mean(np.asarray(bootstrap_model) == "hard_gap"))
    improvement_fraction = 1.0 - float(central["best_gap_residual"]) / float(central["cone_residual"])

    processed = pd.DataFrame(
        {
            "measurement_index": indices,
            "R_H_mean": means[:, 0],
            "R_H_std": stds[:, 0],
            "R_xx_mean": means[:, 1],
            "R_xx_std": stds[:, 1],
            "R_H_normalised": data_xy[:, 0],
            "R_xx_normalised": data_xy[:, 1],
            "EF_central_eV": central_ef,
            "EF_bootstrap_median_eV": ef_median,
            "EF_bootstrap_p16_eV": ef_p16,
            "EF_bootstrap_p84_eV": ef_p84,
            "fit_residual_norm": np.sqrt(selected_d2),
            "fit_residual_norm_EF_minus_one_step": np.sqrt(neighbour_d2_left),
            "fit_residual_norm_EF_plus_one_step": np.sqrt(neighbour_d2_right),
            "fit_delta_d2_minus_one_step": neighbour_d2_left - selected_d2,
            "fit_delta_d2_plus_one_step": neighbour_d2_right - selected_d2,
        }
    )
    processed.to_csv(args.output_dir / f"{group}_L2_processed_results.csv", index=False)

    cone_sse = float(len(indices) * central["cone_residual"] ** 2)
    best_gap_sse = float(len(indices) * central["best_gap_residual"] ** 2)
    signed_gap_kT, signed_gap_sse = scan_signed_gap(
        data_xy, temperature_K, EF_grid, library
    )
    signed_gap_diagnostic = save_sse_vs_gap_figure(
        signed_gap_kT,
        signed_gap_sse,
        temperature_K,
        cone_sse,
        float(central["best_gap_kT"]),
        best_gap_sse,
        args.output_dir / f"{group}_L2_SSE_vs_signed_gap.png",
    )

    diagnostics = {
        "group": group,
        "temperature_K": temperature_K,
        "kBT_eV": kBT,
        "measurement_count": int(len(indices)),
        "repeats_per_measurement": 3,
        "hall_sign_change_bracket": list(sign_bracket),
        "normalisation": {
            "R_H_max_abs": hall_scale,
            "R_xx_max": sheet_scale,
        },
        "grids": {
            "energy_points": N_ENERGY,
            "energy_range_eV": [-E_MAX_EV, E_MAX_EV],
            "EF_points": N_EF,
            "EF_range_kT": [-8.0, 9.0],
            "EF_step_eV": float(EF_grid[1] - EF_grid[0]),
            "gap_range_kT": [0.50, 8.00],
            "gap_step_kT": 0.25,
        },
        "central_fit": {
            "selected_model": central["model"],
            "selected_Eg_kT": central["Eg_kT"],
            "selected_Eg_eV": float(central["Eg_kT"] * kBT),
            "selected_residual": central["residual"],
            "cone_residual": central["cone_residual"],
            "best_gap_residual": central["best_gap_residual"],
            "best_gap_kT": central["best_gap_kT"],
            "best_gap_improvement_fraction": improvement_fraction,
            "selection_threshold_fraction": 0.15,
        },
        "signed_gap_exploratory": signed_gap_diagnostic,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "hard_gap_selection_fraction": hard_gap_fraction,
            "Eg_kT_median": float(gap_median),
            "Eg_kT_p16": float(gap_p16),
            "Eg_kT_p84": float(gap_p84),
            "Eg_uncertainty_half_width_kT": float(gap_half_width),
        },
    }
    (args.output_dir / f"{group}_L2_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    results = json.loads((args.data_dir / "results_template.json").read_text(encoding="utf-8"))
    results["L2"] = {
        "model": central["model"],
        "Eg_kT": float(central["Eg_kT"]),
        "Eg_uncertainty_kT": float(gap_half_width),
        "EF_trajectory_eV": central_ef.tolist(),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    save_gate_figure(
        data, indices, means, sign_bracket, args.output_dir / f"{group}_L2_gate_figure.png"
    )
    save_model_figure(
        data_xy,
        indices,
        selected_curve,
        central,
        sign_bracket,
        args.output_dir / f"{group}_L2_model_figure.png",
    )
    save_trajectory_figure(
        indices,
        central_ef,
        ef_median,
        ef_p16,
        ef_p84,
        temperature_K,
        hard_gap_fraction,
        sign_bracket,
        args.output_dir / f"{group}_L2_trajectory_figure.png",
    )

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
