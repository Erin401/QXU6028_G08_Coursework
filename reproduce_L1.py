#!/usr/bin/env python
"""
QXU6028 Polymer Devices - Virtual Lab, Group G08
L1 one-command reproduction.

Usage
-----
    python reproduce_L1.py --data "<path to regular_G08>" --out outputs

Produces, with no manual edits:
    outputs/results.json                 schema QXU6028-results-v1, L1 filled
    outputs/L1_counts_and_correction.png  two-count model + correction scan
    outputs/L1_validation.png             model-vs-data and closed-form checks
    outputs/L1_checks.log                 every gate and its verdict
    outputs/L1_summary.json               full precision working numbers

Nothing in this script is hand-entered: every number in results.json and every
point in the figures is recomputed from the supplied CSV + JSON on each run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# keep matplotlib's cache inside the project instead of the user profile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"))

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l1_model as m  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1f1f1e"
INK2 = "#5f5e5a"
BLUE = "#185FA5"
TEAL = "#0F6E56"
AMBER = "#854F0B"
RED = "#A32D2D"
GREY = "#888780"


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def load_l1(data_dir: str):
    csv_path = os.path.join(data_dir, "L1_data.csv")
    par_path = os.path.join(data_dir, "L1_params.json")

    raw = np.genfromtxt(csv_path, delimiter=",", names=True)
    EF = np.asarray(raw["EF_eV"], dtype=float)
    n_band_data = np.asarray(raw["n_band_cm2"], dtype=float)
    rho_data = np.asarray(raw["rho_ohm_sq"], dtype=float)

    with open(par_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    if not np.allclose(rho_data, rho_data[0], rtol=1e-12):
        raise ValueError("rho_ohm_sq is not constant across the sweep")
    return EF, n_band_data, rho_data, params


def crossing_energy(target: float, T: float, c: float, hi: float = 0.40) -> float:
    """Energy at which C(E_F) first reaches `target`, by bisection."""
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if m.correction_factor(mid, T, c) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding the G08 pack")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--commit", default="UNCOMMITTED", help="pipeline git commit")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    EF, n_band_data, rho_data, params = load_l1(args.data)
    c = float(params["c"])
    T = float(params["T"])
    EF_op = float(params["EF_op_eV"])
    rho_op = float(params.get("rho_ohm_sq", rho_data[0]))

    log: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        log.append(line)

    emit("=" * 74)
    emit("QXU6028 L1  |  group G08  |  forward model + mobility correction")
    emit("=" * 74)
    emit(f"c                 = {c!r}")
    emit(f"T                 = {T} K   (k_B T = {m.K_B_EV_PER_K * T:.6f} eV)")
    emit(f"E_F operating pt  = {EF_op!r} eV")
    emit(f"rho               = {rho_op!r} ohm/sq")
    emit(f"sweep             = {EF.size} rows, {EF[0]} -> {EF[-1]} eV")
    emit("")

    # -- Gate 0: model must reproduce the supplied n_band column -------------
    n_band_model = np.array([m.n_band(e, T, c) for e in EF])
    rel_dev = (n_band_model - n_band_data) / n_band_data
    max_dev = float(np.max(np.abs(rel_dev)))
    emit("[GATE 0] model reproduces the supplied band-edge count")
    emit(f"         max |n_band_model - n_band_data| / n_band_data = {max_dev:.3e}")
    emit(f"         {'PASS' if max_dev < 1e-9 else 'FAIL'}")
    emit("")

    # -- Gate 1: handbook ratio table + monotonicity -------------------------
    gates = m.run_gates(T, c)
    emit("[GATE 1] handbook verified-ratio table and monotonicity")
    emit(f"         {'E_F (eV)':>16} {'expected':>10} {'computed':>10} {'|err|':>10}  verdict")
    for g in gates:
        exp = g["expected"]
        cmp_ = g["computed"]
        if isinstance(exp, str):
            emit(f"         {str(g['EF_eV']):>16} {exp:>10} {str(bool(cmp_)):>10} {'--':>10}  {'PASS' if g['pass'] else 'FAIL'}")
        else:
            emit(f"         {g['EF_eV']:>16.3f} {exp:>10.2f} {cmp_:>10.3f} {g['abs_err']:>10.4f}  {'PASS' if g['pass'] else 'FAIL'}")
    emit("")

    # -- Gate 2: closed form vs numerical integral --------------------------
    cf = m.closed_form_agreement(T, c)
    conv = m.closed_form_convergence(T, c, EF_op)
    emit("[GATE 2] closed-form Fermi-referenced count vs numerical integral")
    emit("         the closed form is exact for an infinite upper limit, so the")
    emit("         residual must be pure trapezoid O(h^2) discretisation error.")
    emit(f"         max rel diff on handbook grid (20001 pts) = {cf['max_rel_diff']:.3e}"
         f"   (tol {cf['tol']:.0e})  {'PASS' if cf['pass'] else 'FAIL'}")
    emit("         grid refinement at the operating point:")
    for n, e in zip(conv["n_points"], conv["rel_err"]):
        emit(f"           n_points = {n:>7d}   rel err = {e:.3e}")
    emit(f"         observed convergence order (first decade) = "
         f"{conv['observed_order_first_decade']:.3f}  (expect 2 for O(h^2))"
         f"  {'PASS' if conv['pass'] else 'FAIL'}")
    emit("")

    # -- Gate 3: correction factor derived from the CSV column ---------------
    C_from_data = n_band_data / np.array([m.n_fermi(e, T, c) for e in EF])
    C_from_model = n_band_model / np.array([m.n_fermi(e, T, c) for e in EF])
    emit("[GATE 3] CSV-derived correction factor vs model-derived correction factor")
    emit(f"         max relative difference = {np.max(np.abs(C_from_data - C_from_model) / C_from_model):.3e}")
    emit(f"         {'PASS' if np.max(np.abs(C_from_data - C_from_model) / C_from_model) < 1e-9 else 'FAIL'}")
    emit("")

    # -- Working point -------------------------------------------------------
    C_op = m.correction_factor(EF_op, T, c)
    n_band_op = m.n_band(EF_op, T, c)
    n_fermi_op = m.n_fermi(EF_op, T, c)
    n_fermi_op_closed = m.n_fermi_closed_form(EF_op, T, c)
    mu_band = m.mobility_from_density(n_band_op, rho_op)
    mu_corrected = m.mobility_from_density(n_fermi_op, rho_op)

    # what you would get by lazily snapping to the nearest CSV row
    i_near = int(np.argmin(np.abs(EF - EF_op)))
    C_near = C_from_data[i_near]

    emit("[WORKING POINT] evaluated at the EXACT E_F from L1_params.json")
    emit(f"         n_band  (datasheet count)  = {n_band_op:.6e} cm^-2")
    emit(f"         n_Fermi (corrected count)  = {n_fermi_op:.6e} cm^-2")
    emit(f"         closed form check          = {n_fermi_op_closed:.6e} cm^-2"
         f"  (rel diff {abs(n_fermi_op - n_fermi_op_closed) / n_fermi_op:.2e})")
    emit(f"         correction factor C        = {C_op:.6f}")
    emit(f"         mobility (datasheet basis) = {mu_band:.2f} cm^2/Vs")
    emit(f"         mobility (corrected)       = {mu_corrected:.2f} cm^2/Vs")
    emit(f"         ratio mu_corr / mu_band    = {mu_corrected / mu_band:.6f}  (= C, as required)")
    emit("")
    emit(f"         nearest-CSV-row shortcut: row {i_near} at E_F = {EF[i_near]:.2f} eV"
         f" gives C = {C_near:.4f}")
    emit(f"         -> that would misreport C by {(C_near - C_op) / C_op * 100:+.3f} %")
    emit("")

    EF_c2 = crossing_energy(2.0, T, c)
    EF_c4 = crossing_energy(4.0, T, c)
    emit("[CROSSINGS] correction factor reaches")
    emit(f"         C = 2  at E_F = {EF_c2:.5f} eV   ({EF_c2 / (m.K_B_EV_PER_K * T):.3f} k_B T)")
    emit(f"         C = 4  at E_F = {EF_c4:.5f} eV   ({EF_c4 / (m.K_B_EV_PER_K * T):.3f} k_B T)")
    emit("")

    # ------------------------------------------------------------------ figure 1
    EF_scan = np.linspace(0.0, 0.26, 261)
    C_scan = np.array([m.correction_factor(e, T, c) for e in EF_scan])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=200)

    ax1.plot(EF_scan, np.array([m.n_band(e, T, c) for e in EF_scan]), color=BLUE,
             lw=1.8, label=r"$n_{\rm band}$  (band edge, from $E=0$)")
    ax1.plot(EF_scan, np.array([m.n_fermi(e, T, c) for e in EF_scan]), color=TEAL,
             lw=1.8, ls="--", label=r"$n_{\rm Fermi}$  (Fermi referenced, from $E_F$)")
    ax1.errorbar(EF, n_band_data / 1.0, fmt="o", ms=4, mfc="none", mec=BLUE,
                 mew=1.1, label="supplied data (L1_data.csv)")
    ax1.plot([EF_op], [n_band_op], marker="*", ms=13, color=RED, zorder=5,
             label=rf"operating point $E_F$ = {EF_op:.4f} eV")
    ax1.annotate("", xy=(EF_op, n_fermi_op), xytext=(EF_op, n_band_op),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
    ax1.text(EF_op + 0.012, np.sqrt(n_band_op * n_fermi_op),
             f"C = {C_op:.2f}\n(band-edge count\noverestimates by this factor)",
             color=RED, fontsize=8.5, va="center")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Fermi energy $E_F$  (eV)")
    ax1.set_ylabel(r"carrier count  (cm$^{-2}$)")
    ax1.set_title("a.  The two counts differ only in the lower limit", fontsize=10, color=INK)
    ax1.legend(fontsize=7.6, frameon=False, loc="lower right")
    ax1.tick_params(labelsize=8.5, colors=INK2)
    ax1.grid(alpha=0.15, lw=0.6)

    ax2.plot(EF_scan, C_scan, color=INK, lw=1.8)
    ax2.plot(EF, C_from_data, "o", ms=4, mfc="none", mec=BLUE, mew=1.1,
             label="from supplied $n_{\\rm band}$ column")
    for e, exp in m.GATE_TABLE[1:]:
        ax2.plot([e], [exp], "s", ms=7, mfc="none", mec=AMBER, mew=1.4)
        ax2.annotate(f"{exp:.2f}", (e, exp), textcoords="offset points",
                     xytext=(6, -10), fontsize=8, color=AMBER)
    ax2.axhline(1.0, color=GREY, lw=0.9, ls=":")
    for xc, lab in ((EF_c2, "C = 2"), (EF_c4, "C = 4")):
        ax2.axvline(xc, color=GREY, lw=0.8, ls=":")
        ax2.text(xc + 0.004, 1.15, f"{lab}\n@{xc:.3f} eV", fontsize=7.8, color=INK2)
    ax2.axvline(EF_op, color=RED, lw=1.1, ls="--")
    ax2.plot([EF_op], [C_op], marker="*", ms=13, color=RED, zorder=5)
    ax2.text(0.004, C_op + 0.35, f"working point C = {C_op:.3f}", fontsize=8.5, color=RED)
    ax2.set_xlabel(r"Fermi energy $E_F$  (eV)")
    ax2.set_ylabel(r"correction factor  $C = n_{\rm band}/n_{\rm Fermi}$")
    ax2.set_title("b.  Over-reporting factor rises monotonically with doping", fontsize=10, color=INK)
    ax2.set_ylim(0.8, max(C_scan.max(), 5.6))
    ax2.legend(fontsize=7.6, frameon=False, loc="upper left")
    ax2.tick_params(labelsize=8.5, colors=INK2)
    ax2.grid(alpha=0.15, lw=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "L1_counts_and_correction.png"), facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------------ figure 2
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(11.4, 3.9), dpi=200)

    bx1.axhline(0, color=GREY, lw=0.8, ls=":")
    bx1.plot(EF, rel_dev, "o-", color=BLUE, ms=4, lw=1.1)
    bx1.set_xlabel(r"Fermi energy $E_F$  (eV)")
    bx1.set_ylabel(r"$(n_{\rm model}-n_{\rm data})/n_{\rm data}$")
    bx1.set_ylim(-1e-11, 1e-11)
    bx1.set_title(f"a.  Reproduction of L1_data.csv\nmax relative deviation = {max_dev:.1e}",
                  fontsize=9.5, color=INK)
    bx1.tick_params(labelsize=8.5, colors=INK2)
    bx1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    bx1.grid(alpha=0.15, lw=0.6)

    bx2.loglog(np.array(conv["n_points"]), np.array(conv["rel_err"]),
               "o-", color=TEAL, ms=5, lw=1.4, label=r"observed $|n_{\rm num}-n_{\rm exact}|/n_{\rm exact}$")
    ref = np.array(conv["n_points"], dtype=float)
    anchor = conv["rel_err"][0] * (conv["n_points"][0] / ref) ** 2
    bx2.loglog(ref, anchor, "--", color=AMBER, lw=1.1, label=r"$O(h^2)$ reference")
    bx2.set_xlabel("integration points")
    bx2.set_ylabel("relative difference")
    bx2.set_title(f"b.  Residual is $O(h^2)$ discretisation, not a formula error\n"
                  f"observed order = {conv['observed_order_first_decade']:.2f}",
                  fontsize=9.5, color=INK)
    bx2.legend(fontsize=7.5, frameon=False, loc="lower left")
    bx2.tick_params(labelsize=8.5, colors=INK2)
    bx2.grid(alpha=0.15, lw=0.6, which="both")

    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out, "L1_validation.png"), facecolor="white")
    plt.close(fig2)

    # ------------------------------------------------------------------ outputs
    with open(os.path.join(args.data, "results_template.json"), "r", encoding="utf-8") as fh:
        results = json.load(fh)
    results["pipeline_commit"] = args.commit
    results["L1"] = {
        "correction_factor": C_op,
        "n_fermi_op_cm2": n_fermi_op,
        "mobility_corrected_cm2_Vs": mu_corrected,
    }
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    summary = {
        "group": "G08",
        "dataset_set": params.get("dataset_set", "regular"),
        "c": c,
        "T_K": T,
        "kB_T_eV": m.K_B_EV_PER_K * T,
        "EF_op_eV": EF_op,
        "rho_ohm_sq": rho_op,
        "constants": {
            "K_B_eV_per_K": m.K_B_EV_PER_K,
            "N_A_cm2": m.N_A_CM2,
            "e_C": m.E_CHARGE_C,
            "E_max_eV": m.E_MAX_EV,
            "n_points": m.N_POINTS_L1,
        },
        "working_point": {
            "n_band_op_cm2": n_band_op,
            "n_fermi_op_cm2": n_fermi_op,
            "n_fermi_op_closed_form_cm2": n_fermi_op_closed,
            "correction_factor": C_op,
            "mobility_band_basis_cm2_Vs": mu_band,
            "mobility_corrected_cm2_Vs": mu_corrected,
        },
        "snap_to_nearest_csv_row_would_give": {
            "row_index": i_near,
            "EF_eV": float(EF[i_near]),
            "C": float(C_near),
            "relative_error_percent": float((C_near - C_op) / C_op * 100.0),
        },
        "crossings": {"C_eq_2_at_eV": EF_c2, "C_eq_4_at_eV": EF_c4},
        "gates": {
            "reproduction_max_rel_dev": max_dev,
            "reproduction_pass": bool(max_dev < 1e-9),
            "ratio_table": [
                {"EF_eV": g["EF_eV"], "expected": g["expected"],
                 "computed": None if isinstance(g["computed"], (bool, str)) else float(g["computed"]),
                 "pass": bool(g["pass"])}
                for g in gates
            ],
            "closed_form_max_rel_diff": cf["max_rel_diff"],
            "closed_form_tol": cf["tol"],
            "closed_form_pass": bool(cf["pass"]),
            "closed_form_convergence": {
                "n_points": conv["n_points"],
                "rel_err": conv["rel_err"],
                "observed_order": conv["observed_order_first_decade"],
                "pass": bool(conv["pass"]),
            },
            "csv_vs_model_C_pass": bool(
                np.max(np.abs(C_from_data - C_from_model) / C_from_model) < 1e-9),
            "mobility_ratio_equals_C_pass": bool(abs(mu_corrected / mu_band - C_op) < 1e-12),
        },
        "sweep": {
            "EF_eV": EF.tolist(),
            "n_band_csv_cm2": n_band_data.tolist(),
            "n_band_model_cm2": n_band_model.tolist(),
            "C_from_csv": C_from_data.tolist(),
        },
        "figures": ["L1_counts_and_correction.png", "L1_validation.png"],
        "pipeline_commit": args.commit,
    }
    with open(os.path.join(args.out, "L1_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    emit(f"[WRITTEN] {os.path.join(args.out, 'results.json')}")
    emit(f"[WRITTEN] {os.path.join(args.out, 'L1_summary.json')}")
    emit(f"[WRITTEN] {os.path.join(args.out, 'L1_counts_and_correction.png')}")
    emit(f"[WRITTEN] {os.path.join(args.out, 'L1_validation.png')}")
    emit("")

    all_pass = (
        max_dev < 1e-9
        and all(g["pass"] for g in gates)
        and cf["pass"]
        and conv["pass"]
        and abs(mu_corrected / mu_band - C_op) < 1e-12
    )
    emit("=" * 74)
    emit(f"L1 GATE SUMMARY: {'ALL CHECKS PASS' if all_pass else 'CHECK FAILED - DO NOT SUBMIT'}")
    emit("=" * 74)

    with open(os.path.join(args.out, "L1_checks.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
