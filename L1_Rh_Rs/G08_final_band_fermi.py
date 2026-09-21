"""Generate band-edge/Fermi-referenced carrier and transport figures.

Run:
    python G08_final_band_fermi.py --data path/to/regular_GROUP

The --data argument must point to a regular_GROUP folder containing
L1_params.json, or directly to an L1_params.json file.

Required packages:
    numpy, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KB_EV_K = 8.617333e-5
NA_CM2 = 3.816e15
E_CHARGE_C = 1.602176634e-19
E_MAX_EV = 2.5
N_INTEGRATION_POINTS = 20001

BLUE = "#176B87"
RED = "#C43D3D"


def fermi(E_eV: np.ndarray, EF_eV: float, T_K: float) -> np.ndarray:
    """Finite-temperature Fermi function."""
    exponent = np.clip((E_eV - EF_eV) / (KB_EV_K * T_K), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(exponent))


def dos(E_eV: np.ndarray, c: float) -> np.ndarray:
    """Symmetric ideal-cone density of states, g(E)=c|E|."""
    return c * np.abs(E_eV)


def integrate_density(E0: float, E1: float, EF_eV: float, T_K: float,
                      c: float, holes: bool = False) -> float:
    E = np.linspace(E0, E1, N_INTEGRATION_POINTS)
    occupation = 1.0 - fermi(E, EF_eV, T_K) if holes else fermi(E, EF_eV, T_K)
    return float(NA_CM2 * np.trapezoid(dos(E, c) * occupation, E))


def carrier_counts(EF_eV: float, T_K: float, c: float) -> tuple[float, float, float, float]:
    """Return n_band, n_Fermi, p_band and p_Fermi in cm^-2."""
    n_band = integrate_density(0.0, E_MAX_EV, EF_eV, T_K, c)
    n_fermi = integrate_density(EF_eV, E_MAX_EV, EF_eV, T_K, c)
    p_band = integrate_density(-E_MAX_EV, 0.0, EF_eV, T_K, c, holes=True)
    p_fermi = integrate_density(-E_MAX_EV, EF_eV, EF_eV, T_K, c, holes=True)
    return n_band, n_fermi, p_band, p_fermi


def hall_resistance(n_cm2: np.ndarray, p_cm2: np.ndarray) -> np.ndarray:
    """Two-carrier Hall resistance in ohm/T.

    Densities supplied in cm^-2 give cm^2/C; multiplying by 1e-4 converts
    the result to m^2/C, which is numerically equal to ohm/T.
    """
    return (p_cm2 - n_cm2) / (E_CHARGE_C * (p_cm2 + n_cm2) ** 2) * 1e-4


def sheet_resistance(n_cm2: np.ndarray, p_cm2: np.ndarray,
                     mobility_cm2_Vs: float) -> np.ndarray:
    """Two-carrier sheet resistance in ohm/square."""
    return 1.0 / (E_CHARGE_C * mobility_cm2_Vs * (p_cm2 + n_cm2))


def format_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.axvline(0.0, color="#777777", linewidth=0.9)
    ax.set_xlabel(r"Fermi energy $E_F$ (eV)")


def plot_density_comparison(EF: np.ndarray, band: np.ndarray, referenced: np.ndarray,
                            carrier: str, EF_op: float, group: str,
                            output: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.2, 6.1), constrained_layout=True)
    ax.plot(EF, band / 1e12, color=BLUE, linewidth=2.4,
            label=rf"${carrier}_{{band}}$: boundary at $E=0$")
    boundary = "lower limit" if carrier == "n" else "upper limit"
    ax.plot(EF, referenced / 1e12, color=RED, linewidth=2.4, linestyle="--",
            label=rf"${carrier}_F$: {boundary} $E=E_F$")
    ax.axvline(EF_op, color=RED, linewidth=1.2, alpha=0.9,
               label=rf"{group} $E_F^{{op}}={EF_op:.4f}$ eV")
    ax.set_xlim(EF.min(), EF.max())
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(rf"${carrier}$ density ($10^{{12}}$ cm$^{{-2}}$)")
    ax.set_title(rf"{group} ${carrier}$: band-edge and Fermi-referenced integration")
    format_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output, dpi=220, facecolor="white")
    return fig


def plot_transport(EF: np.ndarray, group: str,
                   RH_band: np.ndarray, RH_fermi: np.ndarray,
                   Rs_band: np.ndarray, Rs_fermi: np.ndarray, output: Path) -> plt.Figure:
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(12.0, 6.2), constrained_layout=True)
    fig.suptitle(f"{group} transport curves from the two carrier-count definitions", fontsize=15)

    ax_h.plot(EF, RH_band, color=BLUE, linewidth=2.2, label="from band-edge n,p")
    ax_h.plot(EF, RH_fermi, color=RED, linewidth=2.2, label="from Fermi-referenced n,p")
    ax_h.axhline(0.0, color="#777777", linewidth=0.9)
    ax_h.set_ylabel(r"$R_H$ ($\Omega$/T)")
    ax_h.set_title("Hall resistance comparison")
    format_axes(ax_h)
    ax_h.legend(frameon=False, loc="upper left")

    ax_s.plot(EF, Rs_band, color=BLUE, linewidth=2.2, label="from band-edge n,p")
    ax_s.plot(EF, Rs_fermi, color=RED, linewidth=2.2, label="from Fermi-referenced n,p")
    ax_s.set_ylim(bottom=0.0)
    ax_s.set_ylabel(r"$R_s$ ($\Omega$/sq)")
    ax_s.set_title("Sheet resistance comparison")
    format_axes(ax_s)
    ax_s.legend(frameon=False, loc="upper left")

    fig.savefig(output, dpi=220, facecolor="white")
    return fig


def load_params(data_path: Path) -> tuple[dict, str]:
    """Load L1 parameters from a regular_GROUP folder or JSON file."""
    resolved = data_path.resolve()
    params_file = resolved / "L1_params.json" if resolved.is_dir() else resolved
    if not params_file.is_file():
        raise FileNotFoundError(f"Missing input: {params_file}")
    params = json.loads(params_file.read_text(encoding="utf-8"))
    missing = {"c", "T", "EF_op_eV"} - set(params)
    if missing:
        raise KeyError(f"Missing required parameter(s) in {params_file}: {sorted(missing)}")
    return params, str(params_file)


def mobility_from_params(params: dict, T_K: float, c: float, EF_op: float,
                         explicit_mobility: float | None) -> tuple[float, str]:
    """Use --mobility, or derive mobility from rho_ohm_sq and n_Fermi(EF_op)."""
    if explicit_mobility is not None:
        return explicit_mobility, "--mobility"
    if "rho_ohm_sq" not in params:
        raise KeyError("Missing rho_ohm_sq in parameter file; pass --mobility explicitly.")
    rho_ohm_sq = float(params["rho_ohm_sq"])
    _, n_fermi_op, _, _ = carrier_counts(EF_op, T_K, c)
    return 1.0 / (E_CHARGE_C * n_fermi_op * rho_ohm_sq), "rho_ohm_sq and n_Fermi(EF_op)"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="regular_GROUP folder or L1_params.json file")
    parser.add_argument("--out", type=Path, default=script_dir,
                        help="Output directory (default: beside this script)")
    parser.add_argument("--mobility", type=float,
                        help="Constant mobility in cm^2/(V s) used in Rs; "
                             "if omitted, derive it from rho_ohm_sq")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                        help="Display the three figures after saving (default: show)")
    args = parser.parse_args()

    params, params_source = load_params(args.data)
    group = str(params.get("group") or Path(params_source).parent.name).upper()
    T_K = float(params["T"])
    c = float(params["c"])
    EF_op = float(params["EF_op_eV"])
    mobility, mobility_source = mobility_from_params(params, T_K, c, EF_op, args.mobility)
    EF = np.linspace(-0.25, 0.25, 501)

    counts = np.array([carrier_counts(x, T_K, c) for x in EF])
    n_band, n_fermi, p_band, p_fermi = counts.T

    RH_band = hall_resistance(n_band, p_band)
    RH_fermi = hall_resistance(n_fermi, p_fermi)
    Rs_band = sheet_resistance(n_band, p_band, mobility)
    Rs_fermi = sheet_resistance(n_fermi, p_fermi, mobility)

    middle = len(EF) // 2
    assert abs(n_band[middle] - n_fermi[middle]) / n_band[middle] < 2e-6
    assert abs(p_band[middle] - p_fermi[middle]) / p_band[middle] < 2e-6
    assert abs(RH_band[middle]) < 1e-9
    assert abs(RH_fermi[middle]) < 1e-9

    args.out.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_density_comparison(EF, n_band, n_fermi, "n", EF_op, group,
                                args.out / "06_n_band_vs_n_fermi.png"),
        plot_density_comparison(EF, p_band, p_fermi, "p", EF_op, group,
                                args.out / "07_p_band_vs_p_fermi.png"),
        plot_transport(EF, group, RH_band, RH_fermi, Rs_band, Rs_fermi,
                       args.out / "05_band_vs_fermi_RH_Rs.png"),
    ]

    with (args.out / "final_band_vs_fermi_data.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "EF_eV", "n_band_cm2", "n_fermi_cm2", "p_band_cm2", "p_fermi_cm2",
            "RH_band_ohm_per_T", "RH_fermi_ohm_per_T",
            "Rs_band_ohm_sq", "Rs_fermi_ohm_sq",
        ])
        writer.writerows(zip(
            EF, n_band, n_fermi, p_band, p_fermi,
            RH_band, RH_fermi, Rs_band, Rs_fermi,
        ))

    print(f"Parameter source: {params_source}")
    print(f"Group: {group}")
    print(f"Mobility: {mobility:.12g} cm^2/(V s) from {mobility_source}")
    print(f"Created final outputs in: {args.out.resolve()}")
    if args.show:
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
