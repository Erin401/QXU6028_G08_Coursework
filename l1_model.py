"""
QXU6028 Polymer Devices - Virtual Lab
L1 forward model: carrier counting and mobility correction.

Reference: QXU6028_Virtual_Lab_Student_Handbook_2026-27.pdf, Part 3,
           Section "L1 - a forward model of the carrier count"
           (theory: QXU6028_Chapter_3_Carrier_Counting_and_Mobility_2026-27.pdf)

Physics
-------
Both carrier counts use the SAME linear density of states and the SAME
finite-temperature Fermi function. They differ only in the LOWER BOUND of the
integral:

    n_band(E_F) = N_A * c * Int_{0}^{E_max}    E * f(E, E_F, T) dE   (band edge)
    n_Fermi(E_F) = N_A * c * Int_{E_F}^{E_max} E * f(E, E_F, T) dE   (Fermi referenced)

For the supplied positive-doping sweep, the Fermi-referenced integral starts
part-way up the same spectrum and counts fewer electrons, so the ratio

    C(E_F) = n_band / n_Fermi   >= 1

is the factor by which a textbook band-edge count over-reports the carrier
density actually available for transport in this sweep. C is independent of the DoS slope c
(c multiplies both integrals), so it is a pure ratio diagnostic.

The Fermi-referenced integral also has a closed form (handbook Part 3, p. 18):

    n_Fermi(E_F) ~ N_A * c * k_B T * ( E_F * ln2 + (pi^2 / 12) * k_B T )

which is used here as an independent computational cross-check.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Physical constants (handbook Part 3, "Constants" table)
# ----------------------------------------------------------------------------
K_B_EV_PER_K = 8.617333e-5        # Boltzmann constant, eV/K
N_A_CM2 = 3.816e15                # areal atom density, atoms/cm^2
E_CHARGE_C = 1.602176634e-19      # elementary charge, C
E_MAX_EV = 2.5                    # integration upper bound, eV

N_POINTS_L1 = 20001               # handbook Part 3 grid for L1
EXP_CLIP = (-60.0, 60.0)          # clip the exponent argument to avoid overflow


def fermi_dirac(E: np.ndarray, E_F: float, T: float) -> np.ndarray:
    """Finite-temperature Fermi function with a clipped exponent."""
    x = np.clip((E - E_F) / (K_B_EV_PER_K * T), *EXP_CLIP)
    return 1.0 / (np.exp(x) + 1.0)


def n_band(E_F: float, T: float, c: float, n_points: int = N_POINTS_L1) -> float:
    """Textbook band-edge carrier count: integrate from E = 0."""
    E = np.linspace(0.0, E_MAX_EV, n_points)
    return N_A_CM2 * c * np.trapezoid(E * fermi_dirac(E, E_F, T), E)


def n_fermi(E_F: float, T: float, c: float, n_points: int = N_POINTS_L1) -> float:
    """Fermi-referenced carrier count: integrate from E = E_F."""
    E = np.linspace(E_F, E_MAX_EV, n_points)
    return N_A_CM2 * c * np.trapezoid(E * fermi_dirac(E, E_F, T), E)


def n_fermi_closed_form(E_F: float, T: float, c: float) -> float:
    """Closed form of the Fermi-referenced integral (density-independent)."""
    kT = K_B_EV_PER_K * T
    return N_A_CM2 * c * kT * (E_F * np.log(2.0) + (np.pi**2 / 12.0) * kT)


def correction_factor(E_F: float, T: float, c: float) -> float:
    """C(E_F) = n_band / n_Fermi."""
    return n_band(E_F, T, c) / n_fermi(E_F, T, c)


def mobility_from_density(n_cm2: float, rho_ohm_sq: float) -> float:
    """
    Two-dimensional sheet mobility from a sheet density and a sheet resistance.

        rho = 1 / (e * n * mu)   ->   mu = 1 / (e * n * rho)

    Units: n in cm^-2, rho in ohm/sq -> mu in cm^2 V^-1 s^-1 directly, with n
    being a carrier DENSITY (cm^-2). No extra 1e4 factor: the 1e4 factor in the
    handbook formula applies to the 3D volume density and thickness, neither of
    which is in play here.
    """
    return 1.0 / (E_CHARGE_C * n_cm2 * rho_ohm_sq)


# ----------------------------------------------------------------------------
# Implementation gates (handbook Part 3, p. 18 table)
# ----------------------------------------------------------------------------
GATE_TABLE = [
    (0.0, 1.00),
    (0.073, 2.00),
    (0.100, 2.60),
    (0.200, 5.10),
]


def run_gates(T: float, c: float, tol: float = 0.02) -> list[dict]:
    """Evaluate the handbook's verified-ratio table and the monotonicity gate."""
    rows = []
    for E_F, expected in GATE_TABLE:
        got = correction_factor(E_F, T, c)
        rows.append(
            {
                "EF_eV": E_F,
                "expected": expected,
                "computed": got,
                "abs_err": abs(got - expected),
                "pass": abs(got - expected) <= tol,
            }
        )

    sweep = np.linspace(0.0, 0.25, 251)
    C = np.array([correction_factor(e, T, c) for e in sweep])
    rows.append(
        {
            "EF_eV": "monotonic 0->0.25",
            "expected": "non-decreasing",
            "computed": bool(np.all(np.diff(C) >= -1e-12)),
            "abs_err": np.nan,
            "pass": bool(np.all(np.diff(C) >= -1e-12)),
        }
    )
    return rows


CLOSED_FORM_TOL = 1e-5    # trapezoid discretisation floor at the handbook grid


def closed_form_agreement(T: float, c: float, tol: float = CLOSED_FORM_TOL) -> dict:
    """
    Numerical vs closed-form Fermi-referenced count on a representative grid.

    The closed form is analytically exact for an infinite upper limit, so any
    residual is the composite trapezoid rule's O(h^2) error at the handbook grid
    (20001 points over 2.5 eV). The tolerance therefore sits at the
    discretisation floor, not at machine precision.
    """
    worst = 0.0
    for E_F in np.linspace(0.0, 0.25, 26):
        num = n_fermi(E_F, T, c)
        ana = n_fermi_closed_form(E_F, T, c)
        worst = max(worst, abs(num - ana) / ana)
    return {"max_rel_diff": worst, "pass": worst <= tol, "tol": tol}


def closed_form_convergence(T: float, c: float, E_F: float = 0.08479342068805736) -> dict:
    """
    Grid-refinement study: the closed form is exact, so the residual must fall
    as h^2 (observed convergence order ~2). Reporting the order turns 'the
    numbers agree' into evidence that the residual is discretisation, not a bug.
    """
    counts = [2001, 20001, 200001]
    errs = []
    for n in counts:
        num = n_fermi(E_F, T, c, n_points=n)
        ana = n_fermi_closed_form(E_F, T, c)
        errs.append(abs(num - ana) / ana)

    order = float(np.log(errs[0] / errs[1]) / np.log(10.0))
    return {
        "E_F_eV": E_F,
        "n_points": counts,
        "rel_err": errs,
        "observed_order_first_decade": order,
        "pass": (errs[2] < errs[1] < errs[0]) and abs(order - 2.0) < 0.1,
    }
