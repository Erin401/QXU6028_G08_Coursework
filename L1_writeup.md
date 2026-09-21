# QXU6028 Group G08 — L1 forward model and mobility correction

## 1. Problem statement

The manufacturer’s data-sheet reports the carrier density of the graphene-on-sapphire device using a textbook **band-edge count**, i.e. it counts all occupied states from the Dirac point (`E = 0`) upwards. The physical quantity that sets the transport properties of the device, however, is the **Fermi-referenced count**, i.e. the count from the operating Fermi level `E_F` up to the top of the band. L1 asks by what factor the data-sheet count over-reports the carrier density and how that propagates into the reported mobility.

## 2. Method

Two carrier counts are evaluated with the **same linear density of states** `g(E) = c·E` and the **same finite-temperature Fermi function** `f(E, E_F, T)`; they differ only in the lower limit of the integral:

```
n_band(E_F)  = N_A · c · ∫_{0}^{2.5 eV}    E·f(E, E_F, T) dE
n_Fermi(E_F) = N_A · c · ∫_{E_F}^{2.5 eV}  E·f(E, E_F, T) dE
```

Constants and grid follow the handbook Part 3 (`N_A = 3.816×10¹⁵ cm⁻²`, `k_B = 8.617333×10⁻⁵ eV K⁻¹`, `E_max = 2.5 eV`, 20,001 integration points). The correction factor is

```
C(E_F) = n_band(E_F) / n_Fermi(E_F)  ≥ 1
```

`C` is independent of the DoS slope `c` because `c` cancels. The Fermi-referenced integral also has an exact closed form (Sommerfeld identity):

```
n_Fermi(E_F) ≈ N_A · c · k_B T · (E_F·ln 2 + (π²/12)·k_B T)
```

which is used here as a cross-check. For fixed sheet resistance `ρ`, the mobility follows from `ρ = 1/(e·n·μ)`:

```
μ_corrected = 1/(e·n_Fermi·ρ) = C · 1/(e·n_band·ρ) = C · μ_band
```

All calculations are performed at the **exact operating point** supplied in `L1_params.json`, not at the nearest row of `L1_data.csv`.

## 3. Results

| Quantity | Value | Note |
|---|---|---|
| Operating `E_F` | `0.08479342068805736 eV` | from `L1_params.json` |
| Sheet resistance `ρ` | `256.1152083335392 Ω/sq` | constant across the sweep |
| Data-sheet carrier density `n_band` | `7.648284×10¹¹ cm⁻²` | band-edge count |
| Corrected carrier density `n_Fermi` | `3.389119×10¹¹ cm⁻²` | Fermi-referenced count |
| Correction factor `C` | `2.256717` | over-reporting by ~2.26× |
| Data-sheet mobility `μ_band` | `3.1863×10⁴ cm² V⁻¹ s⁻¹` | from `n_band` |
| Corrected mobility `μ_corrected` | `7.1906×10⁴ cm² V⁻¹ s⁻¹` | from `n_Fermi` |
| `μ_corrected / μ_band` | `2.256717` | equal to `C`, as required |
| `C = 2` crossing | `0.07276 eV` | `2.815 k_B T` |
| `C = 4` crossing | `0.15735 eV` | `6.086 k_B T` |

The ratio `μ_corrected/μ_band` is identical to `C` (to machine precision), confirming that at fixed resistivity the mobility changes by the same factor as the carrier density correction.

## 4. Validation / implementation gates

| Gate | Test | Result |
|---|---|---|
| Model reproduces `L1_data.csv` `n_band` column | max relative deviation | `2.967×10⁻¹²` — **PASS** |
| Handbook ratio table | `C(0)=1.00`, `C(0.073)≈2.00`, `C(0.100)≈2.60`, `C(0.200)≈5.10` | all within 0.5% — **PASS** |
| Monotonicity | `C(E_F)` non-decreasing on `0 → 0.25 eV` | **PASS** |
| Closed-form cross-check | exact formula vs numerical integral | residual `1.184×10⁻⁶` at 20,001 points, consistent with `O(h²)` trapezoid error — **PASS** |
| Grid convergence | observed order on `2001 → 20001 → 200001` points | `2.000` — **PASS** |
| CSV-derived `C` vs model-derived `C` | max relative difference | `2.967×10⁻¹²` — **PASS** |
| Mobility relation | `μ_corrected/μ_band == C` | exact to `10⁻¹²` — **PASS** |

## 5. Why the exact operating point matters

Snapping to the nearest row of `L1_data.csv` (`E_F = 0.09 eV`) would give `C = 2.3719`, an error of **+5.103%**. The calculations therefore evaluate `n_band(E_F)` and `n_Fermi(E_F)` directly at the exact `E_F,op` supplied in the parameter file, as required by the practical guide.

## 6. Quantified sensor recommendation

At its intended operating point the band-edge model over-reports the G08 carrier density by a factor of **C = 2.26** and, at fixed sheet resistance, under-reports the extracted mobility by the same factor. The corrected carrier density is **3.39×10¹¹ cm⁻²** and the corrected mobility is **7.19×10⁴ cm² V⁻¹ s⁻¹**. Any sensor calibration that converts electrical response into carrier density or mobility should therefore use the Fermi-referenced count; otherwise its extracted transport scale will carry this factor-of-2.26 model error. The present L1 data do not by themselves determine the response of a particular analyte or sensing mechanism.
