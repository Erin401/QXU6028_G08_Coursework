# QXU6028 Group G08 — L1 Forward Model and Mobility Correction

## 1. Problem Statement

The manufacturer’s data sheet reports the carrier density of the graphene-on-sapphire device using a textbook **band-edge count**, which counts all occupied states from the Dirac point ($E=0$) upwards. The physical quantity that sets the transport properties of the device, however, is the **Fermi-referenced count**, which counts states from the operating Fermi level $E_F$ to the top of the band.

L1 determines the factor by which the data-sheet count over-reports the carrier density and how this correction propagates into the reported mobility.

## 2. Method

Two carrier counts are evaluated using the **same linear density of states**

```math
g(E)=cE
```

and the **same finite-temperature Fermi function** $f(E,E_F,T)$. They differ only in the lower integration limit:

```math
n_{\mathrm{band}}(E_F)
=
N_Ac\int_{0}^{2.5\,\mathrm{eV}}E f(E,E_F,T)\,dE
```

```math
n_{\mathrm{Fermi}}(E_F)
=
N_Ac\int_{E_F}^{2.5\,\mathrm{eV}}E f(E,E_F,T)\,dE
```

The constants and integration grid follow Part 3 of the handbook:

```math
N_A=3.816\times10^{15}\ \mathrm{cm^{-2}}
```

```math
k_B=8.617333\times10^{-5}\ \mathrm{eV\,K^{-1}},
\qquad
E_{\max}=2.5\ \mathrm{eV}
```

The numerical integration uses 20,001 grid points. The carrier-density correction factor is

```math
C(E_F)
=
\frac{n_{\mathrm{band}}(E_F)}
     {n_{\mathrm{Fermi}}(E_F)}
\geq 1
```

The correction factor $C$ is independent of the density-of-states slope $c$ because $c$ cancels in the ratio.

The Fermi-referenced integral also has the following closed-form approximation:

```math
n_{\mathrm{Fermi}}(E_F)
\approx
N_Ac k_BT
\left(
E_F\ln 2+\frac{\pi^2}{12}k_BT
\right)
```

This expression is used as a cross-check of the numerical integration.

For a fixed sheet resistance $\rho$, the mobility follows from

```math
\rho=\frac{1}{en\mu}
```

Therefore,

```math
\mu_{\mathrm{corrected}}
=
\frac{1}{e n_{\mathrm{Fermi}}\rho}
=
C\frac{1}{e n_{\mathrm{band}}\rho}
=
C\mu_{\mathrm{band}}
```

All calculations are performed at the **exact operating point** supplied in `L1_params.json`, rather than at the nearest row of `L1_data.csv`.

## 3. Results

| Quantity | Value | Note |
|---|---:|---|
| Operating $E_F$ | $0.08479342068805736\ \mathrm{eV}$ | From `L1_params.json` |
| Sheet resistance $\rho$ | $256.1152083335392\ \Omega/\mathrm{sq}$ | Constant across the sweep |
| Data-sheet carrier density $n_{\mathrm{band}}$ | $7.648284\times10^{11}\ \mathrm{cm^{-2}}$ | Band-edge count |
| Corrected carrier density $n_{\mathrm{Fermi}}$ | $3.389119\times10^{11}\ \mathrm{cm^{-2}}$ | Fermi-referenced count |
| Correction factor $C$ | $2.256717$ | Over-reporting by approximately $2.26\times$ |
| Data-sheet mobility $\mu_{\mathrm{band}}$ | $3.1863\times10^{4}\ \mathrm{cm^2\,V^{-1}\,s^{-1}}$ | Calculated using $n_{\mathrm{band}}$ |
| Corrected mobility $\mu_{\mathrm{corrected}}$ | $7.1906\times10^{4}\ \mathrm{cm^2\,V^{-1}\,s^{-1}}$ | Calculated using $n_{\mathrm{Fermi}}$ |
| $\mu_{\mathrm{corrected}}/\mu_{\mathrm{band}}$ | $2.256717$ | Equal to $C$ |
| $C=2$ crossing | $0.07276\ \mathrm{eV}$ | $2.815\,k_BT$ |
| $C=4$ crossing | $0.15735\ \mathrm{eV}$ | $6.086\,k_BT$ |

The ratio $\mu_{\mathrm{corrected}}/\mu_{\mathrm{band}}$ is identical to $C$ to machine precision. This confirms that, at fixed sheet resistance, the mobility changes by the same factor as the carrier-density correction.

## 4. Validation and Implementation Gates

| Gate | Test | Result |
|---|---|---|
| Model reproduction of the `L1_data.csv` $n_{\mathrm{band}}$ column | Maximum relative deviation | $2.967\times10^{-12}$ — **PASS** |
| Handbook ratio table | $C(0)=1.00$, $C(0.073)\approx2.00$, $C(0.100)\approx2.60$, and $C(0.200)\approx5.10$ | All within $0.5\%$ — **PASS** |
| Monotonicity | $C(E_F)$ is non-decreasing over $0\leq E_F\leq0.25\ \mathrm{eV}$ | **PASS** |
| Closed-form cross-check | Closed-form result compared with numerical integration | Residual $1.184\times10^{-6}$ using 20,001 points — **PASS** |
| Grid convergence | Observed order using 2,001, 20,001, and 200,001 points | $2.000$ — **PASS** |
| CSV-derived $C$ compared with model-derived $C$ | Maximum relative difference | $2.967\times10^{-12}$ — **PASS** |
| Mobility relation | $\mu_{\mathrm{corrected}}/\mu_{\mathrm{band}}=C$ | Exact within $10^{-12}$ — **PASS** |

The closed-form residual is consistent with the expected second-order trapezoidal integration error, $O(h^2)$.

## 5. Why the Exact Operating Point Matters

Using the nearest row of `L1_data.csv`, corresponding to $E_F=0.09\ \mathrm{eV}$, would give

```math
C=2.3719
```

This would introduce an error of **+5.103%** relative to the correction calculated at the exact operating point.

The calculations therefore evaluate $n_{\mathrm{band}}(E_F)$ and $n_{\mathrm{Fermi}}(E_F)$ directly at the exact value

```math
E_{F,\mathrm{op}}=0.08479342068805736\ \mathrm{eV}
```

supplied in `L1_params.json`, as required by the practical guide.

## 6. Quantified Sensor Recommendation

At the intended operating point, the band-edge model over-reports the G08 carrier density by a factor of

```math
C=2.26
```

At fixed sheet resistance, it consequently under-reports the extracted mobility by the same factor. The corrected quantities are

```math
n_{\mathrm{Fermi}}
=
3.39\times10^{11}\ \mathrm{cm^{-2}}
```

and

```math
\mu_{\mathrm{corrected}}
=
7.19\times10^{4}\ \mathrm{cm^2\,V^{-1}\,s^{-1}}
```

Any sensor calibration that converts electrical response into carrier density or mobility should therefore use the Fermi-referenced count. Otherwise, the extracted transport scale will contain this factor-of-2.26 model error.

The present L1 data do not by themselves determine the response of a particular analyte or sensing mechanism.

## 7. Supplemental L1 R<sub>H</sub> and R<sub>s</sub> Analysis

The supplemental analysis extends the L1 carrier-count comparison to both electrons and holes. It evaluates the band-edge counts $n_{\mathrm{band}}$ and $p_{\mathrm{band}}$, together with the Fermi-referenced counts $n_{\mathrm{Fermi}}$ and $p_{\mathrm{Fermi}}$, over the range

```math
-0.25\leq E_F\leq0.25\ \mathrm{eV}
```

The following equations are used to calculate the Hall coefficient and sheet resistance:

```math
R_H=\frac{p-n}{e(p+n)^2}
```

```math
R_s=\frac{1}{e\mu(p+n)}
```
