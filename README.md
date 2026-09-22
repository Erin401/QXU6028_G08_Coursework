# QXU6028 Group G08 - L1 Deliverable

This repository contains the source data, analysis scripts, numerical results, validation figures, and supplementary carrier-count comparison for the QXU6028 Group G08 L1 task.

## 1. How to Reproduce the L1 Results

Install the required Python packages:

```bash
python -m pip install numpy pandas matplotlib
```

From the repository root, run:

```bash
python reproduce_L1.py --data data/regular_G08 --out outputs
```

The script reads the following input files:

```text
data/regular_G08/L1_data.csv
data/regular_G08/L1_params.json
```

The regenerated files are saved in the `outputs` folder.

## 2. How to Reproduce the Four-Curve Results

This supplementary analysis compares the band-edge and Fermi-level carrier-count definitions and their corresponding \(R_H\) and \(R_s\) results.

From the repository root, run:

```bash
python Supplemental_L1_Rh_Rs/G08_final_band_fermi.py --data data/regular_G08 --out Supplemental_L1_Rh_Rs --no-show
```

This command directly uses the existing `data/regular_G08/L1_params.json` file, so no duplicate data folder is required.

## 3. Outputs

### Main L1 Outputs

| File | Description |
|---|---|
| `outputs/results.json` | Machine-readable L1 results |
| `outputs/L1_summary.json` | Full-precision carrier counts, correction factors, sweeps, and validation values |
| `outputs/L1_checks.log` | Gate-by-gate calculation and validation checks |
| `outputs/L1_counts_and_correction.png` | Comparison of the two carrier-count definitions and the correction-factor scan |
| `outputs/L1_validation.png` | Numerical reproduction and closed-form convergence validation |

### Supplemental L1 \(R_H\) and \(R_s\) Outputs

| File | Description |
|---|---|
| `Supplemental_L1_Rh_Rs/05_band_vs_fermi_RH_Rs.png` | Comparison of \(R_H\) and \(R_s\) using the band-edge and Fermi-level carrier definitions |
| `Supplemental_L1_Rh_Rs/06_n_band_vs_n_fermi.png` | Comparison of `n_band` and `n_Fermi` over the Fermi-energy sweep |
| `Supplemental_L1_Rh_Rs/07_p_band_vs_p_fermi.png` | Comparison of `p_band` and `p_Fermi` over the Fermi-energy sweep |
| `Supplemental_L1_Rh_Rs/final_band_vs_fermi_data.csv` | Numerical values used to generate the supplementary figures |

## Repository Notes

- Run all commands from the repository root.
- Both analyses use `data/regular_G08/L1_params.json`.
- Generated cache directories such as `.mplcache` and `__pycache__` are not required.
- `G08_L1_Checkpoint_for_GitHub.zip` contains the packaged L1 checkpoint files.