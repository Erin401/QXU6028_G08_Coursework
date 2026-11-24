# QXU6028 Group G08 — Repository Structure and L1 Reproduction Guide

The original repository was created for the L1 submission, so the L1 files remain in the repository root. These include data, outputs, Supplemental_L1_Rh_Rs, l1_model.py, reproduce_L1.py, L1_briefnote.md, and the L1 checkpoint archive. Their locations have been kept unchanged to preserve the existing L1 data paths and reproduction commands documented in this README. The later L2 and L3 work has been added separately under GNN_L2_Submission and GNN_L3_Submission. 

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

Before running the command, check the end of the PowerShell line.

If it ends with:

```text
QXU6028_G08_L1-main>
```

run:

```powershell
python Supplemental_L1_Rh_Rs/G08_final_band_fermi.py --data data/regular_G08 --out Supplemental_L1_Rh_Rs --no-show
```

If the PowerShell line ends with:

```text
Supplemental_L1_Rh_Rs>
```

first return to the main project folder:

```powershell
cd ..
```

Then run:

```powershell
python Supplemental_L1_Rh_Rs/G08_final_band_fermi.py --data data/regular_G08 --out Supplemental_L1_Rh_Rs --no-show
```

This command uses the existing `data/regular_G08/L1_params.json` file and saves the generated results in `Supplemental_L1_Rh_Rs`.

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
