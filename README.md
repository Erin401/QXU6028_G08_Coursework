# QXU6028 Group G08 — L1 deliverable

One-command reproduction:

```bash
python reproduce_L1.py --data data/regular_G08 --out outputs --commit "$(git rev-parse HEAD)"
```

## Outputs

| File | Content |
|---|---|
| `outputs/results.json` | Machine-readable schema QXU6028-results-v1 with L1 complete |
| `outputs/L1_summary.json` | Full-precision working numbers, gates, crossings, sweep tables |
| `outputs/L1_checks.log` | Verbose gate-by-gate log |
| `outputs/L1_counts_and_correction.png` | Two-count model + correction-factor scan |
| `outputs/L1_validation.png` | Reproduction residual + closed-form convergence |
| `outputs/pipeline_commit.txt` | Git commit that produced the outputs |
| `L1_writeup.md` | Report-ready prose for the L1 section |

## Git

- Source commit recorded in this L1 checkpoint: `306555c4877d872ef6f3f77602233716f9d1a179`
- The repository includes the L1 source data, analysis code, report text and checkpoint evidence needed to review the result.
- Cache directories such as `.mplcache` and `__pycache__` are excluded.
