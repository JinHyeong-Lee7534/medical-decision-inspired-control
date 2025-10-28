# medical-decision-inspired-control
Medical decision–inspired Proactive Latency Control (PLC) framework built on WebRTC GCC. Combines pre-intervention band, self-damping gain, and multi-resource allocation to ensure temporal stability in real-time human–machine interaction systems. Reproducible Python simulation and statistical validation included.

**Version:** 1.1.0 • **License:** MIT • **Python:** 3.8+

Medical decision–inspired **Proactive Latency Control (PLC)** layer on top of WebRTC GCC.  
Reproducible simulation code with paired bootstrap tests, block bootstrap, effect sizes, and manifest logging.

> Core script: `Final_B2plusPLC_ver1_1_COMPLETE.py`

---

## 📦 Install & Run

1. Download the main simulation script from GitHub:
   [Final_B2plusPLC_ver1_1_COMPLETE.py](https://github.com/JinHyeong-Lee7534/medical-decision-inspired-control/blob/main/Final_B2plusPLC_ver1_１_COMPLETE.py)
```

## ▶️ Quick Start： Run it in your local Python environment (Python 3.8+)

```bash
python Final_B2plusPLC_ver1_1_COMPLETE.py --N 30000 --boot 300 --seed 4242
```

**Outputs (CSV):**
- `results_summary.csv`
- `results_bootstrap_tests.csv`
- `results_bootstrap_block_tests.csv`
- `results_lambda_sweep.csv`
- `results_ablation_summary.csv`
- `results_ablation_bootstrap.csv`
- `results_effect_sizes.csv`
- `run_manifest.json`

## 🔬 What’s inside? (high level)

- B2 baseline: WebRTC GCC (Kalman trend + loss-aware AIMD) simulation  
- PLC enhancement: Pre-intervention band, self-damping gain, multi-resource pre-allocation  
- Statistics: paired bootstrap (mean/variance/p99/compliance), Holm–Bonferroni correction  
- Effect sizes: Cohen’s d_z for key metrics  
- Reproducibility: exact seeds + manifest file logging environment and args

## 🧪 Reproduce Full Experiment

```bash
python Final_B2plusPLC_ver1_1_COMPLETE.py --N 100000 --boot 2000 --seed 4242
```

## 📚 Citation

Please cite via `CITATION.cff` or the preprint listed there.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 🔐 Security

See [SECURITY.md](SECURITY.md).

## 🧑‍💻 Use of AI Assistance

See [docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md).

## 🗺️ Repository Structure

```
b2plc/
├── Final_B2plusPLC_ver1_1_COMPLETE.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── CITATION.cff
├── docs/
│   └── AI_ASSISTANCE.md
└── scripts/
    ├── run_quick.sh
    └── run_full.sh
```

## ⬆️ Publish to GitHub (commands)

```bash
# in the repo root
git init
git add .
git commit -m "chore: initial public release v1.1.0"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/b2plc.git
git push -u origin main
```

---

© 2025 Jin‑Hyeong Lee, MD. MIT License.
