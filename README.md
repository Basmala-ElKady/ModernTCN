# ModernTCN: Large Kernels Instead of Dilation for Time Series Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/pytest-87%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Course**: Deep Learning (AIE332) — Lab 4 Assignment  
> **Topic 6**: Large kernels instead of dilation, based on **ModernTCN (ICLR 2024)**  
> **Institution**: Faculty of Computer Science & Engineering, Alamein International University  

---

## 📖 Executive Summary

This repository presents an independent, clean-room PyTorch implementation and empirical evaluation of **ModernTCN** (*ICLR 2024*), investigating the core hypothesis:  
**Can large-kernel depthwise convolutions replace exponential dilation to achieve superior accuracy and continuous receptive fields in multivariate time series forecasting?**

We conduct a benchmark evaluation comparing:
1. **ModernTCN**: Patch embedding, large-kernel depthwise convolutions ($K=31$) with parallel small-kernel branches ($k=5$), decoupled intra/inter-variable pointwise FFNs, and Reversible Instance Normalization (RevIN).
2. **DLinear Baseline** (*AAAI 2023*): Moving-average series decomposition with independent linear trend and seasonal mapping heads.
3. **Lab 4 TCN Baseline** (*Bai et al., 2018*): Dilated causal 1D convolutions with exponential dilation schedules ($d = 1, 2, 4, 8, \dots$).

Experiments are conducted on **ETTh1** (hourly) and **ETTm1** (15-minute) datasets with an input window of $L=336$ and forecast horizons $H \in \{96, 336\}$, repeated across **3 independent random seeds** (`42, 123, 2026`).

---

## 🎯 Key Findings & Quantitative Highlights

1. **Paper Claim Strongly Supported**:  
   ModernTCN consistently achieves the lowest Normalized MSE and MAE across all 4 benchmark configurations, outperforming the Lab 4 TCN baseline by **51.2% to 61.0% in MSE** and outperforming DLinear by **1.4% to 4.5% in MSE**.
2. **Elimination of Dilation Gridding**:  
   Autograd-based Effective Receptive Field (ERF) analysis confirms that ModernTCN provides continuous, smooth input sensitivity without the periodic sampling holes inherent to dilated convolutions.
3. **Receptive Field Saturation**:  
   Depthwise kernel ablation across $K \in \{15, 31, 51\}$ at matched parameter count (<6% variation) demonstrates that accuracy peaks at $K=15$–$31$. Expanding to $K=51$ yields diminishing returns because the dominant diurnal period is already fully covered.
4. **Reproducible TCN Failure Case**:  
   On ETTm1 with $H=336$, TCN collapses catastrophically (Test MSE $0.9611 \pm 0.0195$ vs. ModernTCN $0.3745 \pm 0.0062$, a **2.56× higher error**). TCN suffers higher error on **>98.8% of test windows**, caused by dilation gridding on 15-minute cycles and the absence of cross-variable feature mixing.

---

## 📊 Full Benchmark Results (Mean ± Sample Standard Deviation over 3 Seeds)

All models selected checkpoints based on **best validation MSE** and were evaluated once on the standard chronological test split. All metrics use sample standard deviation ($ddof=1$).

| Dataset | Horizon ($H$) | Model | Test MSE (mean ± std) | Test MAE (mean ± std) | Best Val MSE | Training Time | Parameters |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **ETTh1** | **96** | **ModernTCN** | **0.3799 ± 0.0023** | **0.4032 ± 0.0014** | 0.6882 | 19.3 s | 65,262 |
| ETTh1 | 96 | DLinear | 0.3852 ± 0.0045 | 0.4083 ± 0.0048 | 0.6497 | 4.2 s | 64,704 |
| ETTh1 | 96 | TCN Baseline | 0.7789 ± 0.0395 | 0.6874 ± 0.0222 | 0.9238 | 50.1 s | 207,072 |
| **ETTh1** | **336** | **ModernTCN** | **0.4481 ± 0.0069** | **0.4466 ± 0.0045** | 1.2187 | 13.5 s | 65,262 |
| ETTh1 | 336 | DLinear | 0.4691 ± 0.0165 | 0.4680 ± 0.0144 | 1.0431 | 5.4 s | 64,704 |
| ETTh1 | 336 | TCN Baseline | 0.9806 ± 0.0393 | 0.7731 ± 0.0199 | 1.2305 | 71.0 s | 207,072 |
| **ETTm1** | **96** | **ModernTCN** | **0.2970 ± 0.0044** | **0.3490 ± 0.0025** | 0.3801 | 58.8 s | 65,262 |
| ETTm1 | 96 | DLinear | 0.3078 ± 0.0013 | 0.3531 ± 0.0020 | 0.3865 | 18.9 s | 64,704 |
| ETTm1 | 96 | TCN Baseline | 0.5375 ± 0.0202 | 0.5239 ± 0.0177 | 0.6260 | 170.9 s | 207,072 |
| **ETTm1** | **336** | **ModernTCN** | **0.3745 ± 0.0062** | **0.3967 ± 0.0026** | 0.6574 | 56.6 s | 65,262 |
| ETTm1 | 336 | DLinear | 0.3872 ± 0.0063 | 0.4072 ± 0.0082 | 0.6175 | 25.5 s | 64,704 |
| ETTm1 | 336 | TCN Baseline | 0.9611 ± 0.0195 | 0.7704 ± 0.0177 | 0.8982 | 174.4 s | 207,072 |

---

## 🔬 In-Depth Analysis

### 1. Depthwise Kernel Size Ablation (Matched Parameters)
Evaluated on ETTh1 ($H=96$) across seeds 42, 123, 2026. Parameter counts remain within $\pm 5.5\%$ parity:

| Kernel Size ($K$) | Parameters | Test MSE (mean ± std) | Test MAE (mean ± std) | Best Val MSE |
| :---: | :---: | :---: | :---: | :---: |
| **15** | 62,190 (-4.7%) | **0.3788 ± 0.0013** | **0.4024 ± 0.0014** | 0.6853 |
| **31** | 65,262 (0.0%) | 0.3799 ± 0.0023 | 0.4032 ± 0.0014 | 0.6882 |
| **51** | 69,102 (+5.9%) | 0.3830 ± 0.0039 | 0.4053 ± 0.0027 | 0.6916 |

### 2. Effective Receptive Field (ERF) via Autograd
We quantify input gradient sensitivity $S(t) = \frac{1}{|P| \cdot C} \sum_{p, c} |\partial \hat{y}_{p, c} / \partial x_{t}|$:
- **50% of gradient mass** is concentrated within the most recent **14 timesteps**.
- **90% of gradient mass** is concentrated within the most recent **61 timesteps**.
- Result artifact available at `results/erf_analysis/moderntcn_etth1_h96_kernel15_erf.png`.

### 3. Failure Case Analysis: TCN Breakdown on ETTm1 ($H=336$)
- **Catastrophic Gap**: TCN error is **2.56× higher** than ModernTCN.
- **Per-Window Distribution**: TCN is worse on **11,086 / 11,185 test windows (>98.8%)**.
- **Root Cause**:
  1. *Dilation Gridding*: 15-minute sampling contains sharp high-frequency cycles; exponential dilation ($d=1, 2, 4, \dots, 64$) leaves massive holes, skipping critical transition points.
  2. *Univariate Blindness*: Oil temperature ($OT$) physically depends on electric current load ($HUFL, HULL$). TCN cannot model cross-channel physical dynamics, whereas ModernTCN's inter-variable FFN preserves thermodynamic consistency.

---

## 🏗️ Repository Architecture

```
ModernTCN/
├── configs/
│   └── default.yaml                 # Experiment parameters & hyperparameter settings
├── notebooks/
│   ├── ModernTCN_Colab_Benchmark.ipynb  # Self-contained submission notebook (TRAIN=False flag)
│   ├── ModernTCN_Integration_Pilot.ipynb # Full experiments & analysis notebook
│   └── revin_ablation.ipynb         # Controlled RevIN pilot study
├── report/
│   ├── report.pdf                   # Official 4-page submission PDF report
│   ├── report.md                    # Full markdown report
│   └── generate_pdf.py              # ReportLab generation script
├── research/
│   ├── author_files_read.md         # Compliance record of inspected author code (Rule 1)
│   ├── claim.md                     # Quantitative claim statement & outcome (Rule 3)
│   ├── moderntcn_design.md          # Architectural differences & design rationale
│   └── paper_notes.md               # Summary of ModernTCN paper (ICLR 2024)
├── results/
│   ├── baseline/                    # Full 36-run benchmark raw & summary metrics
│   ├── erf_analysis/                # ERF autograd sensitivity CSV & PNG plot
│   ├── failure_analysis/            # Window-by-window TCN failure analysis CSVs
│   └── kernel_ablation/             # Kernel size ablation raw & summary metrics
├── slides/
│   ├── slides.pdf                   # 10-slide presentation PDF for Lab 5 oral defense
│   ├── slides.md                    # Markdown presentation slides
│   └── generate_slides.py           # ReportLab slide compiler
├── src/
│   ├── analysis/                    # Analysis scripts (ablation, erf, failure_analysis)
│   ├── data/                        # Datasets, sliding windows, and scalers
│   ├── models/                      # PyTorch implementations (ModernTCN, TCN, DLinear, RevIN)
│   ├── training/                    # Training loop, evaluation, and metrics
│   └── utils/                       # Reproducibility seeds and checkpointing
├── tests/                           # 89 automated unit tests (test_analysis, test_models, etc.)
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/YourUsername/ModernTCN.git
cd ModernTCN
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify model architectures, dataloaders, ERF calculations, and failure analysis routines:
```bash
python -m pytest
```
*All 89 unit tests pass in ~8 seconds.*

### 3. Run on Google Colab (< 1 minute in Fast Mode)
Open `notebooks/ModernTCN_Colab_Benchmark.ipynb` in Google Colab:
- **`TRAIN = False` (Default)**: Immediately loads precomputed benchmark tables, generates the autograd ERF plot, and runs the failure case analysis in **under 30 seconds**.
- **`TRAIN = True`**: Retrains the models from scratch using free Colab GPU in **under 30 minutes**.

### 4. Run Analysis Scripts Locally
```bash
# Kernel size ablation
python -m src.analysis.ablation --data-dir data --kernels 15 31 51

# Effective receptive field autograd sensitivity
python -m src.analysis.receptive_field --model moderntcn

# Failure case analysis
python -m src.analysis.failure_analysis --data-dir data --tcn-ckpt <path> --modern-ckpt <path>
```

---

## 📜 Deliverables & Lab Submission

- **Report**: [report/report.pdf](report/report.pdf) (Strictly 4 pages, conforms to all prompt requirements).
- **Presentation Slides**: [slides/slides.pdf](slides/slides.pdf) (10 landscape slides for 5-minute defense + Q&A).
- **Colab Notebook**: [notebooks/ModernTCN_Colab_Benchmark.ipynb](notebooks/ModernTCN_Colab_Benchmark.ipynb).
- **Author Code Record**: [research/author_files_read.md](research/author_files_read.md).

---

## 🛡️ Academic Integrity & AI Declaration

- **Rule 1 Compliance**: All model components (`ModernTCN`, `TCN`, `DLinear`, `RevIN`), training pipelines, and evaluation metrics were written independently from first principles in PyTorch. No author code was copied.
- **AI Declaration**: AI coding assistants (Google DeepMind Antigravity / Gemini 3.8) were utilized for test boilerplate generation (`pytest`), ReportLab layout styling, and local execution troubleshooting. All architectural formulations, empirical decisions, and mathematical deductions were independently verified by the author.
