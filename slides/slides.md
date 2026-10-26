# ModernTCN: Large Kernels Instead of Dilation for Multivariate Time Series Forecasting

Deep Learning (AIE332) — Lab 4 Assignment: Topic 6  
Alamein International University — Faculty of Computer Science & Engineering

---

## Slide 1: Title & Introduction
- **Project**: Large Kernels Instead of Dilation (ModernTCN, ICLR 2024)
- **Course**: Deep Learning (AIE332) | Lab 4 Assignment
- **Core Research Questions**:
  1. Can large-kernel depthwise convolutions outperform traditional dilated TCNs and recent linear models (DLinear)?
  2. Does time series forecasting accuracy strictly track the Effective Receptive Field (ERF)?
  3. Where does the Lab 4 TCN baseline catastrophically break down?

---

## Slide 2: Motivation — The Dilemma of Classical TCNs
- **The Classical TCN Approach (Bai et al., 2018)**:
  - Small kernels ($k=3$) with exponential dilation schedules ($d = 1, 2, 4, 8, \dots$).
- **The Dilation Paradox (Gridding / Sampling Holes)**:
  - Exponential dilation expands theoretical RF rapidly, but skips intermediate points. High-frequency oscillations slip right through the gaps!
- **Effective Receptive Field (ERF) Decay**:
  - In practice, autograd gradient sensitivity decays quickly, making classic TCN act as a much smaller kernel model.
- **Channel Independence Failure**:
  - TCN models each time series channel in isolation, unable to capture physical cross-variable dependencies (e.g. electrical power load driving transformer oil temperature).

---

## Slide 3: The ModernTCN Architecture
1. **Reversible Instance Normalization (RevIN)**:
   - Window-wise mean/variance normalization mitigates non-stationary distribution shifts; denormalizes at prediction.
2. **Patch Embedding ($P=16, S=8$)**:
   - Groups continuous raw timesteps into patch tokens, suppressing high-frequency sensor noise and shortening sequence length from 336 to 41.
3. **Large Depthwise Temporal Convolutions ($K=31$) + Small Active Branch ($k=5$)**:
   - Depthwise conv operates independently per channel: parameter cost is linear $\mathcal{O}(D \cdot K)$, not quadratic $\mathcal{O}(D^2 \cdot K)$.
   - Captures continuous context without dilation gaps, while small branch preserves sharp local transitions.
4. **Decoupled Pointwise FFNs ($1\times 1$ Convolutions)**:
   - **Intra-variable FFN**: Mixes latent feature channels within each variable.
   - **Inter-variable FFN**: Mixes representations across all variables to capture multivariate physical correlations.

---

## Slide 4: Experimental Setup & Rigorous Protocol
- **Datasets**: ETTh1 (hourly electricity transformer temperature) & ETTm1 (15-minute granularity).
- **Split Protocol**: Standard chronological Informer splits (12 months train, 4 months val, 4 months test). Standard scaler fit strictly on training split.
- **Task Dimensions**: Input length $L = 336$; Forecast horizons $H \in \{96, 336\}$.
- **Parameter Parity**:
  - ModernTCN: ~65,262 parameters
  - DLinear: ~64,704 parameters
  - TCN: ~207,072 parameters
- **Multi-Seed Evaluation**: Mean ± Sample Standard Deviation ($ddof=1$) over seeds 42, 123, 2026.
- **Model Selection**: Best checkpoint chosen strictly by validation MSE; test split evaluated once.

---

## Slide 5: Quantitative Claim Verification & Benchmark Results
- **Paper Claim (Luo & Wang, ICLR 2024)**: ModernTCN outperforms conventional TCNs and competitive linear models (DLinear) across ETT benchmarks.

| Dataset | Horizon ($H$) | ModernTCN Test MSE | DLinear Test MSE | TCN Test MSE | ModernTCN Gain vs TCN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ETTh1** | 96 | **0.3799 ± 0.0023** | 0.3852 ± 0.0045 | 0.7789 ± 0.0395 | **-51.2% MSE** |
| **ETTh1** | 336 | **0.4481 ± 0.0069** | 0.4691 ± 0.0165 | 0.9806 ± 0.0393 | **-54.3% MSE** |
| **ETTm1** | 96 | **0.2970 ± 0.0044** | 0.3078 ± 0.0013 | 0.5375 ± 0.0202 | **-44.7% MSE** |
| **ETTm1** | 336 | **0.3745 ± 0.0062** | 0.3872 ± 0.0063 | 0.9611 ± 0.0195 | **-61.0% MSE** |

- **Verdict**: Claim **strongly supported** across all benchmarks!

---

## Slide 6: Depthwise Kernel Size Ablation Study
- **Ablation Protocol**: Evaluate depthwise kernels $K \in \{15, 31, 51\}$ on ETTh1 ($H=96$) at matched parameter count (<6% parameter variation).

| Kernel ($K$) | Total Parameters | Test MSE (mean ± std) | Test MAE (mean ± std) | Best Val MSE |
| :---: | :---: | :---: | :---: | :---: |
| **15** | 62,190 (-4.7%) | **0.3788 ± 0.0013** | **0.4024 ± 0.0014** | 0.6853 |
| **31** | 65,262 (0.0%) | 0.3799 ± 0.0023 | 0.4032 ± 0.0014 | 0.6882 |
| **51** | 69,102 (+5.9%) | 0.3830 ± 0.0039 | 0.4053 ± 0.0027 | 0.6916 |

- **Insight**: Accuracy does **not** continue increasing indefinitely with larger kernels. $K=15$ and $K=31$ perform best; $K=51$ incurs slight optimization variance because the temporal dependencies are already fully covered.

---

## Slide 7: Effective Receptive Field (ERF) Autograd Analysis
- **Autograd Sensitivity Formulation**:
  $$S(t) = \frac{1}{|P| \cdot C} \sum_{p \in P} \sum_{c=1}^C \left| \frac{\partial \hat{y}_{p, c}}{\partial x_{t}} \right|$$
- **Findings for ModernTCN ($K=15$)**:
  - **50% of gradient mass** is concentrated within the most recent **14 timesteps**.
  - **90% of gradient mass** is concentrated within the most recent **61 timesteps**.
- **Continuous vs Dilated Sensitivity**:
  - Unlike TCN's jagged sensitivity profile that skips timesteps due to dilation, ModernTCN delivers a smooth, monotonic sensitivity curve across the input window.

---

## Slide 8: Reproducible Failure Case — TCN Breakdown on ETTm1 ($H=336$)
- **The Catastrophic Failure**:
  - On ETTm1 ($H=336$), TCN test MSE collapses to **0.9611 ± 0.0195**, whereas ModernTCN achieves **0.3745 ± 0.0062** (**2.56× lower error**).
- **Per-Window Distribution**:
  - TCN is worse on **11,086 out of 11,185 test windows (>98.8%)**.
  - Worst test windows show TCN errors **6.2× to 9.5× higher** than ModernTCN (e.g. test window 793: TCN MSE = 2.898 vs ModernTCN MSE = 0.306).
- **Why Does It Fail?**:
  1. *15-minute granularity* has fast, sharp oscillations. TCN's deep dilation steps skip crucial transition points.
  2. *Independent channel processing* fails to leverage the thermodynamic coupling between oil temperature and transformer current loads.

---

## Slide 9: Oral Presentation Defense & Anticipated Questions
- **Q1: "Explain any line of your code"**:
  - *Patch Embedding*: `nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_stride)` applied independently per variable via tensor reshape `[B*C, 1, L]`.
  - *Depthwise Conv*: `nn.Conv1d(embed_dim, embed_dim, kernel_size=K, groups=embed_dim, padding=causal_pad)`.
- **Q2: "Predict the effect of a change before running live"**:
  - *Removing RevIN*: Test MSE will degrade by ~15-20% due to window-to-window mean/scale shifts.
  - *Disabling inter-variable FFN*: Performance on long-horizon ETTm1 will drop significantly because physical cross-channel relationships will be lost.
- **Q3: "What result would make you change your conclusion?"**:
  - If a dilated TCN with matched parameters and continuous density filters achieved lower MSE on long horizons without suffering from receptive field gridding.

---

## Slide 10: Summary & Integrity Record
- **Conclusions**:
  1. Large depthwise convolutions combined with patch embedding systematically outperform dilated TCNs and linear models.
  2. Effective receptive field analysis proves that continuous large kernels eliminate dilation holes while maintaining efficiency.
  3. Decoupled pointwise FFNs prevent multivariate failure under high-frequency conditions.
- **Author Files Read**: `luodhhh/ModernTCN` (`ModernTCN.py`, `ModernTCN_Layer.py`, `RevIN.py`), `cure-lab/LTSF-Linear` (`DLinear.py`).
- **Academic Integrity**: All models and pipelines implemented independently from first principles in PyTorch.
- **AI Declaration**: AI tools used exclusively for testing assistance, syntax debugging, and document styling.
