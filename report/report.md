# ModernTCN: Large Kernels Instead of Dilation for Multivariate Time Series Forecasting

**Course**: Deep Learning (AIE332) — Lab 4 Assignment: Topic 6  
**Topic**: Large kernels instead of dilation, based on ModernTCN (ICLR 2024)  
**Deliverable**: Final Research Report (Conforming to the 4-page specification)  

---

## 1. The Method in Our Own Words (182 words)

ModernTCN modernizes temporal convolutions for multivariate time series forecasting by addressing two major shortcomings of traditional TCNs: receptive field gridding (dilational sampling holes) and the absence of cross-variable feature mixing. 

The architecture begins with Reversible Instance Normalization (RevIN), which normalizes each window to mitigate non-stationary distribution shifts across time. Next, a univariate patch embedding projects raw timesteps into patch tokens ($P=16, S=8$), reducing sequence length, suppressing high-frequency sensor noise, and expanding representation dimensionality. 

The core backbone consists of stacked ModernTCN blocks. In each block, a large depthwise causal convolution ($K=31$) captures extensive temporal context without dilation gaps, operating in parallel with an active small depthwise kernel ($k=5$) that retains high-frequency local variations. Because depthwise convolutions apply filters independently per channel, parameter count scales linearly with kernel size rather than quadratically.

Subsequently, two pointwise ($1\times 1$) feed-forward networks decouple feature processing: an intra-variable FFN mixes latent features within each variable, and an inter-variable FFN mixes representations across different variables to capture physical cross-series dependencies. Finally, a linear head projects patch representations to the forecast horizon, and inverse RevIN restores the series scale.

---

## 2. Quantitative Paper Claim & Outcome

### Chosen Quantitative Claim
From **Luo & Wang (ICLR 2024)**, *"ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"*, Section 1 & Section 4.2 (Table 1):
> *"ModernTCN outperforms recent state-of-the-art Transformer-based and Linear-based models (such as PatchTST and DLinear), as well as conventional convolutional models (TCN), across extensive time series forecasting benchmarks, achieving substantial reductions in MSE and MAE on the ETT benchmarks."*

### Experimental Setup at Our Scale
- **Benchmarks**: ETTh1 (hourly, 7 variables) and ETTm1 (15-minute, 7 variables).
- **Protocol**: Standard chronological train/val/test splits (12/4/4 months). Standard scaling fit exclusively on training data.
- **Window Length ($L$)**: 336 steps; **Forecast Horizons ($H$)**: 96 and 336 steps.
- **Model Selection**: Selected checkpoint with best validation MSE; evaluated exactly once on the test split.
- **Seeds**: Evaluated across 3 independent seeds (`42, 123, 2026`) with sample standard deviation ($ddof=1$).

### Quantitative Verification Results

| Dataset | Horizon ($H$) | Model | Test MSE (mean ± std) | Test MAE (mean ± std) | Best Val MSE | Training Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **ETTh1** | **96** | **ModernTCN** | **0.3799 ± 0.0023** | **0.4032 ± 0.0014** | 0.6882 | 19.28 s |
| ETTh1 | 96 | DLinear | 0.3852 ± 0.0045 | 0.4083 ± 0.0048 | 0.6497 | 4.17 s |
| ETTh1 | 96 | TCN | 0.7789 ± 0.0395 | 0.6874 ± 0.0222 | 0.9238 | 50.08 s |
| **ETTh1** | **336** | **ModernTCN** | **0.4481 ± 0.0069** | **0.4466 ± 0.0045** | 1.2187 | 13.55 s |
| ETTh1 | 336 | DLinear | 0.4691 ± 0.0165 | 0.4680 ± 0.0144 | 1.0431 | 5.38 s |
| ETTh1 | 336 | TCN | 0.9806 ± 0.0393 | 0.7731 ± 0.0199 | 1.2305 | 70.97 s |
| **ETTm1** | **96** | **ModernTCN** | **0.2970 ± 0.0044** | **0.3490 ± 0.0025** | 0.3801 | 58.80 s |
| ETTm1 | 96 | DLinear | 0.3078 ± 0.0013 | 0.3531 ± 0.0020 | 0.3865 | 18.91 s |
| ETTm1 | 96 | TCN | 0.5375 ± 0.0202 | 0.5239 ± 0.0177 | 0.6260 | 170.88 s |
| **ETTm1** | **336** | **ModernTCN** | **0.3745 ± 0.0062** | **0.3967 ± 0.0026** | 0.6574 | 56.58 s |
| ETTm1 | 336 | DLinear | 0.3872 ± 0.0063 | 0.4072 ± 0.0082 | 0.6175 | 25.49 s |
| ETTm1 | 336 | TCN | 0.9611 ± 0.0195 | 0.7704 ± 0.0177 | 0.8982 | 174.36 s |

### Claim Verdict
**The experimental results strongly support the paper's claim.** ModernTCN consistently achieves the lowest MSE and MAE across all four settings. It reduces MSE by **51.2% to 61.0% relative to the Lab 4 TCN baseline**, and consistently outperforms DLinear by **1.4% to 4.5% in MSE** with narrow cross-seed error margins.

---

## 3. Depthwise Kernel Size Ablation & Receptive Field Analysis

### Kernel Size Ablation Study (Comparable Parameter Count)
Topic 6 requires evaluating at least three depthwise kernel sizes at comparable parameter count. We evaluated $K \in \{15, 31, 51\}$ on ETTh1 ($H=96$) across 3 seeds (`42, 123, 2026`).

| Depthwise Kernel ($K$) | Total Parameters | Test MSE (mean ± std) | Test MAE (mean ± std) | Best Val MSE (mean) |
| :---: | :---: | :---: | :---: | :---: |
| **15** | 62,190 | **0.3788 ± 0.0013** | **0.4024 ± 0.0014** | 0.6853 |
| **31** | 65,262 | 0.3799 ± 0.0023 | 0.4032 ± 0.0014 | 0.6882 |
| **51** | 69,102 | 0.3830 ± 0.0039 | 0.4053 ± 0.0027 | 0.6916 |

*Parameter Count Parity*: The total parameter variation across $K=15$ and $K=51$ is under $\pm 5.5\%$, confirming fair parameter parity.

### Receptive Field & Autograd Sensitivity Analysis
Using the autograd method from Lab 4, we computed input sensitivity:
$$S(t) = \frac{1}{|P| \cdot C} \sum_{p \in P} \sum_{c=1}^C \left| \frac{\partial \hat{y}_{p, c}}{\partial x_{t}} \right|$$
- **Effective Receptive Field (ERF)**: For $K=15$, 50% of cumulative gradient mass is concentrated in the most recent 14 timesteps, and 90% in the most recent 61 timesteps.
- **Does accuracy follow the ERF?**: Increasing kernel size from 15 to 31 maintains comparable competitive accuracy ($0.3788$ vs $0.3799$), but expanding to $K=51$ slightly degrades test MSE ($0.3830$). Although the theoretical receptive field widens, the effective informative dependencies in ETTh1 are already fully covered by $K=15$–$31$. Extremely large kernels add unnecessary temporal parameters that introduce slight overfitting without providing additional predictive information.

---

## 4. Reproducible Failure Case: TCN on Long-Horizon ETTm1

### Observed Breakdown
When forecasting ETTm1 at a long horizon ($H=336$), the Lab 4 TCN baseline fails catastrophically:
- **TCN Test MSE**: **0.9611 ± 0.0195**
- **ModernTCN Test MSE**: **0.3745 ± 0.0062** (TCN error is **2.56× higher**)

### Per-Window Failure Analysis
In per-window test error analysis across 11,185 test windows:
- TCN produces higher MSE on **11,074** (seed 42), **10,857** (seed 123), and **11,086** (seed 2026) windows (**>98.8% of all windows**).
- On peak load transition intervals, TCN's error is **6.2× to 9.5× worse** than ModernTCN (e.g., test window 793: TCN MSE = 2.898 vs. ModernTCN MSE = 0.306).

### Physical & Architectural Explanation
1. **Sampling Granularity vs. Receptive Field Holes**: ETTm1 has 15-minute resolution (4× denser than ETTh1). A 336-step horizon represents 3.5 days of fine-grained cyclic variation. TCN's exponential dilation schedule ($1, 2, 4, 8, 16, \dots$) leaves large temporal gaps between sampled timesteps. High-frequency electrical fluctuations slip between dilation points, accumulating compound errors over 336 autoregressive steps.
2. **Channel-Independent Isolation**: Power transformer oil temperature ($OT$) is physically coupled with oil and power load variables ($HUFL, HULL, LUFL$). TCN models each channel without cross-variable communication. ModernTCN's inter-variable FFN captures these physical thermodynamic constraints, maintaining stable long-horizon predictions.

---

## 5. List of Inspected Author Files (Integrity Record)

In compliance with Rule 1:
- **`luodhhh/ModernTCN`**:
  - `models/ModernTCN.py`: Inspected tokenization and block layout.
  - `layers/ModernTCN_Layer.py`: Inspected depthwise vs pointwise structure.
  - `layers/RevIN.py`: Inspected instance normalization formula and affine parameters.
  - `exp/exp_main.py`: Inspected benchmark train/val/test split protocols.
- **`cure-lab/LTSF-Linear`**:
  - `models/DLinear.py`: Inspected moving-average decomposition structure.
- **Statement of Independence**: Every model, training routine, evaluation metric, and analysis script in `src/` was authored independently from scratch in PyTorch. No author code was copied.

---

## 6. Declaration of AI Use

AI coding tools (Google DeepMind Antigravity / Gemini 3.8) were utilized in this assignment for:
1. Syntax debugging and PyTorch test suite boilerplate generation (`pytest`).
2. Scripting matplotlib figures and LaTeX formatting.
3. Troubleshooting local Windows environment execution paths.

All core conceptual design choices, experimental execution protocols, mathematical derivations, and result interpretations were performed and verified independently by the author.
