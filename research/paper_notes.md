# Paper Notes: ModernTCN (ICLR 2024)

**Paper**: *ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis*  
**Authors**: Donghao Luo, Xue Wang (Tsinghua University)  
**Venue**: International Conference on Learning Representations (ICLR 2024)  
**Topic**: Large kernels instead of dilation for multivariate time series forecasting  

---

## 1. Executive Summary & Context

Over recent years, deep time series forecasting progressed through three major phases:
1. **Classical Temporal Convolutions (TCN, 2018)**: Replaced RNNs with 1D causal convolutions and exponential dilation to capture sequence history in parallel.
2. **Transformer Epoch (Informer, Autoformer, FEDformer, PatchTST, 2021–2023)**: Leveraged self-attention across time tokens. However, DLinear (AAAI 2023) challenged this trend by demonstrating that a simple moving-average + linear layer often outperformed complex Transformers.
3. **Modern Pure Convolution (ModernTCN, 2024)**: Bridges the gap by applying modern Computer Vision convolutional design principles (from ConvNeXt and RepLKNet) to temporal modeling. Instead of relying on exponential dilation or heavy quadratic attention, ModernTCN uses **large-kernel depthwise convolutions** combined with **patch embeddings** and **decoupled pointwise feed-forward networks**.

---

## 2. Key Limitations of Conventional TCNs

The Lab 4 baseline TCN (Bai et al., 2018) suffers from three fundamental structural bottlenecks:
1. **The Dilation Paradox (Receptive Field Gridding / Holes)**:
   While exponential dilation ($d = 1, 2, 4, 8, \dots$) geometrically expands the *theoretical* receptive field:
   $$TRF = 1 + \sum_{l=0}^{L-1} (K_l - 1) \cdot d_l$$
   it samples the input history sparsely. Intermediate points are skipped, creating "holes" in temporal perception that degrade performance on high-frequency signals.
2. **Effective Receptive Field (ERF) Decay**:
   Theoretical RF assumes all connections propagate equal influence. In practice, gradient backpropagation (Luo et al., NeurIPS 2016) reveals that the effective receptive field follows a Gaussian distribution that shrinks rapidly away from the prediction point, making classic TCN behave like a much smaller kernel model.
3. **Absence of Inter-Variable Modeling**:
   Standard TCN models multivariate time series either by flattening all channels into 1D (causing parameter explosion) or treating each channel independently (failing to capture cross-variable physical dynamics like voltage vs. temperature in power transformers).

---

## 3. The ModernTCN Architecture

ModernTCN addresses each of these bottlenecks systematically:

```
Input Window [B, L, C]
        │
     [RevIN] (Instance Normalization across L)
        │
 [Patch Embedding] (Univariate 1D Conv, patch size P=16, stride S=8)
        │ ──> Token shape: [B * C, D, N_patches]
┌────────────────────────────────────────────────────────┐
│ ModernTCN Block (repeated for depth = 3):              │
│                                                        │
│  1. Depthwise Temporal Convolutions (Causal):          │
│     Output = DWConv_Large(x, K=31) + DWConv_Small(x, 5)│
│     Residual add: Output = Output + x                  │
│                                                        │
│  2. Pointwise Intra-Variable FFN (1x1 Conv):           │
│     Mixes feature channels within each variable        │
│     Linear(D -> 2D) -> GELU -> Linear(2D -> D)         │
│                                                        │
│  3. Pointwise Inter-Variable FFN (1x1 Conv):           │
│     Mixes features across all C variables              │
│     Linear(C -> 2C) -> GELU -> Linear(2C -> C)         │
└────────────────────────────────────────────────────────┘
        │
 [Linear Projection Head] (Maps last patch representation to Horizon H)
        │
 [Inverse RevIN] (Restores original mean and standard deviation)
        │
Output Forecast [B, H, C]
```

### Key Architectural Pillars:
1. **Patch Embedding**:
   Aggregates local continuous timesteps into patch tokens ($P=16, S=8$). Patching serves three purposes:
   - Suppresses high-frequency sensor noise via local aggregation.
   - Reduces the sequence length from $L=336$ to $N=41$ patch tokens, drastically lowering computational cost.
   - Endows each token with semantic temporal context before any convolution begins.

2. **Large Depthwise Temporal Convolutions**:
   - Uses large kernels ($K \in \{15, 31, 51\}$) directly on patch tokens without dilation.
   - Because depthwise convolutions apply filters independently to each feature channel, parameter count scales as $\mathcal{O}(D \cdot K)$ rather than $\mathcal{O}(D^2 \cdot K)$. This enables large receptive fields with negligible parameter overhead (<10% parameter difference between $K=15$ and $K=51$).
   - Avoids dilation gridding: every single patch in the kernel window is sampled continuously.

3. **Active Dual-Scale Branching**:
   - In parallel with the large depthwise kernel ($K=31$), a small depthwise kernel ($k=5$) operates simultaneously.
   - The large kernel captures long-range seasonal trends, while the small kernel captures sharp local transitions.

4. **Decoupled Pointwise FFNs (Cross-Time & Cross-Variable)**:
   - ModernTCN decouples temporal mixing from spatial/variable mixing.
   - Intra-variable FFN mixes latent representation dimensions within each variable.
   - Inter-variable FFN projects across variables to capture cross-channel correlation (e.g., oil temperature vs. electrical load).

5. **Reversible Instance Normalization (RevIN)**:
   - Normalizes each sample along the time dimension using its current window mean and variance:
     $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
   - The forecast head output is denormalized before calculating the final loss:
     $$\hat{y} = \frac{y_{pred} - \beta}{\gamma} \cdot \sqrt{\sigma^2 + \epsilon} + \mu$$
   - Effectively removes distribution shift and trend drift across non-stationary windows.

---

## 4. Receptive Field Analysis: Theory vs Autograd

In our experiments:
- **Theoretical Receptive Field (TRF)** for ModernTCN with patch size 16, stride 8, depth 3, and kernel 31 easily spans the full 336-step input window.
- **Effective Receptive Field (ERF)** measured via autograd sensitivity:
  $$\text{Sensitivity}(t) = \frac{1}{|P| \cdot C} \sum_{p \in P} \sum_{c=1}^C \left| \frac{\partial \hat{y}_{p, c}}{\partial x_{t}} \right|$$
  demonstrates smooth, non-zero gradient propagation across the entire 336-timestep history, without the periodic gaps and vanishing sensitivity observed in dilated TCNs.

---

## 5. Summary of Baseline Comparison

| Criterion | Lab 4 Baseline TCN | DLinear | ModernTCN |
| :--- | :--- | :--- | :--- |
| **Temporal Operator** | Dilated Causal Conv ($k=3, d=2^l$) | Moving Average + Linear | Large DWConv ($K=31$) + Small DWConv ($k=5$) |
| **Token Representation** | Raw Timesteps | Decomposed Series | Patch Tokens ($P=16, S=8$) |
| **Inter-Variable Mixing** | None (Channel-Independent) | Channel-Independent | Pointwise Inter-Variable FFN |
| **Receptive Field Gridding** | Yes (Severe dilation holes) | No (Direct dense projection) | No (Continuous large kernel) |
| **Handling Distribution Shift** | None / Manual Scaling | Trend Linear Decomposition | Reversible Instance Norm (RevIN) |
| **Average Test MSE (ETTh1, 96)**| 0.7789 | 0.3852 | **0.3799** |
| **Average Test MSE (ETTm1, 336)**| 0.9611 | 0.3872 | **0.3745** |
