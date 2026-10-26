# ModernTCN: Author Files Read & Integrity Record

Deep Learning (AIE332) — Lab 4 Assignment: Topic 6 (ModernTCN, ICLR 2024)

---

## 1. Requirement & Compliance Statement

In accordance with **Rule 1** of the Lab 4 Assignment specification:
> *"Write the core of the model yourself in PyTorch. Libraries are fine for data loading, optimisers, metrics and plotting... You may read the authors' code but not copy it; list the files you read in your report."*

Every model component in this repository (`src/models/moderntcn.py`, `src/models/tcn.py`, `src/models/dlinear.py`, and `src/models/revin.py`), as well as the training pipelines, data preprocessors, effective receptive field analyzers, and failure analyzers, was designed and written independently from scratch in standard PyTorch. No author code was copied, cloned, or pasted.

---

## 2. Inspected Repositories and Author Code Files

### A. ModernTCN Official Repository
- **Paper**: Luo, D., & Wang, X. (2024). *ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis*. International Conference on Learning Representations (ICLR 2024).
- **Repository**: [https://github.com/luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN)
- **Author code files inspected**:
  1. `models/ModernTCN.py`:
     - *Purpose of inspection*: Reviewed high-level architecture organization, token/patch dimensions, and how patch projection is shaped before feeding into convolutional blocks.
     - *Key observation*: The official codebase implements a multi-stage downsampling backbone (similar to ConvNeXt stages). For our lab setup and parameter parity, we designed a unified single-stage backbone that explicitly preserves causal left-padding and keeps the small-kernel branch active.
  2. `layers/ModernTCN_Layer.py`:
     - *Purpose of inspection*: Examined how depthwise large-kernel temporal convolution is separated from pointwise inter-variable and intra-variable feed-forward operations.
     - *Key observation*: Noted the split between depthwise temporal mixing and pointwise channel/variable mixing. We implemented this independently using standard `nn.Conv1d` layers and residual connections with GELU activations.
  3. `layers/RevIN.py`:
     - *Purpose of inspection*: Checked the Reversible Instance Normalization implementation (Kim et al., ICLR 2022) to verify standard statistical normalization across time and learnable affine scaling (`gamma`, `beta`).
     - *Key observation*: RevIN computes mean and variance along the sequence dimension ($L$) per batch and channel, normalizes before the backbone, and reverses the transformation on predicted horizons.
  4. `exp/exp_main.py` & `data_provider/data_loader.py`:
     - *Purpose of inspection*: Confirmed chronological train/val/test split boundaries for Informer/Autoformer benchmark conventions on ETTh1 and ETTm1 (12 months train, 4 months val, 4 months test).

### B. DLinear Reference Code
- **Paper**: Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). *Are Transformers Effective for Time Series Forecasting?* (AAAI 2023).
- **Repository**: [https://github.com/cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
- **Author code files inspected**:
  1. `models/DLinear.py`:
     - *Purpose of inspection*: Verified the moving average series decomposition kernel and how trend and seasonal components are processed through separate single-layer linear projections.
     - *Key observation*: DLinear decomposes the input window into trend (moving average) and seasonal (residual), passes each through an `nn.Linear(input_length, forecast_horizon)`, and sums the predictions. Implemented cleanly in `src/models/dlinear.py`.

### C. Lab 4 Baseline TCN
- **Paper**: Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271.
- **Reference**: Course Lab 4 materials (Part C & D).
- **Files inspected**:
  - Lab 4 TCN reference implementation with dilated 1D convolutions, causal padding $(K - 1) \times d$, dilation factor doubling ($1, 2, 4, 8, \dots$), residual connections, and output projection.

---

## 3. Deliberate Structural Design Decisions

Our implementation adheres strictly to the assignment's research guidelines:
1. **Clean-Room Implementation**: All layers, parameterizations, and tensor shape transformations were derived mathematically and written independently.
2. **Explicit Dual-Scale Branching**: Rather than fusing small-kernel branches into the large kernel at deployment, our ModernTCN implementation maintains the active small-kernel branch ($K_{small}=5$) in parallel with the large depthwise kernel ($K_{large} \in \{15, 31, 51\}$).
3. **Causal Integrity**: We utilize causal left padding on all temporal convolutions so the model strictly predicts forward in time without future information leakage.
4. **Fair Parameter Parity**: ModernTCN, TCN, and DLinear are matched in model capacity and hyperparameter tuning to ensure rigorous benchmark comparison.
