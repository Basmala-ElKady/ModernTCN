# ModernTCN-style implementation notes

## What this implementation contains

- Learned, overlapping univariate patch embedding.
- Causal large-kernel depthwise temporal convolutions.
- An active small-kernel depthwise branch added to every large-kernel output.
- Pointwise (1x1) feed-forward layers with GELU and residual connections.
- Optional RevIN before the backbone and inverse RevIN after the forecast head.

## Deliberate differences from ModernTCN

This is a small teaching implementation rather than a reproduction of the
paper. It shares one backbone across variables after channel-independent patch
embedding, uses a single last-patch forecast head, and keeps causal left
padding. The original forecasting architecture uses a richer multi-stage
backbone and is not reproduced here. The small branch remains explicitly active
instead of being fused for deployment, as required by D4.

## RevIN

RevIN normalizes each sample and variable using only that input window, then
restores the original mean and scale on the prediction. It is useful when ETT
level/variance shift across windows. It can improve robustness, but it also
removes absolute scale from the backbone and its window-wide statistics mean
intermediate per-time-step features are not strictly causal. Therefore, run ERF
causality checks with `use_revin=False`; evaluate RevIN as an explicit
validation-selected setting, not an unreported default advantage.
