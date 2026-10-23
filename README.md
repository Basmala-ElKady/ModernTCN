# ModernTCN Assignment

Deep Learning AIE332 - Assignment 1

Topic: Large kernels instead of dilation, based on ModernTCN (ICLR 2024).

## Research goals

- Compare a Lab 4 TCN baseline, DLinear, and a ModernTCN-style model.
- Evaluate ETTh1 and ETTm1.
- Use an input window of 336 and forecast horizons of 96 and 336.
- Report normalized MSE and MAE.
- Measure effective receptive fields using autograd.
- Compare at least three large depthwise kernel sizes.
- Report mean and standard deviation over three seeds.
- Identify and explain a reproducible failure case.

## Project rules

- Implement the core model components in PyTorch.
- Do not copy the authors' model implementation.
- Use validation data for model selection.
- Evaluate the test set only after model selection.
- Record the source files read from the authors' repository.
- Record AI assistance in the report.

## Repository layout

- src/ contains model, data, training, and analysis code.
- tests/ contains automated checks.
- configs/ contains experiment configurations.
- notebooks/ contains the final Colab notebook.
- results/ contains metrics, figures, and run manifests.
- research/ contains paper notes and the experimental claim.
- report/ and slides/ contain the final deliverables.

## Colab workflow

Clone or open the repository in Colab.
Mount Google Drive for checkpoints and large run artifacts.
Install the tested requirements.
Run a pilot before full training.
Use TRAIN=False to load compatible checkpoints for final evaluation when required.

Do not commit large datasets, credentials, or model checkpoints to ordinary Git history.
