# ModernTCN Assignment

Deep Learning AIE332 - Assignment 1

Topic: Large kernels instead of dilation, based on ModernTCN (ICLR 2024).

## Research goals

- Compare the Lab 4 TCN baseline, DLinear, and a ModernTCN-style model.
- Evaluate ETTh1 and ETTm1.
- Use an input window of 336 and forecast horizons of 96 and 336.
- Report normalized MSE and MAE.
- Measure effective receptive fields using autograd.
- Compare at least three depthwise kernel sizes.
- Report mean and standard deviation over three seeds.
- Identify and explain a reproducible failure case.

## Rules

- Implement core model components in PyTorch.
- Do not copy the authors' model implementation.
- Select models using validation data.
- Evaluate the test set after model selection.
- Record author-code files inspected.
- Declare AI assistance in the report.

## Structure

- src/: data, models, training, and analysis.
- tests/: automated tests.
- configs/: experiment settings.
- notebooks/: final Colab notebook.
- research/: claim, paper notes, and inspected source files.
- results/: metrics, figures, and run manifests.
- checkpoints/: checkpoint placeholders.
- report/ and slides/: final deliverables.

## Colab workflow

1. Clone the repository.
2. Mount Google Drive for checkpoints and large artifacts.
3. Install the tested requirements.
4. Run pilot experiments before full training.
5. Use TRAIN=False to evaluate compatible saved checkpoints when needed.

Do not commit credentials, datasets, or large checkpoints to Git.
