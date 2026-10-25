# ModernTCN RevIN ablation report

## Status

Implementation and unit-test validation are complete. No ETT CSV files are in
this repository workspace, so no training/test metrics are reported here.

## Protocol

Run matched ModernTCN experiments for ETTh1 horizons 96 and 336, with seeds
`0, 1, 2`, once with RevIN enabled and once disabled. All architecture and
training fields are identical except `moderntcn.use_revin`. The best epoch is
chosen by validation MSE; that saved checkpoint is evaluated once on test.

```powershell
python -m src.analysis.revin_ablation --data-dir data
```

Artifacts are written under `results/metrics/revin_ablation/`:

- `revin_ablation_raw.csv`: one test MSE/MAE result per seed/run.
- `revin_ablation_summary.csv`: mean and sample standard deviation over seeds.
- `revin_ablation_manifest.json`: exact matrix, base configuration, and ddof
  values.

The summary explicitly uses `np.std(values, ddof=1)`. The existing training
data `StandardScaler` deliberately remains `np.std(..., ddof=0)` and is fit on
training rows only. A positive percentage in the summary means RevIN reduced
that metric relative to the no-RevIN baseline.

## Inspected and modified files

Inspected: ModernTCN, RevIN, TCN, preprocessing, dataset/dataloader, training,
evaluation, seed helper, config, and current tests. Modified `seed.py`, added
the ablation runner and tests, and added this report. No split, scaler, or
unrelated model was changed.

## Limitation / remaining work

Put `ETTh1.csv` (and any desired ETTm1 CSV) in the chosen data directory, run
the command above, and paste the generated raw and summary tables here. Do not
claim an improvement until those actual runs complete.
