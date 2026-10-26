# ETTh1 sample comparison report

- Runtime: local/non-Colab; selected device: cpu.
- Dataset: ETTh1; 17420 rows; variables: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT.
- Input / horizon: 336 / 96; features mode M; train-only standard scaling.
- Training windows sampled: 410/8209 (4.99%); validation windows: 2785; test windows: 2785.
- Sample seed: 42; epochs: 2; batch size: 32; learning rate: 0.001.
- Checkpoint selection: best validation MSE; project metrics: normalized MSE and MAE; RMSE derived from MSE.

## Results (ranked by test MSE)

| rank_by_test_mse | model | status | best_epoch | best_val_mse | test_mse | test_mae | test_rmse | training_time_seconds | test_inference_time_seconds | peak_gpu_memory_mb | device | checkpoint | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | DLinear | ok | 2 | 0.82117 | 0.47816 | 0.47846 | 0.69149 | 0.24409 | 0.06713 | None | cpu | D:\University Materials\year 4\Deep Learning\ModernTCN\colab_sample_outputs\20261026_171430\checkpoints\dlinear_best.pt | 64704 |
| 2 | ModernTCN | ok | 2 | 0.91206 | 0.50455 | 0.47414 | 0.71032 | 5.20162 | 0.87676 | None | cpu | D:\University Materials\year 4\Deep Learning\ModernTCN\colab_sample_outputs\20261026_171430\checkpoints\moderntcn_best.pt | 65262 |
| 3 | TCN | ok | 2 | 1.39471 | 1.09237 | 0.79598 | 1.04517 | 7.03446 | 1.26769 | None | cpu | D:\University Materials\year 4\Deep Learning\ModernTCN\colab_sample_outputs\20261026_171430\checkpoints\tcn_best.pt | 207072 |


## Pilot outcomes

```json
{
  "ModernTCN": {
    "device": "cpu",
    "input_shape": [
      32,
      336,
      7
    ],
    "target_shape": [
      32,
      96,
      7
    ],
    "pilot_train_mse": 0.7550206184387207,
    "val_batch_shape": [
      32,
      96,
      7
    ],
    "test_batch_shape": [
      32,
      96,
      7
    ],
    "ok": true
  },
  "DLinear": {
    "device": "cpu",
    "input_shape": [
      32,
      336,
      7
    ],
    "target_shape": [
      32,
      96,
      7
    ],
    "pilot_train_mse": 1.1378624439239502,
    "val_batch_shape": [
      32,
      96,
      7
    ],
    "test_batch_shape": [
      32,
      96,
      7
    ],
    "ok": true
  },
  "TCN": {
    "device": "cpu",
    "input_shape": [
      32,
      336,
      7
    ],
    "target_shape": [
      32,
      96,
      7
    ],
    "pilot_train_mse": 1.0652925968170166,
    "val_batch_shape": [
      32,
      96,
      7
    ],
    "test_batch_shape": [
      32,
      96,
      7
    ],
    "ok": true
  }
}
```

## Errors and limitations

No model errors recorded.
Peak GPU memory is unavailable on CPU. Inference time covers the full test loader.

Artifacts: `D:\University Materials\year 4\Deep Learning\ModernTCN\colab_sample_outputs\20261026_171430`.
