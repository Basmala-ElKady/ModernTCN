# ModernTCN RevIN ablation pilot

- Dataset: ETTh1; ETTh1 feature mode M; input 336; horizon 96.
- Preprocessing: existing chronological ETT 12/4/4 split; project standard scaler fitted on train rows only.
- Training sample: 410/8209 windows (4.99%), seed 42; same sorted indices for both settings.
- Training: batch size 32, Adam at 0.001, 2 maximum epochs, patience 2; best checkpoint selected using validation MSE.
- RevIN: existing per-window input mean/std, detached statistics, affine=True, eps=1e-5; output is inverse-normalized by the same window statistics. OFF sets model.revin=None. Global train-fitted standardization remains unchanged for both.
- Only intentional config difference: moderntcn.use_revin. Both runs use fresh model/optimizer, same initial shared weights, seed, shuffled minibatch order, validation and test windows.
- Runtime/device: local/non-Colab / cpu; GPU memory: not applicable on CPU.
- Runs with errors: none.

## Comparison

    model revin status  best_epoch  best_val_mse  test_mse  test_mae  test_rmse  training_time_seconds  test_inference_time_seconds peak_gpu_memory_mb device                                                                                                                                                   checkpoint  parameters
ModernTCN   OFF     ok           2      1.060704  0.645556  0.560504   0.803465              10.242913                     2.098981               None    cpu D:\University Materials\year 4\Deep Learning\ModernTCN\colab_sample_outputs\20261026_173100_moderntcn_revin_ablation\checkpoints\moderntcn_revin_off_best.pt       65248
ModernTCN    ON     ok           2      0.912056  0.504554  0.474142   0.710320              10.100025                     2.232711               None    cpu  D:\University Materials\year 4\Deep Learning\ModernTCN\colab_sample_outputs\20261026_173100_moderntcn_revin_ablation\checkpoints\moderntcn_revin_on_best.pt       65262

## RevIN ON improvement vs OFF

- MSE: +21.84%
- MAE: +15.41%
- RMSE: +11.59%

Positive values mean lower error with RevIN ON; negative values mean higher error. Interpret all three metrics together. This is a sample-based pilot, not a full-dataset result.

Plot: D:\University Materials\year 4\Deep Learning\ModernTCN\colab_sample_outputs\20261026_173100_moderntcn_revin_ablation\plots\test_metric_comparison.png
