import os
import numpy as np, pandas as pd, pytest
from src.data.preprocessing import SPLIT_ROWS, get_split_borders, split_and_normalize, load_ett

L = 336
COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

def make_csv(tmp_path, name, extra=500):
    n = sum(SPLIT_ROWS[name]) + extra            # real files have extra unused rows
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(n, 7)) + np.arange(n)[:, None] / n * 50   # drift: leakage would show
    df = pd.DataFrame(vals, columns=COLS)
    df.insert(0, "date", pd.date_range("2016-07-01", periods=n, freq="h"))
    p = tmp_path / f"{name}.csv"; df.to_csv(p, index=False); return str(p)

@pytest.mark.parametrize("name,sizes", [("ETTh1", (8640, 2880, 2880)), ("ETTm1", (34560, 11520, 11520))])
def test_split_sizes(name, sizes):
    assert SPLIT_ROWS[name] == sizes
    b = get_split_borders(name, L)
    assert b["train"] == (0, sizes[0])
    assert b["val"][1] - b["val"][0] == sizes[1] + L
    assert b["test"][1] - b["test"][0] == sizes[2] + L

@pytest.mark.parametrize("name", ["ETTh1", "ETTm1"])
def test_chronological_no_leakage(name):
    b = get_split_borders(name, L)
    # target region of each split starts exactly where the previous one ended
    assert b["val"][0] + L == b["train"][1]
    assert b["test"][0] + L == b["val"][1]

def test_scaler_fit_on_train_only(tmp_path):
    data, cols = load_ett(make_csv(tmp_path, "ETTh1"))
    assert cols == COLS
    splits, sc = split_and_normalize(data, "ETTh1", L)
    tr = data[:8640]
    np.testing.assert_allclose(sc.mean, tr.mean(0), rtol=1e-5)
    np.testing.assert_allclose(sc.std, tr.std(0), rtol=1e-4)
    assert np.allclose(splits["train"].mean(0), 0, atol=1e-4)
    assert np.allclose(splits["train"].std(0), 1, atol=1e-3)
    assert splits["test"].mean() > 1.0           # drift => not re-normalised on test

def test_too_short_raises(tmp_path):
    with pytest.raises(ValueError):
        split_and_normalize(np.zeros((1000, 7), np.float32), "ETTh1", L)

# ---- torch-dependent tests -------------------------------------------------
@pytest.mark.parametrize("name,H", [("ETTh1", 96), ("ETTh1", 336), ("ETTm1", 96), ("ETTm1", 336)])
def test_windows(tmp_path, name, H):
    torch = pytest.importorskip("torch")
    from src.data.dataset import build_datasets
    sets, _ = build_datasets(make_csv(tmp_path, name), name, L, H)
    n_tr, n_va, n_te = SPLIT_ROWS[name]
    assert len(sets["train"]) == n_tr - L - H + 1
    assert len(sets["val"]) == n_va - H + 1      # lookback prefix => no windows lost
    assert len(sets["test"]) == n_te - H + 1
    x, y = sets["train"][5]
    assert x.shape == (L, 7) and y.shape == (H, 7) and x.dtype == torch.float32
    assert torch.equal(torch.cat([x, y]), sets["train"].data[5:5 + L + H])  # y follows x

def test_dataloader_batches(tmp_path):
    pytest.importorskip("torch")
    from src.data.dataset import build_dataloaders
    make_csv(tmp_path, "ETTh1")
    cfg = dict(dataset="ETTh1", data_dir=str(tmp_path), input_length=L, forecast_horizon=96, batch_size=16)
    x, y = next(iter(build_dataloaders(cfg)["train"]))
    assert x.shape == (16, L, 7) and y.shape == (16, 96, 7)

# ---- real data (skipped if CSVs are not in data/) --------------------------
@pytest.mark.parametrize("name,rows", [("ETTh1", 17420), ("ETTm1", 69680)])
def test_real_data(name, rows):
    path = f"data/{name}.csv"
    if not os.path.exists(path):
        pytest.skip("real CSV not present")
    data, cols = load_ett(path)
    assert cols == COLS and len(data) == rows
    splits, sc = split_and_normalize(data, name, L)
    n_tr, n_va, n_te = SPLIT_ROWS[name]
    assert [len(splits[k]) for k in ("train", "val", "test")] == [n_tr, n_va + L, n_te + L]
    assert np.abs(splits["train"].mean(0)).max() < 1e-3
    assert np.abs(splits["train"].std(0) - 1).max() < 1e-3
    assert np.isfinite(splits["test"]).all()
