import pytest
torch = pytest.importorskip("torch")
from src.models.tcn import build_tcn, receptive_field

CFG = {"num_variables": 7, "input_length": 336, "forecast_horizon": 96,
       "tcn": {"hidden_channels": 32}}

def test_output_shape():
    for H in (96, 336):
        model = build_tcn({**CFG, "forecast_horizon": H})
        assert model(torch.randn(4, 336, 7)).shape == (4, H, 7)

def test_receptive_field_covers_input():
    model = build_tcn(CFG)
    assert model.receptive_field >= 336
    assert receptive_field(3, model.num_levels - 1) < 336  # minimal depth

def test_gradients_flow():
    model = build_tcn(CFG)
    model(torch.randn(2, 336, 7)).pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_causality():
    model = build_tcn(CFG).eval()
    x = torch.randn(1, 336, 7)
    x2 = x.clone(); x2[:, 200:] += 5.0       # change the future only
    with torch.no_grad():
        a, b = model.forward_features(x), model.forward_features(x2)
    assert torch.allclose(a[:, :, :200], b[:, :, :200], atol=1e-6)
    assert not torch.allclose(a[:, :, 200:], b[:, :, 200:])

def test_small_batch_overfit():
    torch.manual_seed(0)
    model = build_tcn(CFG)
    x, y = torch.randn(8, 336, 7), torch.randn(8, 96, 7)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    first = None
    for _ in range(30):
        loss = torch.nn.functional.mse_loss(model(x), y)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first
