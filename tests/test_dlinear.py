#  Unit tests for DLinear. Synthetic tensors only; no dataset is required.
import pytest
import torch
from src.models.dlinear import DLinear, MovingAverageDecomposition, count_parameters


L = 336

@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# shapes
@pytest.mark.parametrize("horizon", [96, 336])
@pytest.mark.parametrize("features", [1, 7, 21])
@pytest.mark.parametrize("individual", [False, True])
def test_output_shape(horizon, features, individual):
    model = DLinear(L, horizon, individual=individual, num_features=features)
    x = torch.randn(4, L, features)
    assert model(x).shape == (4, horizon, features)

def test_horizon_96():
    assert DLinear(L, 96)(torch.randn(2, L, 7)).shape == (2, 96, 7)

def test_horizon_336():
    assert DLinear(L, 336)(torch.randn(2, L, 7)).shape == (2, 336, 7)

@pytest.mark.parametrize("batch", [0, 1, 2, 64])
def test_batch_size_edge_cases(batch):
    out = DLinear(L, 96)(torch.randn(batch, L, 7))
    assert out.shape == (batch, 96, 7)

def test_single_and_multi_feature_same_model():
    model = DLinear(L, 96)  # shared weights accept any C
    for c in (1, 3, 862):
        assert model(torch.randn(2, L, c)).shape == (2, 96, c)

@pytest.mark.parametrize("shape", [(L, 7), (2, 100, 7), (2, L, 7, 1)])
def test_invalid_input_rejected(shape):
    with pytest.raises(ValueError):
        DLinear(L, 96)(torch.randn(*shape))


def test_individual_wrong_feature_count_rejected():
    with pytest.raises(ValueError):
        DLinear(L, 96, individual=True, num_features=7)(torch.randn(2, L, 5))

def test_individual_requires_num_features():
    with pytest.raises(ValueError):
        DLinear(L, 96, individual=True)

@pytest.mark.parametrize("k", [0, 2, 24, -3])
def test_invalid_kernel_rejected(k):
    with pytest.raises(ValueError):
        MovingAverageDecomposition(k)


# decomposition
@pytest.mark.parametrize("kernel", [1, 5, 25, 51])
def test_decomposition_reconstruction(kernel):
    x = torch.randn(3, L, 5)
    seasonal, trend = MovingAverageDecomposition(kernel)(x)
    assert seasonal.shape == trend.shape == x.shape
    torch.testing.assert_close(trend + seasonal, x)

def test_decomposition_constant_series():
    x = torch.full((2, L, 3), 4.2)
    seasonal, trend = MovingAverageDecomposition(25)(x)
    torch.testing.assert_close(trend, x)  # replicate padding keeps edges unbiased
    assert seasonal.abs().max() < 1e-5

def test_decomposition_matches_manual_moving_average():
    k = 5
    x = torch.randn(1, 20, 1)
    _, trend = MovingAverageDecomposition(k)(x)
    p = (k - 1) // 2
    padded = torch.cat([x[:, :1].repeat(1, p, 1), x, x[:, -1:].repeat(1, p, 1)], dim=1)
    manual = torch.stack([padded[0, i:i + k, 0].mean() for i in range(20)]).view(1, 20, 1)
    torch.testing.assert_close(trend, manual)

def test_decomposition_has_no_parameters():
    assert count_parameters(MovingAverageDecomposition(25)) == 0

def test_trend_is_smoother_than_input():
    x = torch.randn(1, L, 1)
    _, trend = MovingAverageDecomposition(25)(x)
    assert trend.diff(dim=1).abs().mean() < x.diff(dim=1).abs().mean()


# gradients
@pytest.mark.parametrize("individual", [False, True])
def test_gradient_propagation(individual):
    model = DLinear(L, 96, individual=individual, num_features=7)
    x = torch.randn(4, L, 7, requires_grad=True)
    model(x).pow(2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0

def test_one_sgd_step_reduces_loss():
    model = DLinear(L, 96)
    x, y = torch.randn(8, L, 7), torch.randn(8, 96, 7)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    before = torch.nn.functional.mse_loss(model(x), y)
    opt.zero_grad(); before.backward(); opt.step()
    after = torch.nn.functional.mse_loss(model(x), y)
    assert after < before


# stability
def test_forward_is_finite_and_deterministic_in_eval():
    model = DLinear(L, 336).eval()
    x = torch.randn(4, L, 7)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b)

@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e4])
def test_forward_stable_across_input_scales(scale):
    out = DLinear(L, 96)(torch.randn(2, L, 7) * scale)
    assert torch.isfinite(out).all()

def test_batch_independence():
    model = DLinear(L, 96).eval()
    x = torch.randn(5, L, 7)
    with torch.no_grad():
        torch.testing.assert_close(model(x)[2:3], model(x[2:3]), rtol=1e-5, atol=1e-5)

# Changing one variable must not alter the forecast of the others.
def test_channel_independence_shared_weights():
    model = DLinear(L, 96).eval()
    x = torch.randn(2, L, 4)
    x2 = x.clone(); x2[..., 0] += 5.0
    with torch.no_grad():
        a, b = model(x), model(x2)
    torch.testing.assert_close(a[..., 1:], b[..., 1:])
    assert not torch.allclose(a[..., 0], b[..., 0])

# DLinear is affine: f(ax) - f(0) = a (f(x) - f(0)).
def test_linearity_without_bias():
    model = DLinear(L, 96).eval()
    x = torch.randn(2, L, 3)
    with torch.no_grad():
        f0 = model(torch.zeros_like(x))
        torch.testing.assert_close(model(2 * x) - f0, 2 * (model(x) - f0), rtol=1e-4, atol=1e-4)


# parameter count
@pytest.mark.parametrize("horizon,expected", [(96, 64_704), (336, 226_464)])
def test_parameter_count_shared(horizon, expected):
    model = DLinear(L, horizon)
    assert model.num_parameters() == expected == count_parameters(model)
    assert model.num_parameters() == DLinear.expected_num_parameters(L, horizon)

@pytest.mark.parametrize("features", [1, 7])
def test_parameter_count_individual(features):
    model = DLinear(L, 96, individual=True, num_features=features)
    assert model.num_parameters() == 64_704 * features
    assert model.num_parameters() == DLinear.expected_num_parameters(L, 96, True, features)

def test_shared_parameter_count_independent_of_features():
    model = DLinear(L, 96)
    n = model.num_parameters()
    model(torch.randn(1, L, 50))
    assert model.num_parameters() == n


# config / interface
def test_from_config_and_summary():
    cfg = {"forecast_horizon": 336}
    model = DLinear.from_config(cfg)
    x = torch.randn(3, L, 7)
    assert model(x).shape == (x.shape[0], cfg["forecast_horizon"], x.shape[2])
    s = model.config_summary()
    assert s["forecast_horizon"] == 336 and s["num_parameters"] == 226_464

def test_from_config_missing_horizon():
    with pytest.raises(KeyError):
        DLinear.from_config({})

def test_registry_style_usage():
    registry = {"dlinear": DLinear}
    model = registry["dlinear"].from_config({"forecast_horizon": 96, "num_features": 7})
    assert model(torch.randn(2, L, 7)).shape == (2, 96, 7)