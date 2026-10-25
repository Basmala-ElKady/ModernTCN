import pytest

torch = pytest.importorskip("torch")

from src.models.moderntcn import ModernTCN, build_moderntcn
from src.models.revin import RevIN
from src.utils.seed import set_seed


CFG = {
    "num_variables": 7, "input_length": 48, "forecast_horizon": 12,
    "moderntcn": {"embed_dim": 12, "depth": 2, "kernel_size": 7,
                   "small_kernel_size": 3, "patch_size": 8, "patch_stride": 4,
                   "dropout": 0.0},
}


def test_output_shape_for_required_horizons():
    for horizon in (96, 336):
        model = build_moderntcn({**CFG, "forecast_horizon": horizon})
        assert model(torch.randn(2, 48, 7)).shape == (2, horizon, 7)


def test_gradients_reach_all_trainable_parameters():
    model = build_moderntcn(CFG)
    model(torch.randn(2, 48, 7)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_small_kernel_branch_is_enabled_and_counted():
    model = build_moderntcn(CFG)
    assert model.blocks[0].token_mixer.small.weight.requires_grad
    assert model.num_parameters() == sum(p.numel() for p in model.parameters() if p.requires_grad)
    no_small = ModernTCN(7, 48, 12, embed_dim=12, depth=2, kernel_size=7,
                         small_kernel_size=1, patch_size=8, patch_stride=4)
    assert model.num_parameters() > no_small.num_parameters()


def test_patch_features_are_causal_without_window_normalization():
    model = ModernTCN(3, 48, 12, embed_dim=8, depth=1, kernel_size=7,
                      small_kernel_size=3, patch_size=8, patch_stride=4,
                      dropout=0.0, use_revin=False).eval()
    x = torch.randn(1, 48, 3)
    changed = x.clone(); changed[:, 32:] += 10
    with torch.no_grad():
        a, b = model.forward_features(x), model.forward_features(changed)
    # Patch 6 ends at index 31; it cannot depend on observations from 32 onward.
    assert torch.allclose(a[..., :6], b[..., :6], atol=1e-6)


def test_revin_normalization_denormalization_round_trip_and_shapes():
    x = torch.randn(3, 48, 7) * 4 + 12
    revin = RevIN(7, affine=True)
    normalized = revin.normalize(x)
    restored = revin.denormalize(normalized)
    assert normalized.shape == x.shape == restored.shape
    assert torch.allclose(restored, x, atol=1e-5)


@pytest.mark.parametrize("use_revin", [False, True])
def test_both_revin_options_have_expected_shape_and_gradients(use_revin):
    model = ModernTCN(7, 48, 12, embed_dim=8, depth=1, kernel_size=7,
                      small_kernel_size=3, patch_size=8, patch_stride=4,
                      dropout=0.0, use_revin=use_revin)
    prediction = model(torch.randn(2, 48, 7))
    assert prediction.shape == (2, 12, 7)
    assert (model.revin is not None) is use_revin
    prediction.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_fixed_seed_reproduces_initial_prediction():
    """The ablation can start both variants from reproducible RNG states."""
    set_seed(1)
    model_a = ModernTCN(7, 48, 12, embed_dim=8, depth=1, kernel_size=7,
                        small_kernel_size=3, patch_size=8, patch_stride=4,
                        dropout=0.0, use_revin=True).eval()
    x = torch.randn(2, 48, 7)
    set_seed(1)
    model_b = ModernTCN(7, 48, 12, embed_dim=8, depth=1, kernel_size=7,
                        small_kernel_size=3, patch_size=8, patch_stride=4,
                        dropout=0.0, use_revin=True).eval()
    # Recreate the input after model construction to consume the same RNG draws.
    x_b = torch.randn(2, 48, 7)
    assert torch.equal(x, x_b)
    assert torch.equal(model_a(x), model_b(x_b))
