"""Loss-construction invariants (design_spec §2 H2, §3, invariant #7 "δ vs signal scale").

These are the F-01 kill probes: they prove the Huber knee δ, quoted in dB in config,
stays O(1)-meaningful in the z-scored domain the trainer actually optimizes — i.e. the
loss does NOT silently degenerate to MSE (the H2 pathology).
"""
import torch

from amcd.training.loss import delta_db_to_norm, build_criterion
from tests.conftest import tiny_config


def test_delta_db_to_norm_preserves_db_threshold() -> None:
    """A residual of exactly `delta_db` dB maps to the Huber knee in the normalized
    domain: delta_db_to_norm(d, s) * s == d (within the 1e-8 std guard)."""
    for delta_db, high_std in [(1.0, 16.0), (0.5, 8.3), (2.0, 25.0)]:
        delta_norm = delta_db_to_norm(delta_db, high_std)
        assert abs(delta_norm * high_std - delta_db) < 1e-4


def test_huber_delta_active_in_training_domain() -> None:
    """The falsifier's F-01 kill-or-confirm probe, on the REAL (trainer) operands.

    Draw a realistic residual distribution in dB (std ~3 dB), express it in the
    z-scored domain the criterion sees, and assert a materially non-trivial fraction
    falls in the Huber LINEAR regime (|residual| > δ). With the correct dB-scaled δ
    this is large (~0.7); with the pre-fix bug (raw δ=1.0 on z-scored operands) it
    collapses to ~0 — the assertion pins the fix, not the bug.
    """
    cfg = tiny_config()                    # cfg.huber_delta == 1.0 dB (config)
    high_std = 16.0                        # a realistic log-band-energy std (dB); test datum
    norm_stats = {"high_mean": 0.0, "high_std": high_std}

    criterion = build_criterion(cfg, norm_stats)
    delta_norm = criterion.delta           # nn.HuberLoss stores its delta

    torch.manual_seed(0)
    residual_db = torch.randn(4, 4, 10, 20) * 3.0        # ~3 dB residual spread
    residual_norm = residual_db / high_std               # what the loss operates on

    linear_frac = (residual_norm.abs() > delta_norm).float().mean().item()
    assert linear_frac > 0.3, (
        f"Huber knee inert: only {linear_frac:.2%} of residuals in the linear regime — "
        f"δ is not O(1)-meaningful in dB (H2 degeneracy)."
    )

    # Contrast: the pre-fix bug used raw δ=1.0 against z-scored operands. Show that
    # would have been degenerate (knee at high_std dB ≈ 16 dB → almost all quadratic).
    buggy_linear_frac = (residual_norm.abs() > cfg.huber_delta).float().mean().item()
    assert buggy_linear_frac < 0.05, "sanity: the pre-fix knee should be effectively MSE"
