from pathlib import Path

import numpy as np
import torch


def test_exponential_action_matches_dense_nonnormal_operator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from run_sit_order_controls import exp_action

    rng = torch.Generator().manual_seed(101)
    a = torch.randn(2, 7, 7, dtype=torch.float64, generator=rng) * 0.4
    x = torch.randn(2, 7, dtype=torch.float64, generator=rng)
    action = exp_action(lambda v: (a @ v[..., None]).squeeze(-1), x, 7, (4, 7))
    torch.testing.assert_close(action[7], (torch.matrix_exp(a) @ x[..., None]).squeeze(-1), rtol=1e-11, atol=1e-11)
    zero = exp_action(lambda v: v * 0, x, 5, (5,))
    torch.testing.assert_close(zero[5], x)


def test_simpson_integrates_cubic_and_probes_preserve_norm(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from run_sit_order_controls import isotropic, simpson_weights

    w = simpson_weights(8)
    np.testing.assert_allclose(w @ np.linspace(0.5, 1, 9)**3, (1 - 0.5**4) / 4)
    defect = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    p = isotropic(defect, [2, 3], 8)
    torch.testing.assert_close(p.flatten(2).norm(dim=2), defect.flatten(1).norm(dim=1)[:, None].expand(-1, 8))
    torch.testing.assert_close(p[1:], isotropic(defect[1:], [3], 8))
