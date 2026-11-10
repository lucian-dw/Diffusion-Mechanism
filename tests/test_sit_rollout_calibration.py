from pathlib import Path

import torch


def test_paired_heun_endpoints_and_linear_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from run_sit_rollout_calibration import heun, metrics, paired_endpoints

    def velocity(x, _t, labels):
        return labels[:, None].to(x) + torch.zeros_like(x)

    x = torch.tensor([[0.25, 0.5], [0.75, 1.0]], dtype=torch.float64)
    labels = torch.tensor([1, 3])
    ref = heun(velocity, x, labels, 0.0, 1.0, 8, keep=True)
    torch.testing.assert_close(ref[-1], x + labels[:, None])
    defect = torch.tensor([[0.4, -0.1], [0.2, 0.3]], dtype=torch.float64)
    amplitudes = (0.0, 0.01, 0.1, 1.0, -0.01, -0.005, 0.005)
    low = paired_endpoints(velocity, ref[4], defect, labels, amplitudes, 4)
    high = paired_endpoints(velocity, ref[4], defect, labels, amplitudes, 8)
    for a in amplitudes:
        torch.testing.assert_close(high[a], ref[-1] + a * defect)
    scored = metrics(ref[-1], ref[-1], ref[-1], low, high)
    for name, result in scored.items():
        if not name.startswith("signal_"):
            assert result.abs().max() < 1e-10, name
