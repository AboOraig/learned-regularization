import numpy as np, torch
from src.radon_pb import ParallelBeamOperator
from src.pnp_ct import pnp_ct, pnp_ct_sched, anneal_schedule


class Shrink(torch.nn.Module):          # sigma-dependent toy denoiser: x -> x * (1 - sigma)
    def forward(self, x, s):
        return x * (1 - s.view(-1, 1, 1, 1))


def test_anneal_schedule():
    s = anneal_schedule(0.1, 0.01, 5)
    assert abs(s[0] - 0.1) < 1e-9 and abs(s[-1] - 0.01) < 1e-9 and all(a > b for a, b in zip(s, s[1:]))
    assert anneal_schedule(0.1, 0.01, 1) == [0.01]


def test_constant_schedule_equals_pnp_ct():
    H = 20
    op = ParallelBeamOperator(H, 29, n_angles=12, pixel_size=1.0)
    x = torch.rand(2, H * H) * 0.5; y = op.forward(x); eta = 1 / op.lipschitz(); x0 = torch.rand(2, H * H) * 0.5
    a = pnp_ct(op, y, Shrink(), 0.05, eta, (6,), x0)[6]
    b = pnp_ct_sched(op, y, Shrink(), [0.05] * 6, eta, (6,), x0)[6]
    assert torch.allclose(a, b, atol=1e-6)
    c = pnp_ct_sched(op, y, Shrink(), anneal_schedule(0.2, 0.01, 6), eta, (6,), x0)[6]
    assert not torch.allclose(a, c, atol=1e-4)
