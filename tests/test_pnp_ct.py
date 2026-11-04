import numpy as np, torch
from src.radon_pb import ParallelBeamOperator
from src.pnp_ct import pnp_ct, denoise_img


class Id(torch.nn.Module):
    def forward(self, x, s):
        return x


def test_denoise_img_chunks_and_size():
    x = torch.rand(10, 16 * 16)
    assert torch.allclose(denoise_img(Id(), x, 0.05, 16, chunk=3), x)


def test_pnp_with_identity_denoiser_is_landweber():
    H = 24
    op = ParallelBeamOperator(H, 35, n_angles=20, pixel_size=1.0)
    x = torch.rand(2, H * H) * 0.5
    y = op.forward(x)
    eta = 1.0 / op.lipschitz()
    out = pnp_ct(op, y, Id(), 0.05, eta, (1, 5, 30), torch.zeros(2, H * H))
    r = lambda z: float((op.forward(z) - y).norm())
    assert r(out[30]) < r(out[5]) < r(out[1])
