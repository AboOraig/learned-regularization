"""Noise-level-conditioned DnCNN (FFDNet-style: sigma map as 2nd input channel).
One network handles all sigma in [0, sigma_max]; sigma_d is then a tunable PnP parameter."""
import torch, torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, depth=8, ch=64):
        super().__init__()
        L = [nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            L += [nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        L += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*L)

    def forward(self, x, sigma):  # x: (B,1,H,W), sigma: (B,)
        smap = sigma.view(-1, 1, 1, 1).expand_as(x)
        return x - self.net(torch.cat([x, smap], 1))  # residual learning


def load_denoiser(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    m = DnCNN(**ck["config"]).to(device)
    m.load_state_dict(ck["state"]); m.eval()
    return m


@torch.no_grad()
def denoise(model, x_flat, sigma, size=28):
    x = x_flat.reshape(-1, 1, size, size)
    s = torch.full((len(x),), float(sigma), device=x.device)
    return model(x, s).clamp(0, 1).reshape(len(x), -1)
