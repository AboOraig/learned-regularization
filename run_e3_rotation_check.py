"""E3 control: is the constant-angle-offset 'degradation' just a rigid rotation of the reconstruction?
Re-runs the E3 offset experiment (same data, seeds, validation tuning and frozen parameters as run_robustness.py --exp mismatch)
and scores each reconstruction against (a) the true digit and (b) the true digit rotated by +delta and -delta degrees
(skimage.transform.rotate, cubic, zero padding). Also reports the 'ceiling' PSNR(rot(x), x) of a perfect but rotated reconstruction.
Usage: python run_e3_rotation_check.py        -> results/control/e3_offset_rotation.csv"""
import os, numpy as np, pandas as pd, torch
from skimage.transform import rotate
from src.operators import RadonOperator, add_noise
from src.denoiser import load_denoiser
from src.experiment import tune_all, reconstruct_all
from src.metrics import per_image_metrics
from src.data import load_mnist

dev = "cuda" if torch.cuda.is_available() else "cpu"
_, val, test = load_mnist(); val, test = val[:100].to(dev), test[:200].to(dev)
den = load_denoiser("results/dncnn.pt", dev)
op = RadonOperator(28, 20, device=dev)
params = tune_all(op, den, val, add_noise(op.forward(val), 0.05, seed=0)); print("frozen params:", params)
yt = add_noise(op.forward(test), 0.05, seed=1)          # data from the TRUE operator (as in E3)
base = np.linspace(0, 180, 20, endpoint=False)

def rot(x, ang):
    a = x.reshape(-1, 28, 28).cpu().numpy()
    r = np.stack([rotate(i, ang, resize=False, order=3, mode="constant", cval=0, preserve_range=True) for i in a])
    return torch.tensor(np.clip(r, 0, 1).reshape(len(a), -1), dtype=torch.float32, device=dev)

rows = []
for delta in [0, 1, 2, 4]:
    xs = reconstruct_all(RadonOperator(28, angles=base + delta, device=dev), den, yt, params)
    for name in ["FBP", "TV", "PnP-DnCNN"]:
        r = dict(delta=delta, method=name, vs_true=float(per_image_metrics(xs[name], test)[0].mean()))
        for sgn in (+1, -1):
            r[f"vs_rot{sgn:+d}"] = float(per_image_metrics(xs[name], rot(test, sgn * delta))[0].mean())
        rows.append(r)
    if delta:
        rows.append(dict(delta=delta, method="ceiling (rot(x) vs x)", vs_true=float(per_image_metrics(rot(test, delta), test)[0].mean())))
df = pd.DataFrame(rows); print(df.round(2).to_string())
os.makedirs("results/control", exist_ok=True); df.to_csv("results/control/e3_offset_rotation.csv", index=False)
