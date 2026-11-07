"""E19 control: bootstrap interval for the K=60 PSNR difference PnP-const - TV-long on the 20 shape phantoms (Table tab:shift3a, K=60 rows, which has none).
Reads the per-phantom PSNR stored by run_stage3_step4.py (results/stage3/s4c/step4_shapes_dose{1,0.25}.npz; SSIM was not stored for K=60).
Percentile bootstrap over phantoms, 5000 resamples, as in the other Stage 3 tables.
Usage (repository root):  python run_e19_k60_psnr_ci.py        Output: results/control/e19_k60_psnr_ci.csv"""
import os, numpy as np, pandas as pd
rng = np.random.default_rng(0); rows = []
for dose in ("1", "0.25"):
    d = np.load(f"results/stage3/s4c/step4_shapes_dose{dose}.npz", allow_pickle=True)
    for a, b in (("PnP-const", "TV-long"), ("PnP-const", "TV"), ("PnP-const", "FBP-Hann")):
        x = d[f"psnr|{a}"].astype(float) - d[f"psnr|{b}"].astype(float); n = len(x)
        bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(5000)])
        rows.append(dict(K=60, dose=float(dose), a=a, b=b, n=n, diff=x.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
df = pd.DataFrame(rows); os.makedirs("results/control", exist_ok=True)
df.to_csv("results/control/e19_k60_psnr_ci.csv", index=False); print(df.round(3).to_string(index=False))
