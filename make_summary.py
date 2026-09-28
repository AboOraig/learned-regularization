"""Aggregate all results/*.csv into one markdown table. Usage: python make_summary.py"""
import pandas as pd, glob, os

os.makedirs("results", exist_ok=True)
frames = []
for f in glob.glob("results/*.csv"):
    d = pd.read_csv(f); d["experiment"] = os.path.basename(f).replace(".csv", "")
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
id_cols = [c for c in ["experiment", "sigma", "n_angles", "kind", "delta", "dataset"] if c in df.columns]
df[id_cols] = df[id_cols].fillna("")  # groupby drops rows with NaN keys by default; not every
                                       # experiment has every id column, so this would zero out everything
g = df.groupby(id_cols + ["method"], dropna=False).agg(
    psnr_mean=("psnr", "mean"), psnr_sem=("psnr", "sem"),
    ssim_mean=("ssim", "mean"), ssim_sem=("ssim", "sem")).reset_index()
g = g.round(3)
with open("results/summary_table.md", "w") as f:
    f.write(g.to_markdown(index=False))
print(f"wrote results/summary_table.md ({len(g)} rows)")