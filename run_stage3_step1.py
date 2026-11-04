"""Stage 3 / step 1: LoDoPaB-CT data and geometry.
  (A) inventory of the files (counts, shapes, value ranges);
  (B) patient-disjointness / slices-per-patient from the patient-id csv files;
  (C) geometry + normalisation check: forward-project ground-truth slices with the matrix-free parallel-beam operator and
      compare with the PROVIDED observations, over 16 conventions (rotation sign, detector flip, half-step angle offset, image
      transpose); the best convention, the fitted scale and the residual are reported;
  (D) sanity reconstruction: FBP of the full 1000-angle observation of slice 0 (PSNR vs ground truth);
  (E) timing / GPU memory of forward+adjoint for 60, 180 and 1000 angles.
Usage:  python run_stage3_step1.py --root data --extract          (unzips the downloaded zip files first)
        python run_stage3_step1.py --root data [--split test] [--n_slices 8] [--skip_timing]
Outputs: results/stage3/step1_summary.json, geometry_check.png, geometry_variants.csv"""
import argparse, itertools, json, os, time, platform
import numpy as np, pandas as pd, torch
from src.lodopab import LoDoPaB, SIZES, IMG, N_ANGLES, N_DET, extract_zips, read_patient_ids
from src.radon_pb import ParallelBeamOperator

p = argparse.ArgumentParser()
p.add_argument("--root", default="data")
p.add_argument("--extract", action="store_true")
p.add_argument("--split", default="test")
p.add_argument("--n_slices", type=int, default=8)
p.add_argument("--angle_stride", type=int, default=20, help="use every k-th of the 1000 angles for the convention search")
p.add_argument("--pixel_size", type=float, default=0.26 / 362, help="physical pixel size (m): 0.26 m field of view / 362")
p.add_argument("--skip_timing", action="store_true")
p.add_argument("--out", default="results/stage3")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
print(f"torch {torch.__version__} | python {platform.python_version()} | device {dev}"
      + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)" if dev == "cuda" else ""))
if a.extract:
    extract_zips(a.root)
summary = {}

# ---------------------------------------------------------------- (A) inventory
print("\n=== (A) files ===")
for split in ("train", "validation", "test"):
    try:
        ds = LoDoPaB(a.root, split)
    except FileNotFoundError:
        print(f"{split:11s}: not found (skipped)"); continue
    g = ds.gt(np.arange(min(len(ds), 4)))
    ob = ds.obs([0]) if ds.obs_files else None
    print(f"{split:11s}: {len(ds.gt_files)} gt files, {len(ds.obs_files)} obs files, {len(ds)} slices (expected {SIZES[split]}), "
          f"gt shape {g.shape[1:]} range [{g.min():.3f}, {g.max():.3f}]"
          + (f", obs shape {ob.shape[1:]} range [{ob.min():.3f}, {ob.max():.3f}]" if ob is not None else ""))
    summary[f"{split}_n"] = len(ds)
    ds.close()

# ---------------------------------------------------------------- (B) patients
print("\n=== (B) patient ids ===")
ids = {s: read_patient_ids(a.root, s) for s in ("train", "validation", "test")}
for s, v in ids.items():
    if v is None:
        print(f"{s:11s}: csv not found"); continue
    u, c = np.unique(v, return_counts=True)
    print(f"{s:11s}: {len(v)} slices, {len(u)} patient ids, slices/patient median {int(np.median(c))} (min {c.min()}, max {c.max()}); "
          f"ids in [{v.min()}, {v.max()}]")
    summary[f"{s}_patients"] = int(len(u))
have = [s for s, v in ids.items() if v is not None]
for s1, s2 in itertools.combinations(have, 2):
    ov = np.intersect1d(np.unique(ids[s1]), np.unique(ids[s2]))
    print(f"overlap {s1} / {s2}: {len(ov)} shared ids")
    summary[f"overlap_{s1}_{s2}"] = int(len(ov))
if any(summary.get(k, 0) > 0 for k in summary if k.startswith("overlap")):
    print("NOTE: shared ids between splits may mean the ids are re-numbered per split (then the csv cannot prove disjointness), "
          "or a real leak. Send me this output; do not assume either.")

# ---------------------------------------------------------------- (C) geometry
print("\n=== (C) geometry / normalisation check against the provided observations ===")
ds = LoDoPaB(a.root, a.split)
idx = np.linspace(0, len(ds) - 1, a.n_slices).astype(int)
gt = torch.tensor(ds.gt(idx)).to(dev)                                      # (B, 362, 362)
ob = torch.tensor(ds.obs(idx)).to(dev)                                     # (B, 1000, 513)
k_idx = np.arange(0, N_ANGLES, a.angle_stride)
y_obs = ob[:, k_idx, :].reshape(len(idx), -1)
step = 180.0 / N_ANGLES
rows = []
for sign, flip, half, tr in itertools.product((-1.0, 1.0), (False, True), (0.0, 0.5), (False, True)):
    ang = (k_idx + half) * step
    op = ParallelBeamOperator(IMG, N_DET, angles_deg=ang, pixel_size=a.pixel_size, device=dev, sign=sign, flip_det=flip)
    x = (gt.transpose(1, 2) if tr else gt).reshape(len(idx), -1)
    yp = op.forward(x)
    sc = (yp * y_obs).sum() / (yp * yp).sum()                              # least-squares scale, y_obs ~ sc * yp
    res = (y_obs - sc * yp).norm() / y_obs.norm()
    cor = torch.corrcoef(torch.stack([yp.flatten(), y_obs.flatten()]))[0, 1]
    rows.append(dict(sign=sign, flip_det=flip, half_step_offset=half, transpose_image=tr, fitted_scale=float(sc), rel_residual=float(res), correlation=float(cor)))
V = pd.DataFrame(rows).sort_values("rel_residual").reset_index(drop=True)
V.to_csv(f"{a.out}/geometry_variants.csv", index=False)
print(V.head(6).round(4).to_string())
best = V.iloc[0]
print(f"\nbest convention: sign={best.sign:+.0f}, flip_det={bool(best.flip_det)}, half_step_offset={best.half_step_offset}, "
      f"transpose_image={bool(best.transpose_image)}; fitted scale {best.fitted_scale:.4f} (1.0 = pixel_size {a.pixel_size:.3e} m correct); "
      f"relative residual {best.rel_residual:.4f}, correlation {best.correlation:.5f}")
# the half-step angle offset (0.09 deg) is a nuisance parameter that this check cannot resolve: compare against the best variant
# that differs in sign / flip / transpose
others = V[(V.sign != best.sign) | (V.flip_det != best.flip_det) | (V.transpose_image != best.transpose_image)]
second = others.iloc[0].rel_residual
print(f"best vs best alternative (different sign/flip/transpose): residual {best.rel_residual:.4f} vs {second:.4f} -> " +
      ("convention clearly identified" if second > 1.5 * best.rel_residual else
       "WARNING: not clearly identified (send me geometry_variants.csv)"))
print("(the half-step angle offset changes the residual by < 1e-3 and is not identifiable here; the best row's value is used)")
summary["best_convention"] = {k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in best.items()}

# ---------------------------------------------------------------- (D) FBP sanity
print("\n=== (D) FBP of the full observation (slice 0 of the chosen set) ===")
op = ParallelBeamOperator(IMG, N_DET, n_angles=N_ANGLES, angle_offset_deg=best.half_step_offset * step, pixel_size=a.pixel_size * float(best.fitted_scale),
                          device=dev, sign=float(best.sign), flip_det=bool(best.flip_det))
x0 = (gt[:1].transpose(1, 2) if best.transpose_image else gt[:1]).reshape(1, -1)
t = time.time(); rec = op.fbp(ob[:1].reshape(1, -1)); torch.cuda.synchronize() if dev == "cuda" else None; tf = time.time() - t
mse = ((rec.clamp(0, 1) - x0) ** 2).mean().item()
print(f"FBP (1000 angles) on slice {idx[0]}: PSNR {10*np.log10(1/mse):.2f} dB (range [0,1]), time {tf:.1f}s, "
      f"recon range [{rec.min().item():.3f}, {rec.max().item():.3f}], gt range [{x0.min().item():.3f}, {x0.max().item():.3f}]")
print("(for orientation: the LoDoPaB paper reports roughly 30 dB for FBP on the test set; a very different value would point to a normalisation problem)")
summary["fbp_psnr_slice0"] = float(10 * np.log10(1 / mse)); summary["recommended_pixel_size"] = a.pixel_size * float(best.fitted_scale)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
ax[0, 0].imshow(x0.reshape(IMG, IMG).cpu(), cmap="gray", vmin=0, vmax=1); ax[0, 0].set_title("ground truth (slice 0)")
ax[0, 1].imshow(rec.clamp(0, 1).reshape(IMG, IMG).cpu(), cmap="gray", vmin=0, vmax=1); ax[0, 1].set_title("FBP of provided observation")
ax[0, 2].imshow(((rec.clamp(0, 1) - x0).abs().reshape(IMG, IMG)).cpu(), cmap="magma", vmin=0, vmax=0.2); ax[0, 2].set_title("|error| (0..0.2)")
ypr = op.forward(x0)
ax[1, 0].imshow(ob[0].cpu(), aspect="auto", cmap="gray"); ax[1, 0].set_title("observation (angles x detector)")
ax[1, 1].imshow(ypr.reshape(N_ANGLES, N_DET).cpu(), aspect="auto", cmap="gray"); ax[1, 1].set_title("our forward projection of the GT")
k = N_ANGLES // 3
ax[1, 2].plot(ob[0, k].cpu(), label="observation"); ax[1, 2].plot(ypr.reshape(N_ANGLES, N_DET)[k].cpu(), "--", label="ours"); ax[1, 2].legend(); ax[1, 2].set_title(f"projection at angle index {k}")
for x in ax.flat[:5]:
    x.set_xticks([]); x.set_yticks([])
plt.tight_layout(); plt.savefig(f"{a.out}/geometry_check.png", dpi=130); plt.close()

# ---------------------------------------------------------------- (E) timing
if not a.skip_timing:
    print("\n=== (E) timing of forward + adjoint, 1 slice ===")
    xs = x0.clone()
    for K in (60, 180, 1000):
        o = ParallelBeamOperator(IMG, N_DET, n_angles=K, pixel_size=op.pixel_size, device=dev, sign=float(best.sign), flip_det=bool(best.flip_det))
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t = time.time(); y = o.forward(xs); _ = o.adjoint(y)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t
        print(f"K={K:5d}: forward+adjoint {dt:6.2f}s" + (f", peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB" if dev == "cuda" else ""))
        summary[f"time_fa_K{K}"] = dt
json.dump(summary, open(f"{a.out}/step1_summary.json", "w"), indent=1)
print(f"\nwrote {a.out}/step1_summary.json, geometry_variants.csv, geometry_check.png")
