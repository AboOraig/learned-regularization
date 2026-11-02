"""Stage 3 / step 4c: side-by-side comparison of the step 4/4b results (original grids) with the re-tuned run (extended grids, TV-long tuned
separately, 20 validation slices). Prints what changed and which conclusions survive.
Usage: python run_stage3_step4c_compare.py --old results/stage3 --new results/stage3/s4c"""
import argparse, json, glob, os
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--old", default="results/stage3"); p.add_argument("--new", default="results/stage3/s4c")
a = p.parse_args()
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
rd = lambda d, n: pd.read_csv(f"{d}/{n}") if os.path.exists(f"{d}/{n}") else None

print("=== tuned parameters in the new run ===")
for fn in sorted(glob.glob(f"{a.new}/step4_params_dose*.json")):
    print(os.path.basename(fn), json.load(open(fn)))

o, n = rd(a.old, "step4_lesion_summary.csv"), rd(a.new, "step4_lesion_summary.csv")
if o is not None and n is not None:
    k = ["set", "dose", "method"]
    g = lambda D: D.groupby(k, sort=False)["psnr_absent"].first().rename
    t = pd.concat([g(o)("psnr_old"), g(n)("psnr_new")], axis=1).dropna().reset_index(); t["delta_dB"] = t.psnr_new - t.psnr_old
    print("\n=== lesion-free PSNR (mean over the same slices): old grids vs re-tuned ===\n" + t.round(3).to_string(index=False))

o, n = rd(a.old, "step4b_paired.csv"), rd(a.new, "step4b_paired.csv")
if o is not None and n is not None:
    k = ["set", "dose", "types", "metric", "A", "B"]
    m = o.merge(n, on=k, suffixes=("_old", "_new"))
    m["same_verdict"] = m.excludes_0_old == m.excludes_0_new
    P = m.query("types == 'pooled'")[["set", "dose", "metric", "A", "B", "diff_old", "lo_old", "hi_old", "diff_new", "lo_new", "hi_new", "excludes_0_old", "excludes_0_new"]]
    print("\n=== PAIRED pooled differences, old vs new ===\n" + P.round(3).to_string(index=False))
    ch = P[P.excludes_0_old != P.excludes_0_new]
    print(f"\npooled verdicts that CHANGED (interval excludes 0 or not): {len(ch)} of {len(P)}")
    if len(ch):
        print(ch.round(3).to_string(index=False))

o, n = rd(a.old, "step4b_theta_sweep.csv"), rd(a.new, "step4b_theta_sweep.csv")
if o is not None and n is not None:
    k = ["set", "dose", "sigma_l", "contrast", "theta_mult", "method", "kind"]
    m = o.merge(n, on=k, suffixes=("_old", "_new"))
    m = m[(m.kind == "paired_diff") & (m.theta_mult >= 1.0)]
    print("\n=== threshold sweep, paired differences at >= 1x lesion strength, old vs new ===\n" +
          m[["set", "dose", "sigma_l", "theta_mult", "method", "false_per_img_old", "lo_old", "hi_old", "false_per_img_new", "lo_new", "hi_new"]].round(3).to_string(index=False))
