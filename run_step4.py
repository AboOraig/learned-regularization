"""Stage 1 / step 4: seeds, wider sigma_d range, matched-prior control.  One resumable driver.

Denoiser families (3 seeds each):
  m03  MNIST,   sigma_max 0.3  (= the v1 recipe; measures seed-to-seed variability)
  m05  MNIST,   sigma_max 0.5  (can actually use sigma_d up to 0.5 -> fixes the sigma=0.2 grid-edge tuning)
  f03  Fashion, sigma_max 0.3  (matched-prior control: tuned and tested with Fashion-MNIST as the in-distribution set)
Per model:  train  ->  shift experiment (bicubic f=5 data, sigma=0.05, 20 angles; every method re-tuned on the
model's OWN validation distribution, then frozen)  ->  (MNIST families only) noise sweep sigma in {0.05, 0.1, 0.2}.
All runs use the extended sigma_d grid so families are compared on identical search spaces.

Usage:   python run_step4.py                    # everything (resumable: finished outputs are skipped)
         python run_step4.py --stage train      # or --stage eval
         python run_step4.py --seeds 0 --families m05      # subset;  --epochs 8;  --force to redo
"""
import argparse, os, subprocess, sys, time

p = argparse.ArgumentParser()
p.add_argument("--stage", default="all", choices=["train", "eval", "all"])
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--families", default="m03,m05,f03")
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--out", default="results/step4")
p.add_argument("--force", action="store_true")
a = p.parse_args()

FAM = {"m03": dict(data="mnist", sigma_max=0.3), "m05": dict(data="mnist", sigma_max=0.5), "f03": dict(data="fashion", sigma_max=0.3)}
SD_GRID = "0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5"
os.makedirs(a.out, exist_ok=True)


def run(cmd, done_file=None):
    if done_file and os.path.exists(done_file) and not a.force:
        print(f"skip (exists): {done_file}", flush=True); return
    print("\n>>", " ".join(cmd), flush=True); t0 = time.time()
    subprocess.run([sys.executable] + cmd, check=True)
    print(f"   done in {time.time() - t0:.0f}s", flush=True)


for fam in a.families.split(","):
    for seed in [int(s) for s in a.seeds.split(",")]:
        name = f"{fam}_s{seed}"; cfg = FAM[fam]; ck = f"{a.out}/dn_{name}.pt"
        if a.stage in ("train", "all"):
            run(["train_denoiser.py", "--data", cfg["data"], "--seed", str(seed), "--sigma_max", str(cfg["sigma_max"]),
                 "--epochs", str(a.epochs), "--out", ck], ck)
        if a.stage in ("eval", "all"):
            common = ["run_nullspace.py", "--model", ck, "--fine_factor", "5", "--up_mode", "bicubic", "--sd_grid", SD_GRID, "--out", a.out]
            tune = "fashion" if cfg["data"] == "fashion" else "mnist"
            run(common + ["--exp", "shift", "--tune_set", tune, "--suffix", f"_shift_{name}"], f"{a.out}/nullspace_pnp_vs_tv_shift_{name}.csv")
            if cfg["data"] == "mnist":
                run(common + ["--exp", "noise", "--sigmas", "0.05,0.1,0.2", "--suffix", f"_noise_{name}"], f"{a.out}/nullspace_pnp_vs_tv_noise_{name}.csv")
print("\nAll requested stages finished. Next: python summarize_step4.py")
