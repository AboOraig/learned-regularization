"""Stage 2 / lesion test under POISSON noise with the MATRIX-FREE operator. (Same protocol as Stage 1 run_lesions.py; see below.)
Extra args: --I0 (photons/ray), --mu (attenuation). Outputs go to --out (use one dir per dose, then run summarize_step5.py --dir <that dir>).

Original docstring: Stage 1 / step 5: lesion-insertion test (direct test of Q2: does the reconstruction erase or invent lesion-like structure?).

For every denoiser in --models (checkpoints from run_step4.py) and every base set (MNIST test, Fashion-MNIST test):
  * tune FBP/Tikhonov/TV/PnP on the model's own validation distribution (crime-free fine-grid data), then freeze;
  * lesion-ABSENT arm : reconstruct the base images; lesion-PRESENT arm: insert a known Gaussian lesion at a known random
    location (same location for every model/method), reconstruct;  independent noise draws for the two arms;
    lesions whose inserted contrast is < 50% of nominal in the TRUTH (clipped on saturated strokes) are excluded from detection scores;
  * for each (lesion size, contrast, method) report  paired AUC (known location AND same base image: background cancels) and
    unpaired AUC (background varies), contrast retention, false lesions per
    lesion-absent image, and the lesion's observability w.r.t. the weakly-determined subspace of the operator.
Outputs (results/step5): lesion_<model>_metrics.csv, lesion_<model>_per_image.csv, lesion_<model>_examples_<set>.png.  Resumable.
Usage: python run_lesions.py [--models m03_s0,m03_s1,m03_s2,f03_s0,f03_s1,f03_s2] [--n_test 200] [--force]
"""
import argparse, os, time
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.radon_mf import MatrixFreeRadon
from src.mf_solvers import poisson_from_clean, strong_basis
from src.mf_tune import MFSolvers
from src.denoiser import load_denoiser
from src.finegrid import make_simulator
from src.data import load_mnist, load_fashion_train, load_fashion
from src import lesions as L

p = argparse.ArgumentParser()
p.add_argument("--models", default="m03_s0,m03_s1,m03_s2,f03_s0,f03_s1,f03_s2")
p.add_argument("--ckpt_dir", default="results/step4")
p.add_argument("--out", default="results/stage2/lesions_I100")
p.add_argument("--I0", type=float, default=100.0)
p.add_argument("--mu", type=float, default=2.0)
p.add_argument("--sets", default="mnist,fashion")
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--n_angles", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sizes", default="0.5,0.75,1.25,2.0", help="lesion Gaussian std in px")
p.add_argument("--contrasts", default="0.25,0.5")
p.add_argument("--fine_factor", type=int, default=5)
p.add_argument("--up_mode", default="bicubic")
p.add_argument("--sd_grid", default="0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
p.add_argument("--n_boot", type=int, default=500)
p.add_argument("--example_type", type=int, default=3, help="index into (size x contrast) list used for the example figure")
p.add_argument("--force", action="store_true")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
SDG = [float(v) for v in a.sd_grid.split(",")]
SIZES = [float(s) for s in a.sizes.split(",")]; CONTR = [float(c) for c in a.contrasts.split(",")]
TYPES = [(s, c) for s in SIZES for c in CONTR]
LO, HI = L.centre_range(SIZES)
METHODS = ["FBP", "Tikhonov", "TV", "PnP-DnCNN"]
op = MatrixFreeRadon(28, a.n_angles, device=dev)
sim = make_simulator(op, a.fine_factor, a.up_mode)
Vs = strong_basis(op, 0.05)             # matrix-free well-determined subspace (465 dims at 20 angles)
solv = MFSolvers(op)
noisy = lambda clean, seed: poisson_from_clean(clean, a.I0, seed, a.mu)[0]
mn_train, mn_val, mn_test = load_mnist()
BASES = {"mnist": ("MNIST", mn_test[:a.n_test])}
if "fashion" in a.sets:
    BASES["fashion"] = ("Fashion-MNIST", load_fashion()[:a.n_test])
fa_val = load_fashion_train()[1] if any(m.startswith("f") for m in a.models.split(",")) else None
print(f"device {dev} | lesion types (sigma_l, contrast): {TYPES} | centre range [{LO},{HI}] | Poisson I0={a.I0:g}, mu={a.mu}, {a.n_angles} angles", flush=True)


def rec(den, params, y):
    return {k: v.clamp(0, 1) for k, v in solv.reconstruct(den, y, params).items()}


for name in a.models.split(","):
    f_met, f_img = f"{a.out}/lesion_{name}_metrics.csv", f"{a.out}/lesion_{name}_per_image.csv"
    want = {BASES[k][0] for k in a.sets.split(",")}
    have = set(pd.read_csv(f_met).test_set) if os.path.exists(f_met) else set()
    if os.path.exists(f_img) and want <= have and not a.force:        # skip only if EVERY requested base set was already computed
        print(f"skip (complete for {sorted(want)}): {f_met}", flush=True); continue
    if have and not want <= have:
        print(f"recomputing {name}: existing file only has {sorted(have)}", flush=True)
    tstart = time.time(); fam, seed = name.rsplit("_s", 1); tuned_on = "fashion" if fam.startswith("f") else "mnist"
    den = load_denoiser(f"{a.ckpt_dir}/dn_{name}.pt", dev)
    val = (fa_val if tuned_on == "fashion" else mn_val)[:a.n_val].to(dev)
    params = solv.tune(den, val, noisy(sim(val), a.seed), SDG)
    print(f"\n[{name}] tuned on {tuned_on} validation: {params}", flush=True)
    mrows, irows = [], []
    for si, (key, (set_name, xb)) in enumerate([(k, BASES[k]) for k in a.sets.split(",")]):
        x = xb.to(dev); n = len(x)
        role = "in-dist" if key == tuned_on else "shifted"
        rng = np.random.default_rng(a.seed + 7 + si)                       # lesion centres: identical for every model
        centers = torch.tensor(rng.integers(LO, HI + 1, size=(n, 2)), device=dev)
        rec0 = rec(den, params, noisy(sim(x), a.seed + 100 + si))      # lesion-absent arm
        for li, (sl, c) in enumerate(TYPES):
            rho = L.lesion_rho(sl); theta = 0.5 * L.nominal_response(sl, c)
            xl = L.insert_lesion(x, centers, sl, c)
            recl = rec(den, params, noisy(sim(xl), a.seed + 200 + 10 * si + li))
            e = xl - x; en = (e ** 2).sum(1); obs = ((e @ Vs) ** 2).sum(1).div(en.clamp_min(1e-12)).masked_fill(en == 0, float('nan')).cpu().numpy()   # fraction of lesion energy in the well-determined subspace
            tt1, tt0 = L.contrast_at(xl, rho, centers).cpu().numpy(), L.contrast_at(x, rho, centers).cpu().numpy()
            valid = (tt1 - tt0) >= 0.5 * L.nominal_response(sl, c)      # lesion on a saturated stroke is clipped away in the TRUTH: not a lesion
            for m in METHODS:
                t1, t0 = L.contrast_at(recl[m], rho, centers).cpu().numpy(), L.contrast_at(rec0[m], rho, centers).cpu().numpy()
                fl, cand = L.count_false_lesions(rec0[m], x, rho, theta)
                lo, hi = L.bootstrap_auc_ci(t1[valid], t0[valid], a.n_boot, seed=a.seed)
                mrows.append(dict(model=name, family=fam, seed=int(seed), tuned_on=tuned_on, test_set=set_name, role=role, sigma_l=sl, contrast=c, rho=rho,
                                  method=m, n_valid=int(valid.sum()), auc=L.auc(t1[valid], t0[valid]), auc_lo=lo, auc_hi=hi, paired_auc=L.paired_auc(t1[valid], t0[valid]),
                                  retention=float((t1 - t0)[valid].mean() / (tt1 - tt0)[valid].mean()), obs_mean=float(np.nanmean(obs[valid])),
                                  false_per_img=float(fl.float().mean()), cand_per_img=float(cand.float().mean()),
                                  pnp_sd=params["PnP"][0], pnp_k=params["PnP"][1], tv_lam=float(params["TV"]), I0=a.I0))
                irows.append(pd.DataFrame(dict(model=name, test_set=set_name, sigma_l=sl, contrast=c, method=m, img=np.arange(n),
                                               t_present=t1, t_absent=t0, truth_present=tt1, truth_absent=tt0, obs=obs, valid=valid, false_absent=fl.cpu().numpy())))
            if li == a.example_type:                                       # qualitative figure for this lesion type
                rows = [("truth + lesion", xl, None), ("TV, present", recl["TV"], None), ("PnP, present", recl["PnP-DnCNN"], None),
                        ("truth", x, None), ("TV, absent", rec0["TV"], None), ("PnP, absent", rec0["PnP-DnCNN"], None)]
                fig, axs = plt.subplots(len(rows), 6, figsize=(9, 9.2))
                for j in range(6):
                    for i, (lab, arr, _) in enumerate(rows):
                        ax = axs[i, j]; ax.imshow(arr[j].reshape(28, 28).cpu(), cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
                        ax.add_patch(plt.Circle((centers[j, 1].item(), centers[j, 0].item()), rho + 2.5, fill=False, color="tab:red", lw=.8))
                        if j == 0: ax.set_ylabel(lab, fontsize=8)
                fig.suptitle(f"{name} on {set_name} ({role}) - lesion sigma_l={sl}, contrast {c}; red circle = lesion location", fontsize=9)
                plt.tight_layout(); plt.savefig(f"{a.out}/lesion_{name}_examples_{key}.png", dpi=130); plt.close()
    pd.DataFrame(mrows).to_csv(f_met, index=False); pd.concat(irows).to_csv(f_img, index=False)
    print(f"[{name}] done in {time.time() - tstart:.0f}s", flush=True)
print("\nNext: python summarize_step5.py")
