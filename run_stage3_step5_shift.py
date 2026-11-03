"""Stage 3 / step 5b: does the distribution-shift result (step 4c, K=60 only) hold at other angle counts?
For each condition (K, dose) this reuses the parameters tuned in step 5 on LoDoPaB validation slices (results/stage3/s5/step5_params_K<K>_dose<f>.json;
NO retuning on the phantoms) and runs the same lesion experiment as step 5 on --n_shapes random non-anatomical phantoms (ellipses / rectangles /
stripe block; the same phantoms as step 4/4c): lesion-free PSNR / SSIM, location-known detection scores (sigma_l in {2,4}, contrast 0.06, M lesions per
phantom inside the central square) and false-blob counts at 0.5 / 1 / 1.5 x the nominal lesion response.
Output: <out>/step5shift_K<K>_dose<f>.npz (same keys as step5_*.npz; resumable).
Usage: python run_stage3_step5_shift.py                      (K = 30,120 x dose = 1,0.25)
       python run_stage3_step5_shift.py --angles 120 --doses 1
Statistics (shift vs in-distribution): python run_stage3_step5_shift_summary.py"""
import argparse, os, time, platform, json
import numpy as np, torch
from skimage.metrics import structural_similarity
from src.lodopab import make_operator, IMG
from src.denoiser import load_denoiser
from src.ct_sim import CTSimulator, poisson_obs
from src.phantoms import shapes_phantom
from src.tune_ct import recon_all, psnr
from src import lesions as L
from src import lesions_ct as C

p = argparse.ArgumentParser()
p.add_argument("--params", default="results/stage3/s5"); p.add_argument("--out", default="results/stage3/s5/shift")
p.add_argument("--model", default="results/stage3/dncnn_ct.pt")
p.add_argument("--doses", default="1,0.25"); p.add_argument("--angles", default="30,120")
p.add_argument("--n_shapes", type=int, default=20); p.add_argument("--M", type=int, default=4)
p.add_argument("--sigmas", default="2,4"); p.add_argument("--contrast", type=float, default=0.06)
p.add_argument("--tv_iters", type=int, default=150); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
print(f"torch {torch.__version__} | python {platform.python_version()} | device {dev}", flush=True)
fl = lambda s: [float(v) for v in s.split(",")]
SIG, c0 = fl(a.sigmas), a.contrast; MULT = (0.5, 1.0, 1.5); MKEY = {0.5: "fl", 1.0: "fl1", 1.5: "fl1.5"}
METHODS = ["FBP-Hann", "TV", "TV-long", "PnP-const", "PnP-TVinit"]
den = load_denoiser(a.model, dev)
X = torch.tensor(np.stack([shapes_phantom(1000 + i) for i in range(a.n_shapes)]))
cen, val = C.sample_centers([C.square_mask()] * a.n_shapes, a.M, 10 * max(SIG) + 4, seed=a.seed + 5)
NOM = {s: L.nominal_response(s, c0, IMG) for s in SIG}
ssim = lambda r, t: np.array([structural_similarity(r_.clamp(0, 1).reshape(IMG, IMG).cpu().numpy(), t_.reshape(IMG, IMG).cpu().numpy(), data_range=1.0) for r_, t_ in zip(r, t)])
OPS = {}
for K in [int(v) for v in a.angles.split(",")]:
    for f in fl(a.doses):
        tag = f"K{K}_dose{f:g}"; fn = f"{a.out}/step5shift_{tag}.npz"; pf = f"{a.params}/step5_params_{tag}.json"
        if os.path.exists(fn):
            print(f"[{tag}] done, skipping", flush=True); continue
        if not os.path.exists(pf):
            raise SystemExit(f"missing {pf}: run run_stage3_step5.py for this condition first")
        prm = json.load(open(pf)); print(f"\n=== shift condition K={K}, dose={f:g}; parameters from step 5: {prm}", flush=True)
        if K not in OPS:
            OPS[K] = make_operator(K, dev)
        op, ai = OPS[K]; L_ = op.lipschitz(); eta = 1.0 / L_; sim = CTSimulator(op, dev)
        R = {}; put = lambda k, v: R.setdefault(k, []).append(v); t1 = time.time()
        for b in range(0, len(X), 5):
            x = X[b:b + 5].to(dev); cn, vl = cen[b:b + 5], val[b:b + 5]
            rec_a = recon_all(op, den, poisson_obs(sim.clean(x), f, seed=1000 * b + 1), prm, L_, eta, a.tv_iters)
            for m, r in rec_a.items():
                put(f"psnr|{m}", psnr(r, x)); put(f"ssim|{m}", ssim(r, x))
            for j, s in enumerate(SIG):
                rho, nom = L.lesion_rho(s), NOM[s]
                xl = C.insert_multi(x, cn, vl, s, c0, IMG)
                rec_l = recon_all(op, den, poisson_obs(sim.clean(xl), f, seed=1000 * b + 2 + j), prm, L_, eta, a.tv_iters)
                tt1, tt0 = C.responses(xl, rho, cn, IMG).cpu().numpy(), C.responses(x, rho, cn, IMG).cpu().numpy()
                put(f"valid|{s:g}|{c0:g}", vl.numpy() & ((tt1 - tt0) >= 0.5 * nom)); put(f"truth|{s:g}|{c0:g}", tt1 - tt0)
                for m in METHODS:
                    put(f"t1|{s:g}|{c0:g}|{m}", C.responses(rec_l[m], rho, cn, IMG).cpu().numpy())
                    put(f"t0|{s:g}|{c0:g}|{m}", C.responses(rec_a[m], rho, cn, IMG).cpu().numpy())
                    Rm, un = C.unsupported_map(rec_a[m], x, rho, nom, size=IMG)
                    for mu in MULT:
                        put(f"{MKEY[mu]}|{s:g}|{c0:g}|{m}", ((Rm >= mu * nom) & un).sum((1, 2)).cpu().numpy())
            d = min(b + 5, len(X)); el = time.time() - t1
            print(f"  [{tag}] {d}/{len(X)} phantoms, {el / 60:.1f} min, ETA {el / d * (len(X) - d) / 60:.1f} min", flush=True)
        R = {k: np.concatenate(v) for k, v in R.items()}
        np.savez(fn, **R)
        print(f"  [{tag}] phantom PSNR / SSIM: " + ", ".join(f"{m} {R[f'psnr|{m}'].mean():.2f}/{R[f'ssim|{m}'].mean():.3f}" for m in METHODS), flush=True)
print("\ndone; now run: python run_stage3_step5_shift_summary.py")
