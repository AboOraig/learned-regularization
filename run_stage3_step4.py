"""Stage 3 / step 4: do the learned prior and TV create convincing but wrong structures on real CT slices, and under a shift?
Per dose f (fraction of LoDoPaB's 4096 photons) and K angles, for each of two image sets
   ct      real LoDoPaB test slices (one per patient)                         -> in-distribution
   shapes  random ellipse/rectangle/stripe phantoms (not anatomical)           -> shift
M well separated Gaussian lesions of one type are inserted (additive, clipped) into each slice; four types
sigma_l in {2,4} px x contrast in {0.03,0.06} (mu/mu_max units; with water ~ 0.25 this is roughly 12 / 24 % of water, i.e. ~ 120 / 240 HU if
water ~ 1000 HU-scale, an ASSUMPTION about the mapping). Truth is simulated on a 2x finer grid with bicubic up-sampling (no inverse crime) with
LoDoPaB's Poisson model; the lesion-free and lesion-present arms get independent noise. Methods: FBP-Hann, TV, TV-long (2x iterations, same
weight; control for PnP's extra iterations), PnP-const, PnP-TVinit. Tuned on lesion-free VALIDATION slices (real LoDoPaB observations).
Reported per image set / method / lesion type: location-known paired AUC, retention, false lesions per lesion-free image (all with 95% bootstrap
intervals over slices) and PSNR on the lesion-free arm.
Usage: python run_stage3_step4.py --doses 1            (then --doses 0.25)
Outputs (results/stage3): step4_lesion_summary.csv, step4_perslice.npz-like files step4_<set>_dose<f>.npz (resumable)"""
import argparse, os, time, platform
import numpy as np, pandas as pd, torch
from src.lodopab import LoDoPaB, make_operator, gt_oriented, one_per_patient, IMG
from src.lodopab_noise import thin_obs
from src.tv_pdhg import tv_pdhg
from src.denoiser import load_denoiser
from src.pnp_ct import pnp_ct
from src.ct_sim import CTSimulator, poisson_obs
from src.phantoms import shapes_phantom
from src import lesions as L
from src import lesions_ct as C

p = argparse.ArgumentParser()
p.add_argument("--root", default="data"); p.add_argument("--out", default="results/stage3")
p.add_argument("--model", default="results/stage3/dncnn_ct.pt")
p.add_argument("--doses", default="1"); p.add_argument("--angles", type=int, default=60)
p.add_argument("--n_ct", type=int, default=40); p.add_argument("--n_shapes", type=int, default=20)
p.add_argument("--n_val", type=int, default=10); p.add_argument("--M", type=int, default=4)
p.add_argument("--sigmas", default="2,4"); p.add_argument("--contrasts", default="0.03,0.06")
p.add_argument("--tv_iters", type=int, default=150)
p.add_argument("--tv_lams", default="3e-3,1e-2,3e-2"); p.add_argument("--hann", default="0.16,0.2,0.3,0.4")
p.add_argument("--sd_grid", default="0.02,0.03,0.05"); p.add_argument("--ck", default="4,8,12,20,30")
p.add_argument("--n_boot", type=int, default=2000); p.add_argument("--seed", type=int, default=0)
p.add_argument("--fidelity", type=int, default=1, help="also reconstruct REAL observations of the same ct slices with TV (simulator check)")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
print(f"torch {torch.__version__} | python {platform.python_version()} | device {dev}", flush=True)
fl = lambda s: [float(v) for v in s.split(",")]
lams, hanns, SD, CK = fl(a.tv_lams), fl(a.hann), fl(a.sd_grid), tuple(int(v) for v in a.ck.split(","))
TYPES = [(s, c) for s in fl(a.sigmas) for c in fl(a.contrasts)]
den = load_denoiser(a.model, dev)
psnr = lambda r, t: (-10 * torch.log10(((r.clamp(0, 1) - t) ** 2).mean(1).clamp_min(1e-12))).cpu().numpy()
edge = lambda b, g: "  [edge of grid]" if b in (g[0], g[-1]) else ""
dv, dt = LoDoPaB(a.root, "validation"), LoDoPaB(a.root, "test")
op, ai = make_operator(a.angles, dev); L_ = op.lipschitz(); eta = 1.0 / L_
sim = CTSimulator(op, dev)
iv, it = one_per_patient(dv, a.n_val, a.seed), one_per_patient(dt, a.n_ct, a.seed + 1)   # it = same slices as steps 2-3b when n_ct = 40
GTv = torch.tensor(gt_oriented(dv, iv)).to(dev); OBSv = torch.tensor(dv.obs(iv))
GTt = torch.tensor(gt_oriented(dt, it)).to(dev)
cat = lambda fn, *t: torch.cat([fn(*[u[b:b + 10] for u in t]) for b in range(0, len(t[0]), 10)])
rho_of = {s: L.lesion_rho(s) for s, _ in TYPES}
NOM = {(s, c): L.nominal_response(s, c, IMG) for s, c in TYPES}
dmin = 10 * max(s for s, _ in TYPES) + 4


def tune(f):
    y = thin_obs(OBSv, f, seed=a.seed + 11)[:, ai, :].reshape(len(iv), -1).to(dev)
    sc = {h: psnr(op.fbp(y, "hann", h), GTv).mean() for h in hanns}; hb = max(sc, key=sc.get)
    print("Hann -> val PSNR: " + ", ".join(f"{h:g}: {v:.2f}" for h, v in sc.items()) + f" => {hb:g}" + edge(hb, hanns), flush=True)
    x0 = op.fbp(y, "hann", hb).clamp(0, 1)
    tvv = {l: cat(lambda yy, xx: tv_pdhg(op, yy, l, a.tv_iters, xx, L_), y, x0) for l in lams}
    ts = {l: psnr(v, GTv).mean() for l, v in tvv.items()}; lb = max(ts, key=ts.get)
    print("TV lambda -> val PSNR: " + ", ".join(f"{l:g}: {v:.2f}" for l, v in ts.items()) + f" => {lb:g}" + edge(lb, lams), flush=True)
    prm = {"hann": hb, "lam": lb}
    for name, init in (("const", x0), ("tvinit", tvv[lb])):
        best = (-1, None)
        for sd in SD:
            outs = [pnp_ct(op, y[b:b + 10], den, sd, eta, CK, init[b:b + 10]) for b in range(0, len(y), 10)]
            for k in CK:
                s_ = psnr(torch.cat([o[k] for o in outs]), GTv).mean()
                if s_ > best[0]:
                    best = (s_, (sd, k))
        prm[name] = best[1]
        print(f"PnP-{name}: val PSNR {best[0]:.2f} at sigma_d={best[1][0]:g}, k={best[1][1]}" + edge(best[1][0], SD) + edge(best[1][1], list(CK)), flush=True)
    return prm


def recon_all(y, prm):
    x0 = op.fbp(y, "hann", prm["hann"]).clamp(0, 1)
    tv = tv_pdhg(op, y, prm["lam"], a.tv_iters, x0, L_)
    sd, k = prm["const"]; pc = pnp_ct(op, y, den, sd, eta, (k,), x0)[k]
    sd, k = prm["tvinit"]; pt = pnp_ct(op, y, den, sd, eta, (k,), tv)[k]
    return {"FBP-Hann": x0, "TV": tv, "TV-long": tv_pdhg(op, y, prm["lam"], 2 * a.tv_iters, x0, L_), "PnP-const": pc, "PnP-TVinit": pt}


def run_set(name, X, masks, f, prm):
    fn = f"{a.out}/step4_{name}_dose{f:g}.npz"
    if os.path.exists(fn):
        print(f"[{name}] {fn} exists, skipping", flush=True); return dict(np.load(fn, allow_pickle=True))
    S = len(X); cen, val = C.sample_centers(masks, a.M, dmin, seed=a.seed + 5)
    R = {}                                                  # key -> list of arrays over chunks
    put = lambda k, v: R.setdefault(k, []).append(v)
    t0 = time.time()
    for b in range(0, S, 5):
        x, cn, vl = X[b:b + 5].to(dev), cen[b:b + 5], val[b:b + 5]
        ya = poisson_obs(sim.clean(x), f, seed=1000 * b + 1)
        rec_a = recon_all(ya, prm)
        for m, r in rec_a.items():
            put(f"psnr|{m}", psnr(r, x))
        for j, (s, c) in enumerate(TYPES):
            xl = C.insert_multi(x, cn, vl, s, c, IMG)
            yl = poisson_obs(sim.clean(xl), f, seed=1000 * b + 2 + j)
            rec_l = recon_all(yl, prm)
            rho = rho_of[(s, c)[0]]; th = 0.5 * NOM[(s, c)]
            tt1, tt0 = C.responses(xl, rho, cn, IMG).cpu().numpy(), C.responses(x, rho, cn, IMG).cpu().numpy()
            ok = vl.numpy() & ((tt1 - tt0) >= 0.5 * NOM[(s, c)])
            put(f"valid|{s:g}|{c:g}", ok); put(f"truth|{s:g}|{c:g}", tt1 - tt0)
            for m in rec_l:
                put(f"t1|{s:g}|{c:g}|{m}", C.responses(rec_l[m], rho, cn, IMG).cpu().numpy())
                put(f"t0|{s:g}|{c:g}|{m}", C.responses(rec_a[m], rho, cn, IMG).cpu().numpy())
                fls, _ = L.count_false_lesions(rec_a[m], x, rho, th, size=IMG); put(f"fl|{s:g}|{c:g}|{m}", fls.cpu().numpy())
        done = min(b + 5, S)
        print(f"  [{name}] {done}/{S} slices, {(time.time() - t0) / 60:.1f} min, ETA {(time.time() - t0) / done * (S - done) / 60:.1f} min", flush=True)
    R = {k: np.concatenate(v) for k, v in R.items()}
    np.savez(fn, **R)
    return R


def summarise(name, R, f, rows):
    ms = ["FBP-Hann", "TV", "TV-long", "PnP-const", "PnP-TVinit"]
    for s, c in TYPES:
        ok = R[f"valid|{s:g}|{c:g}"]; tr = R[f"truth|{s:g}|{c:g}"]
        for m in ms:
            t1, t0, fl_ = R[f"t1|{s:g}|{c:g}|{m}"], R[f"t0|{s:g}|{c:g}|{m}"], R[f"fl|{s:g}|{c:g}|{m}"]
            Sn = len(t1)
            auc_f = lambda i: C.cluster_paired_auc(t1, t0, ok, i)
            ret_f = lambda i: float(((t1 - t0)[i] * ok[i]).sum() / (tr[i] * ok[i]).sum())
            fl_f = lambda i: float(fl_[i].mean())
            row = dict(set=name, dose=f, K=a.angles, sigma_l=s, contrast=c, method=m, n_slices=Sn, n_lesions=int(ok.sum()),
                       psnr_absent=float(R[f"psnr|{m}"].mean()))
            allv = np.arange(Sn)
            for key, fn_ in (("auc", auc_f), ("retention", ret_f), ("false_per_img", fl_f)):
                lo, hi = C.cluster_boot(fn_, Sn, a.n_boot, 1)
                row.update({key: fn_(allv), key + "_lo": lo, key + "_hi": hi})
            rows.append(row)


rows = []
for f in fl(a.doses):
    print(f"\n=== dose {f:g}, K={a.angles} ===", flush=True)
    prm = tune(f); print("parameters:", prm, flush=True)
    # ---- ct slices
    masks = [C.body_mask(g.reshape(IMG, IMG)) for g in GTt.cpu().numpy()]
    R = run_set("ct", GTt.cpu(), masks, f, prm); summarise("ct", R, f, rows)
    if a.fidelity:
        yr = thin_obs(torch.tensor(dt.obs(it)), f, seed=a.seed + 12)[:, ai, :].reshape(len(it), -1).to(dev)
        x0 = op.fbp(yr, "hann", prm["hann"]).clamp(0, 1)
        tvr = cat(lambda yy, xx: tv_pdhg(op, yy, prm["lam"], a.tv_iters, xx, L_), yr, x0)
        print(f"SIMULATOR CHECK (same {len(it)} slices): TV PSNR real observations {psnr(tvr, GTt).mean():.2f} dB vs simulated {R['psnr|TV'].mean():.2f} dB; "
              f"Hann FBP real {psnr(x0, GTt).mean():.2f} vs simulated {R['psnr|FBP-Hann'].mean():.2f}", flush=True)
    # ---- shapes phantoms (shift)
    Xs = torch.tensor(np.stack([shapes_phantom(1000 + i) for i in range(a.n_shapes)]))
    R = run_set("shapes", Xs, [C.square_mask()] * a.n_shapes, f, prm); summarise("shapes", R, f, rows)
D = pd.DataFrame(rows); fn = f"{a.out}/step4_lesion_summary.csv"
if os.path.exists(fn):                                      # keep earlier doses when run in pieces
    D = pd.concat([pd.read_csv(fn).query("dose not in @D.dose.unique()"), D])
D.to_csv(fn, index=False)
pd.set_option("display.width", 250)
print("\n=== lesion detection / false lesions (95% bootstrap over slices) ===")
print(D.round(3)[["set", "dose", "sigma_l", "contrast", "method", "n_lesions", "auc", "auc_lo", "auc_hi", "retention", "false_per_img", "false_per_img_lo", "false_per_img_hi", "psnr_absent"]].to_string(index=False))
