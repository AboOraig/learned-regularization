"""Per-condition tuning and reconstruction for the CT experiments (same logic as run_stage3_step4.py, importable):
Hann cut-off, TV weight (150 it), TV-long weight (300 it, tuned separately), PnP-const (sigma_d, k from a TV-free start),
PnP-TVinit (sigma_d, k from the TV solution). Grids extend automatically (up to `rounds` times) when the optimum lands on an edge."""
import torch
from .tv_pdhg import tv_pdhg
from .pnp_ct import pnp_ct


def psnr(r, t):
    return (-10 * torch.log10(((r.clamp(0, 1) - t) ** 2).mean(1).clamp_min(1e-12))).cpu().numpy()


def _edge(b, g):
    return "  [edge of grid]" if b in (min(g), max(g)) else ""


def climb(grid, evalfn, down, up, rounds, label, log=print):
    """Evaluate evalfn on `grid`; while the best value is at an edge and rounds remain, add down(best)/up(best). Returns (best, {value: score})."""
    grid = sorted(grid); sc = {}
    for _ in range(rounds + 1):
        for v in grid:
            if v not in sc:
                sc[v] = evalfn(v)
        b = max(sc, key=sc.get)
        if rounds == 0 or b not in (min(sc), max(sc)):
            break
        nxt = down(b) if b == min(sc) else up(b)
        if nxt is None or nxt in sc:
            break
        grid = sorted(set(grid) | {nxt}); rounds -= 1
    b = max(sc, key=sc.get)
    log(f"{label}: " + ", ".join(f"{v:g}: {sc[v]:.2f}" for v in sorted(sc)) + f" => {b:g}" + _edge(b, list(sc)))
    return b, sc


def _cat(fn, *t, n=10):
    return torch.cat([fn(*[u[b:b + n] for u in t]) for b in range(0, len(t[0]), n)])


def tune(op, den, y, GT, L_, eta, hanns, lams, SD, CK, tv_iters=150, rounds=2, log=print):
    h, _ = climb(hanns, lambda v: psnr(op.fbp(y, "hann", v), GT).mean(), lambda b: round(b * 0.75, 3), lambda b: round(b + 0.1, 3) if b < 0.9 else None, rounds, "Hann cut-off -> val PSNR", log)
    x0 = op.fbp(y, "hann", h).clamp(0, 1)
    tvv = {}
    def ev_tv(l):
        tvv[l] = _cat(lambda yy, xx: tv_pdhg(op, yy, l, tv_iters, xx, L_), y, x0); return psnr(tvv[l], GT).mean()
    lb, _ = climb(lams, ev_tv, lambda b: b / 3, lambda b: b * 3, rounds, "TV lambda -> val PSNR", log)
    ev_tl = lambda l: psnr(_cat(lambda yy, xx: tv_pdhg(op, yy, l, 2 * tv_iters, xx, L_), y, x0), GT).mean()
    ll, _ = climb(list(lams) + [lb], ev_tl, lambda b: b / 3, lambda b: b * 3, rounds, f"TV-long ({2 * tv_iters} it) lambda -> val PSNR", log)
    prm = {"hann": h, "lam": lb, "lam_long": ll}
    for name, init in (("const", x0), ("tvinit", tvv[lb])):
        best = {}
        def ev_sd(sd):
            outs = [pnp_ct(op, y[b:b + 10], den, sd, eta, tuple(CK), init[b:b + 10]) for b in range(0, len(y), 10)]
            sc = {k: psnr(torch.cat([o[k] for o in outs]), GT).mean() for k in CK}
            best[sd] = max(sc, key=sc.get); return sc[best[sd]]
        sdb, _ = climb(SD, ev_sd, lambda b: round(b / 1.6, 4), lambda b: round(b * 1.6, 4), rounds, f"PnP-{name}: sigma_d -> best val PSNR over k", log)
        prm[name] = (sdb, best[sdb])
        log(f"PnP-{name}: sigma_d={sdb:g}, k={best[sdb]}" + _edge(best[sdb], list(CK)))
    return prm


def recon_all(op, den, y, prm, L_, eta, tv_iters=150):
    x0 = op.fbp(y, "hann", prm["hann"]).clamp(0, 1)
    tv = tv_pdhg(op, y, prm["lam"], tv_iters, x0, L_)
    sd, k = prm["const"]; pc = pnp_ct(op, y, den, sd, eta, (k,), x0)[k]
    sd, k = prm["tvinit"]; pt = pnp_ct(op, y, den, sd, eta, (k,), tv)[k]
    return {"FBP-Hann": x0, "TV": tv, "TV-long": tv_pdhg(op, y, prm["lam_long"], 2 * tv_iters, x0, L_), "PnP-const": pc, "PnP-TVinit": pt}
