"""Operating-point selection rules for the step-5b study (all selection is done on a held-out, in-distribution SELECTION set)."""
import numpy as np
import pandas as pd


def select_rules(pts, tuned_pnp, tuned_tv):
    """pts: one row per operating point on the selection set with columns
         method ('PnP'|'TV'), sd, k, lam (nan where unused), psnr, pauc (mean over lesion types).
       tuned_pnp = (sd, k) and tuned_tv = lam are the PSNR-tuned parameters (they must be present in pts).
       Returns {rule: dict(method, sd, k, lam, feasible)}:
         PnP-psnr / TV-psnr     the PSNR-tuned points (the v1 operating points)
         PnP-lesion / TV-lesion highest mean paired AUC among all points of that method
         PnP-constrained        highest paired AUC among PnP points whose PSNR >= PSNR of the TV-psnr point (keeps PnP's PSNR advantage);
                                falls back to PnP-psnr (feasible=False) if none qualifies. Ties are broken by higher PSNR."""
    pnp, tv = pts[pts.method == "PnP"], pts[pts.method == "TV"]
    pick = lambda d: d.sort_values(["pauc", "psnr"], ascending=False).iloc[0]
    row = lambda r, feasible=True: dict(method=r.method, sd=r.sd, k=r.k, lam=r.lam, feasible=feasible)
    p0 = pnp[(np.isclose(pnp.sd, tuned_pnp[0])) & (pnp.k == tuned_pnp[1])].iloc[0]
    t0 = tv[np.isclose(tv.lam, tuned_tv)].iloc[0]
    ok = pnp[pnp.psnr >= t0.psnr]
    out = {"PnP-psnr": row(p0), "TV-psnr": row(t0), "PnP-lesion": row(pick(pnp)), "TV-lesion": row(pick(tv))}
    out["PnP-constrained"] = row(pick(ok)) if len(ok) else row(p0, feasible=False)
    return out
