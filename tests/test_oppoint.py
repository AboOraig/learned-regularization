import numpy as np, pandas as pd, pytest, torch
from src import lesions as L
from src.oppoint import select_rules


def _pts():
    rows = [("PnP", 0.1, 5, np.nan, 24.0, 0.70), ("PnP", 0.05, 25, np.nan, 22.5, 0.95), ("PnP", 0.02, 50, np.nan, 19.5, 0.99),
            ("PnP", 0.05, 5, np.nan, 21.5, 0.90), ("TV", np.nan, 0, 0.01, 20.0, 0.85), ("TV", np.nan, 0, 0.001, 18.0, 0.97)]
    return pd.DataFrame(rows, columns=["method", "sd", "k", "lam", "psnr", "pauc"])


def test_select_rules():
    r = select_rules(_pts(), (0.1, 5), 0.01)
    assert (r["PnP-psnr"]["sd"], r["PnP-psnr"]["k"]) == (0.1, 5) and r["TV-psnr"]["lam"] == 0.01
    assert (r["PnP-lesion"]["sd"], r["PnP-lesion"]["k"]) == (0.02, 50)             # best AUC overall
    assert r["TV-lesion"]["lam"] == 0.001
    c = r["PnP-constrained"]                                                      # best AUC among PnP with PSNR >= TV-psnr (20.0)
    assert (c["sd"], c["k"]) == (0.05, 25) and c["feasible"]


def test_constrained_falls_back_when_infeasible():
    p = _pts(); p.loc[p.method == "PnP", "psnr"] = 10.0
    r = select_rules(p, (0.1, 5), 0.01)
    assert not r["PnP-constrained"]["feasible"] and (r["PnP-constrained"]["sd"], r["PnP-constrained"]["k"]) == (0.1, 5)


def test_arm_stats_perfect_reconstruction():
    x = torch.rand(40, 784) * 0.5
    c = torch.tensor(np.random.default_rng(0).integers(6, 22, size=(40, 2)))
    xl = L.insert_lesion(x, c, 1.25, 0.5)
    s = L.arm_stats(x, xl, c, 1.25, 0.5, rec0=x, recl=xl)
    assert s["pauc"] == 1.0 and s["retention"] == pytest.approx(1.0) and s["false_per_img"] == 0.0 and s["n_valid"] == 40
    s2 = L.arm_stats(x, xl, c, 1.25, 0.5, rec0=x, recl=x)                          # lesion completely erased
    assert s2["retention"] == pytest.approx(0.0, abs=1e-6) and s2["pauc"] == pytest.approx(0.5)
