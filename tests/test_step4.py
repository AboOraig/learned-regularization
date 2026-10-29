"""Step 4 plumbing: split helper, settable sigma_d grid, seeded/--data training script."""
import os, subprocess, sys
import numpy as np, pytest, torch, torch.nn as nn
from src import experiment
from src.data import split_train_val

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_split_is_deterministic_disjoint_and_complete():
    x = torch.arange(1000.0).reshape(-1, 1).repeat(1, 4)
    tr, va = split_train_val(x, n_val=100, seed=0)
    tr2, va2 = split_train_val(x, n_val=100, seed=0)
    assert torch.equal(tr, tr2) and torch.equal(va, va2)
    assert len(tr) == 900 and len(va) == 100
    assert set(tr[:, 0].tolist()).isdisjoint(va[:, 0].tolist())
    assert sorted(tr[:, 0].tolist() + va[:, 0].tolist()) == list(range(1000))
    assert not torch.equal(split_train_val(x, 100, seed=1)[1], va)


def test_set_sd_grid_is_in_place_and_used_by_tune_all():
    from src.operators import RadonOperator, add_noise

    class Shrink(nn.Module):                       # stand-in denoiser: mild shrinkage whose strength grows with sigma
        def forward(self, x, sigma):
            return x * (1 - sigma.view(-1, 1, 1, 1))

    old = list(experiment.SD)
    try:
        alias = experiment.SD
        experiment.set_sd_grid([0.1, 0.4])
        assert alias is experiment.SD and experiment.SD == [0.1, 0.4]            # same list object, new content
        op = RadonOperator(28, 5, device="cpu")
        val = torch.rand(3, 784)
        p = experiment.tune_all(op, Shrink(), val, add_noise(op.forward(val), 0.05, seed=0))
        assert p["PnP"][0] in (0.1, 0.4)
    finally:
        experiment.set_sd_grid(old)
    assert experiment.SD == old


def _train(tmp_path, seed, name):
    out = str(tmp_path / name)
    r = subprocess.run([sys.executable, "train_denoiser.py", "--epochs", "1", "--max_train", "256", "--batch", "64",
                        "--depth", "3", "--ch", "8", "--seed", str(seed), "--out", out], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return torch.load(out, map_location="cpu", weights_only=True)["state"]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "MNIST")), reason="MNIST not available locally")
def test_training_seed_controls_result(tmp_path):
    a, b, c = _train(tmp_path, 0, "a.pt"), _train(tmp_path, 0, "b.pt"), _train(tmp_path, 1, "c.pt")
    same = all(torch.allclose(a[k], b[k], atol=1e-5) for k in a if a[k].dtype.is_floating_point)
    diff = any(not torch.allclose(a[k], c[k], atol=1e-5) for k in a if a[k].dtype.is_floating_point)
    assert same, "same seed must give the same weights (CPU)"
    assert diff, "different seeds must give different weights"
