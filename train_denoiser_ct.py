"""Stage 3 / step 3a: train the noise-conditioned DnCNN prior on LoDoPaB ground-truth slices (patch training).
Usage: python train_denoiser_ct.py [--steps 15000] [--sigma_max 0.1] [--out results/stage3/dncnn_ct.pt]
Uses the extracted training files (python run_stage3_step2.py --extract_train 70 gives 70 x 128 = 8960 slices).
Images are used in the operator's orientation (transposed storage), values in [0,1]; random 128x128 crops, random left-right flips,
Gaussian noise with std ~ U(0, sigma_max) (image units)."""
import argparse, os, time
import numpy as np, torch, torch.nn.functional as F
from src.denoiser import DnCNN
from src.lodopab import LoDoPaB, gt_oriented, one_per_patient

p = argparse.ArgumentParser()
p.add_argument("--root", default="data")
p.add_argument("--train_split", default="train")
p.add_argument("--val_split", default="validation")
p.add_argument("--steps", type=int, default=15000)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--crop", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--sigma_max", type=float, default=0.1)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--ch", type=int, default=64)
p.add_argument("--max_files", type=int, default=0, help="debug: use only the first N hdf5 files")
p.add_argument("--n_val", type=int, default=20)
p.add_argument("--eval_every", type=int, default=1500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results/stage3/dncnn_ct.pt")
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
torch.backends.cudnn.benchmark = True

tr = LoDoPaB(a.root, a.train_split, with_obs=False)
if a.max_files:
    tr.gt_files = tr.gt_files[:a.max_files]; tr.n = min(tr.n, 128 * a.max_files)
n = len(tr)
pats = np.unique(tr.patient_ids[:n]) if tr.patient_ids is not None else []
print(f"device {dev} | training slices {n} from {len(pats)} patients | steps {a.steps}, batch {a.batch}, crop {a.crop}, sigma_max {a.sigma_max}", flush=True)
t = time.time()
X = torch.empty(n, 362, 362, dtype=torch.float16)
for i in range(0, n, 256):
    idx = np.arange(i, min(i + 256, n))
    X[i:i + len(idx)] = torch.tensor(gt_oriented(tr, idx).reshape(len(idx), 362, 362)).half()
print(f"loaded training images into RAM ({X.numel() * 2 / 1e9:.1f} GB float16) in {time.time() - t:.0f}s", flush=True)
vd = LoDoPaB(a.root, a.val_split, with_obs=False)
V = torch.tensor(gt_oriented(vd, one_per_patient(vd, a.n_val, 0))).reshape(-1, 1, 362, 362).to(dev)

model = DnCNN(a.depth, a.ch).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
gv = torch.Generator(device=dev).manual_seed(123)


def evaluate():
    model.eval(); res = []
    with torch.no_grad():
        for s in (0.02, 0.05, 0.1):
            if s > a.sigma_max * 1.0001:
                continue
            m = []
            for b in range(0, len(V), 4):
                v = V[b:b + 4]
                noisy = v + s * torch.randn(v.shape, generator=gv, device=dev)
                out = model(noisy, torch.full((len(v),), s, device=dev)).clamp(0, 1)
                m.append((10 * torch.log10(1 / ((out - v) ** 2).mean((1, 2, 3)))).mean().item())
            res.append((s, float(np.mean(m)), 20 * np.log10(1 / s)))
    model.train()
    return res


model.train(); t0 = time.time(); run = 0.0
for step in range(1, a.steps + 1):
    ii = torch.randint(0, n, (a.batch,))
    oy, ox = np.random.randint(0, 362 - a.crop + 1, 2)
    x = X[ii, oy:oy + a.crop, ox:ox + a.crop].to(dev).float().unsqueeze(1)
    if np.random.rand() < 0.5:
        x = x.flip(-1)
    s = torch.rand(a.batch, device=dev) * a.sigma_max
    noisy = x + s.view(-1, 1, 1, 1) * torch.randn_like(x)
    loss = F.mse_loss(model(noisy, s), x)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    if step == 100:
        per = (time.time() - t0) / 100
        print(f"speed: {1 / per:.1f} steps/s -> projected total {per * a.steps / 60:.0f} min", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        r = evaluate()
        print(f"step {step}/{a.steps} loss {run:.6f} | val PSNR (denoised vs noisy-input) " + ", ".join(f"sigma={s}: {d:.2f} ({n0:.2f})" for s, d, n0 in r)
              + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
        torch.save({"state": model.state_dict(), "config": {"depth": a.depth, "ch": a.ch}, "sigma_max": a.sigma_max, "steps": step,
                    "n_train_slices": n, "n_train_patients": int(len(pats))}, a.out)
print(f"saved {a.out}")
