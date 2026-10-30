"""Train the DnCNN prior. Usage: python train_denoiser.py --epochs 8 [--data mnist|fashion] [--seed 0] [--sigma_max 0.3] [--out ...]
Defaults reproduce v1 (MNIST, seed 0, sigma_max 0.3)."""
import argparse, os, time, torch, torch.nn.functional as F
from src.denoiser import DnCNN
from src.data import load_mnist, load_fashion_train

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--sigma_max", type=float, default=0.3)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--ch", type=int, default=64)
p.add_argument("--out", default="results/dncnn.pt")
p.add_argument("--data", default="mnist", choices=["mnist", "fashion"], help="training distribution of the prior")
p.add_argument("--seed", type=int, default=0, help="init / data order / noise seed")
p.add_argument("--max_train", type=int, default=0, help="debug only: use the first N training images")
a = p.parse_args()

torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
train, val = (load_mnist()[:2] if a.data == "mnist" else load_fashion_train())
if a.max_train:
    train = train[:a.max_train]
train, val = train.reshape(-1, 1, 28, 28).to(dev), val[:200].reshape(-1, 1, 28, 28).to(dev)
model = DnCNN(a.depth, a.ch).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
gv = torch.Generator(device=dev).manual_seed(123)

def val_psnr(s=0.1):
    model.eval()
    with torch.no_grad():
        noisy = val + s * torch.randn(val.shape, generator=gv, device=dev)
        out = model(noisy, torch.full((len(val),), s, device=dev)).clamp(0, 1)
        mse = ((out - val) ** 2).mean((1, 2, 3)); mse0 = ((noisy.clamp(0, 1) - val) ** 2).mean((1, 2, 3))
    return (10 * torch.log10(1 / mse)).mean().item(), (10 * torch.log10(1 / mse0)).mean().item()

for ep in range(a.epochs):
    model.train(); t0 = time.time(); perm = torch.randperm(len(train), device=dev); tot = 0
    for i in range(0, len(train), a.batch):
        x = train[perm[i:i + a.batch]]
        s = torch.rand(len(x), device=dev) * a.sigma_max
        noisy = x + s.view(-1, 1, 1, 1) * torch.randn_like(x)
        loss = F.mse_loss(model(noisy, s), x)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(x)
    sched.step()
    d, n = val_psnr()
    print(f"epoch {ep+1}/{a.epochs} loss={tot/len(train):.5f} val PSNR @sigma=0.1: {d:.2f} dB (noisy input {n:.2f}) [{time.time()-t0:.0f}s]")
    torch.save({"state": model.state_dict(), "config": {"depth": a.depth, "ch": a.ch}}, a.out)
