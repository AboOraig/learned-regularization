"""Datasets: MNIST (train/val/test for the prior) and shifted test sets."""
import numpy as np
import torch
from skimage.transform import resize


def _to_flat(t):
    return (t.float() / 255.0).reshape(len(t), -1)


def load_mnist(root="data", n_val=500, seed=0):
    from torchvision import datasets
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    x = _to_flat(tr.data)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    val, train = x[perm[:n_val]], x[perm[n_val:]]
    return train, val, _to_flat(te.data)


def load_fashion(root="data"):
    from torchvision import datasets
    return _to_flat(datasets.FashionMNIST(root, train=False, download=True).data)


def load_emnist_letters(root="data"):
    from torchvision import datasets
    d = datasets.EMNIST(root, split="letters", train=False, download=True).data
    return _to_flat(d.transpose(1, 2))  # EMNIST images are stored transposed


def shepp_logan(size=28):
    from skimage.data import shepp_logan_phantom
    p = resize(shepp_logan_phantom(), (size, size), anti_aliasing=True)
    p = (p - p.min()) / (p.max() - p.min())
    return torch.tensor(p, dtype=torch.float32).reshape(1, -1)
