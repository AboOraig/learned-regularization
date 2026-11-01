"""Non-anatomical test objects for the distribution-shift lesion test: random ellipses, rectangles and a fine stripe/checker block
on a faint disc, values in [0,1] on LoDoPaB's mu/mu_max scale (water ~ 0.25). Nothing like a CT slice, but piecewise constant,
which is the structure a TV prior likes."""
import numpy as np


def shapes_phantom(seed, size=362):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    img = np.zeros((size, size), np.float32)
    c = size / 2
    img[((yy - c) / (0.45 * size)) ** 2 + ((xx - c) / (0.45 * size)) ** 2 <= 1] = 0.12
    for _ in range(rng.integers(7, 12)):
        cy, cx = rng.uniform(0.22, 0.78, 2) * size
        ry, rx = rng.uniform(0.04, 0.16, 2) * size
        v = rng.uniform(0.05, 0.9)
        if rng.random() < 0.5:
            m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        else:
            m = (np.abs(yy - cy) <= ry) & (np.abs(xx - cx) <= rx)
        img[m] = v
    # fine texture block (stripes of period 6 px, or 8 px checker) - high spatial frequency content
    y0, x0 = rng.integers(int(0.2 * size), int(0.55 * size), 2); h = int(0.2 * size)
    blk = np.zeros((h, h), np.float32)
    if rng.random() < 0.5:
        blk[:, (np.arange(h) // 3) % 2 == 0] = 0.7
    else:
        blk[(np.arange(h)[:, None] // 4 + np.arange(h)[None, :] // 4) % 2 == 0] = 0.7
    img[y0:y0 + h, x0:x0 + h] = np.maximum(blk, 0.1)
    return np.clip(img, 0, 1).reshape(-1)
