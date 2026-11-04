"""LoDoPaB-CT access: lazy HDF5 reading + patient ids.  Works on the extracted files
   <root>/ground_truth_<split>/ground_truth_<split>_NNN.hdf5, observation_<split>/observation_<split>_NNN.hdf5
(or any folder layout below <root>), plus patient_ids_rand_<split>.csv.  Splits: train, validation, test.
Every file stores 128 slices (the last one fewer valid ones); the number of valid slices comes from the csv / known sizes."""
import glob, os, re, zipfile
import numpy as np

SIZES = {"train": 35820, "validation": 3522, "test": 3553}
PER_FILE = 128
IMG, N_ANGLES, N_DET = 362, 1000, 513


def extract_zips(root):
    """Extract every *.zip under root into root/<zip stem>/ if not yet extracted. Returns list of folders."""
    out = []
    for z in sorted(glob.glob(os.path.join(root, "*.zip"))):
        dest = os.path.join(root, os.path.splitext(os.path.basename(z))[0])
        if not os.path.isdir(dest) or not glob.glob(os.path.join(dest, "**", "*.hdf5"), recursive=True):
            print(f"extracting {z} -> {dest}", flush=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
        out.append(dest)
    return out


def _files(root, kind, split):
    fs = glob.glob(os.path.join(root, "**", f"{kind}_{split}_*.hdf5"), recursive=True)
    return sorted(fs, key=lambda f: int(re.findall(r"_(\d+)\.hdf5$", f)[0]))


def read_patient_ids(root, split):
    p = glob.glob(os.path.join(root, "**", f"patient_ids_rand_{split}.csv"), recursive=True)
    if not p:
        return None
    ids = []
    for line in open(p[0]):
        tok = line.strip().split(",")[0].strip()
        if tok.lstrip("-").isdigit():
            ids.append(int(tok))
    return np.asarray(ids)


class LoDoPaB:
    def __init__(self, root, split, with_obs=True):
        import h5py
        self.h5py, self.root, self.split = h5py, root, split
        self.gt_files = _files(root, "ground_truth", split)
        self.obs_files = _files(root, "observation", split) if with_obs else []
        if not self.gt_files:
            raise FileNotFoundError(f"no ground_truth_{split}_*.hdf5 under {root}; run: python run_stage3_step1.py --extract")
        self.patient_ids = read_patient_ids(root, split)
        self.n = len(self.patient_ids) if self.patient_ids is not None else SIZES[split]
        cap = PER_FILE * len(self.gt_files)
        self.n = min(self.n, cap)
        self._open = {}

    def __len__(self):
        return self.n

    def _read(self, files, idx):
        idx = np.asarray(idx); out = []
        for i in idx:
            if not 0 <= i < self.n:
                raise IndexError(i)
            f, r = divmod(int(i), PER_FILE)
            key = (id(files), f)
            if key not in self._open:
                h = self.h5py.File(files[f], "r"); self._open[key] = (h, h[list(h.keys())[0]])
            out.append(np.asarray(self._open[key][1][r]))
        return np.stack(out).astype(np.float32)

    def gt(self, idx):
        """(len(idx), 362, 362) float32"""
        return self._read(self.gt_files, idx)

    def obs(self, idx):
        """(len(idx), 1000, 513) float32 observations (log-transformed, normalised line integrals)"""
        if not self.obs_files:
            raise FileNotFoundError("observation files not loaded/found")
        return self._read(self.obs_files, idx)

    def close(self):
        for h, _ in self._open.values():
            h.close()
        self._open = {}


# ------------------------------------------------------------------ conventions found in step 1 (geometry check against the real observations)
# rotation sign +1, no detector flip, stored image transposed, angles at the mid-points of 1000 cells over [0, 180) deg,
# pixel size = 0.26 m / 362 x fitted scale 1.0044
PIXEL_SIZE = 0.26 / 362 * 1.0044


def make_operator(n_angles=1000, device="cpu", **kw):
    """Parallel-beam operator for LoDoPaB with n_angles of the 1000 angles (evenly spaced subset).
    Returns (operator, indices of the used angles in the 1000-angle observation)."""
    from .radon_pb import ParallelBeamOperator
    idx = np.round(np.linspace(0, N_ANGLES, n_angles, endpoint=False)).astype(int)
    ang = (idx + 0.5) * 180.0 / N_ANGLES
    op = ParallelBeamOperator(IMG, N_DET, angles_deg=ang, pixel_size=PIXEL_SIZE, sign=1.0, flip_det=False, device=device, **kw)
    return op, idx


def gt_oriented(ds, idx):
    """Ground truth in the operator's orientation, flattened: (len(idx), 362*362) float32."""
    return np.ascontiguousarray(ds.gt(idx).transpose(0, 2, 1)).reshape(len(idx), -1)


def one_per_patient(ds, n, seed=0):
    """Indices of n slices, one random slice from each of n randomly chosen patients (independent units for statistics)."""
    rng = np.random.default_rng(seed)
    pats = np.unique(ds.patient_ids[:len(ds)])
    rng.shuffle(pats)
    out = []
    for p in pats[:n]:
        out.append(int(rng.choice(np.where(ds.patient_ids[:len(ds)] == p)[0])))
    return np.asarray(out)


def extract_partial(root, stem="ground_truth_train", n_files=70):
    """Extract only the first n_files hdf5 members of root/<stem>.zip into root/<stem>/."""
    z = os.path.join(root, stem + ".zip")
    dest = os.path.join(root, stem)
    with zipfile.ZipFile(z) as zf:
        names = sorted([n for n in zf.namelist() if n.endswith(".hdf5")],
                       key=lambda n: int(re.findall(r"_(\d+)\.hdf5$", n)[0]))[:n_files]
        for k, n in enumerate(names):
            if not os.path.exists(os.path.join(dest, n)):
                zf.extract(n, dest)
            if (k + 1) % 10 == 0:
                print(f"  extracted {k + 1}/{len(names)}", flush=True)
    return dest, len(names)
