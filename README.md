# Learned Regularization for Robust Image Reconstruction

Sparse-view CT reconstruction studied as an inverse problem under noise,
incomplete measurements and forward-model uncertainty. A learned image prior
(a noise-conditioned DnCNN denoiser) is plugged into an optimization-based
reconstruction via Plug-and-Play proximal gradient (PnP-PGD) and compared with
classical regularization (FBP, Tikhonov, TV). The study has three stages:
28x28 MNIST-derived images (Stage 1), a matrix-free operator with Poisson noise
(Stage 2), and real clinical-size CT slices from LoDoPaB-CT (Stage 3).

**Start here:** [REPORT.md](REPORT.md) has every experiment, table and caveat;
the technical paper is [paper/Learned_Regularization.pdf](./paper/Learned_Regularization.pdf).
This README is the summary and the map of the repository.

## Three questions

1. **Distribution change.** Does the learned prior keep its advantage over TV when the image distribution changes?
2. **False structures.** Can we find cases where the method creates convincing but incorrect structures?
3. **Robustness.** Do the conclusions hold over a range of noise levels and reconstruction conditions?

## Headline results

| | Stage 1-2: MNIST-derived 28x28 images | Stage 3: real CT (LoDoPaB, 362x362, 40 test patients) |
|---|---|---|
| **Accuracy vs TV** | PnP beats TV by up to +3.85 dB when the problem is noisy or under-determined (sigma >= 0.05, few angles); TV wins when noise-free (-6.4 dB). | PnP does **not** beat a well-tuned TV: PnP-const minus TV-long is -0.93 to +0.33 dB over 9 conditions (3 significantly positive, 4 significantly negative); SSIM is lower in 8 of 9 (7 significantly). The best variant (PnP started from the TV solution) has the highest mean PSNR in 7 of 9 conditions, by at most 0.49 dB. |
| **Q1: distribution change** | Advantage over TV: MNIST +3.7, EMNIST letters +1.9, Fashion-MNIST **-3.1**, Shepp-Logan **-4.2** dB. A prior trained on Fashion-MNIST gives +2.2 dB on Fashion: the reversal is prior mismatch. | Shift to non-anatomical phantoms: PSNR below TV-long in 6 of 6 conditions, SSIM in 4 of 4. The detection advantage over TV-long is retained in 1 of 6 conditions; with 20 phantoms we cannot say it is lost, only that it is not reliably transferred. |
| **Q2: false structures** | MNIST prior on Fashion-MNIST: 0.20-0.66 false lesions per image (TV <= 0.20; matched prior <= 0.01). | Real slices: significantly fewer false blobs than FBP-Hann in all four conditions examined. Phantoms at dose 0.25: more than FBP-Hann (+1.0, not significant, and +4.0 significant). Large false blobs on real slices also appear in FBP and TV. No reader study. |
| **Q3: robustness** | The sign of PnP minus TV depends on noise level, angle count and forward-model error (a 4 degree angle offset costs PnP 4.0 dB vs 1.6 dB for TV). | Detection advantage over TV-long has the same sign in 9 of 9 dose x angle conditions (significant in 6). PSNR, SSIM and false-blob signs change with dose and angle count without a monotone trend. |
| **Detection of small lesions** | PnP detects *worse* than TV (paired AUC below TV in nearly every condition; PnP attenuates lesions). | PnP detects *better* than TV-long (AUC +0.03 to +0.16, positive in 9 of 9, significant in 6), but never significantly better than plain FBP-Hann. TV tuned for PSNR is worse than FBP-Hann in 9 of 9 (significant in 7) while gaining 0.8-2.3 dB PSNR. |

**Reading across stages.** The "+4 dB over TV" belongs to 28x28 digits with a
matched prior and does not carry over to clinical-size CT. The detection
ranking between PnP and TV is reversed between the stages (digits vs CT);
we report this as an observation and did not isolate the cause. PSNR and
detectability rank methods in opposite order for TV on CT, so PSNR alone is
not a safe selection criterion.

**Other findings.** PnP with an unconstrained CNN denoiser does not converge
to a fixed point that is both data-consistent and maximally accurate: PSNR
peaks after about 5-15 iterations and then decays while the data residual
rises, so early stopping is part of the regularization (tuned: 5-25
iterations for >= 10 angles and sigma <= 0.2; 100-400 for 5 angles). On
CT, a deeper denoiser (17 layers) did not help.

![Reconstruction accuracy vs. noise level (Stage 1)](results/comparison_noise.png)

## How much to trust it

- Stage 3 uses 40 test patients (one slice each), one noise realisation per
  condition and no multiplicity correction; read sign counts across
  conditions, not single intervals.
- The Stage 3 analysis was developed in steps after seeing results. Nothing was
  pre-specified, and an early K=60 comparison (PnP 0.85 dB below TV) was
  retracted after re-tuning on 20 validation slices. See "Analysis history" in REPORT.md.
- One denoiser architecture and one training run (8,960 of 35,820 training
  slices); TV weight grid with steps of a factor 3; two PnP-TVinit optima on a grid edge.
- Lesions are additive Gaussian blobs (120-240 HU under an assumed
  mu/mu_max-to-HU mapping) with a location-known observer. "Convincing" means a
  lesion-like response at the kernel scale; there was no reader study.
- Stage 1-2 limits: 28x28 images, inverse crime in E1-E5 (E6-E10 use finer-grid data),
  three denoisers per configuration in E8-E10 and one in E1-E5, moderate mismatch
  levels, Shepp-Logan is a single image.
- Statistics differ by stage: paired Wilcoxon tests in Stages 1-2; paired
  differences with a 95% bootstrap over patients in Stage 3.

Full limitations: REPORT.md, "Limitations" sections for each stage.

## Method

Sparse-view CT is a linear inverse problem `y = A x + n`. PnP-PGD iterates

```
x_{k+1} = D_sigma_d( x_k - eta * A^T(A x_k - y) ),   eta = 1 / L(A^T A)
```

where `D_sigma_d` is a noise-conditioned DnCNN trained once (on MNIST in
Stages 1-2, on LoDoPaB slices in Stage 3). Stage 1 builds `A` as an explicit
dense matrix; Stages 2-3 use a matrix-free projector with an exact adjoint
(autograd transpose). Noise is Gaussian in Stage 1 and Poisson (Beer-Lambert
dose model) in Stages 2-3. All hyperparameters (Hann cut-off, Tikhonov and TV
weights, PnP sigma_d and iteration count) are tuned by grid search on a
held-out validation split and evaluated on a disjoint test split; the
robustness experiments (E3, E4 and the Stage 3 shift study) freeze the tuned
values and do not retune.

## Repository structure

```
src/
  operators.py    Sparse-view Radon operator (dense, exact adjoint, angle mismatch)
  radon_mf.py     MatrixFreeRadon: matrix-free projector, exact adjoint, FBP
  baselines.py    Tikhonov (closed form) and TV (ADMM)
  mf_solvers.py   CG Tikhonov, CG-based TV-ADMM, Poisson sinograms, randomized SVD
  mf_tune.py      Tuning and reconstruction with the matrix-free operator
  denoiser.py     Noise-conditioned DnCNN
  pnp.py          PnP proximal-gradient solver
  experiment.py   Shared hyperparameter grids + tune/reconstruct helpers
  metrics.py      PSNR, SSIM, relative error
  data.py         MNIST / Fashion-MNIST / EMNIST / Shepp-Logan loaders

train_denoiser.py        Train the Stage 1 DnCNN prior (MNIST)
train_denoiser_ct.py     Train the Stage 3 DnCNN prior (LoDoPaB)
run_*.py, make_*.py      Experiment and figure scripts (tables below)
tests/                   68 tests: python -m pytest tests -q
results/                 CSVs, PNGs, summary tables (generated)
REPORT.md                Full write-up: methods, all results, limitations
paper/                   Technical paper
```

## Setup

```
pip install -r requirements.txt
pip install h5py          # Stage 3 only (LoDoPaB-CT zip files go in data/)
```

## Reproducing

Stage 1 (about minutes on a GPU):

```
python train_denoiser.py --epochs 8
python run_baselines.py --exp noise && python run_baselines.py --exp angles
python run_pnp.py --exp noise && python run_pnp.py --exp angles
python make_plots.py --exp noise && python make_plots.py --exp angles
python run_robustness.py --exp mismatch && python run_robustness.py --exp shift
python make_figures.py
python make_summary.py
```

Smoke test with no download (synthetic shapes instead of MNIST):

```
python run_baselines.py --exp noise --synthetic --n_val 30 --n_test 50
```

Stage 2 and Stage 3: see the script tables below; the exact Stage 3 command
sequence (with arguments) is listed at the end of the Stage 3 part of
REPORT.md. The Stage 3 dose x angle grid (`run_stage3_step5.py`) is resumable
and takes hours on a 4 GB GPU.

## Scripts by stage

### Stage 1 extensions (branch stage1-hallucination-metrics)

| Script | What it does | Main outputs (results/) |
|---|---|---|
| run_nullspace.py [--fine_factor 5 --up_mode bicubic --suffix _fine5c] | error decomposition along the SVD of A; noise / angle / shift sweeps; optional finer-grid data | nullspace_*.csv, nullspace_*.png |
| compare_crime.py --suffix _fine5c | inverse crime vs crime-free comparison | crime_comparison.csv/.png |
| run_spectral.py | iteration and sigma_d control, spectral bands | iteration_control*.csv, spectral_profile*.png |
| run_step4.py + summarize_step4.py | 3 seeds x {MNIST sigma_max 0.3, MNIST sigma_max 0.5, Fashion} denoisers, matched-prior control | step4/step4_summary.csv/.png |
| run_lesions.py + summarize_step5.py | lesion-insertion test (paired AUC, retention, false lesions) | step5/step5_summary.csv/.png, lesion_*_examples_*.png |
| run_oppoint.py | operating-point study (PSNR vs lesion detectability) | step5b/oppoint_*.csv/.png |

### Stage 2 extensions (branch stage2-matrix-free)

| File | What it does | Main outputs (results/) |
|---|---|---|
| src/radon_mf.py | `MatrixFreeRadon`: matrix-free projector, exact adjoint, Lipschitz estimate, FBP; matches the dense `RadonOperator` to ~3e-7 | - |
| src/mf_solvers.py | CG Tikhonov, CG-based TV-ADMM, Poisson (Beer-Lambert) sinograms, Poisson-weighted PnP, randomized SVD, strong/weak error split | - |
| src/mf_tune.py | tuning and reconstruction with the matrix-free operator | - |
| run_stage2.py --exp equiv | dense vs matrix-free comparison (accuracy, time, GPU memory) | console |
| run_stage2.py --exp dose | Poisson dose sweep I0 = 1000...10 photons/ray, 4 test sets, 5 methods | stage2/dose_sweep.csv/.png |
| run_lesions_poisson.py --I0 100 --out results/stage2/lesions_I100 | lesion-insertion test under Poisson noise (also --I0 30) | stage2/lesions_I*/lesion_*_metrics.csv |
| summarize_step5.py --dir results/stage2/lesions_I100 | aggregates the lesion results | step5_summary.csv/.png in that folder |

### Stage 3 extensions (LoDoPaB-CT)

| Script | What it does | Main outputs (results/stage3/) |
|---|---|---|
| run_stage3_step1.py --root data | data files, patient-disjoint check, geometry conventions vs the real observations, FBP and timing | step1_summary.json, geometry_variants.csv |
| run_stage3_step2.py --exp noise / --exp baselines / --extract_train 70 | Poisson-model check (integrality, dispersion), FBP/TV baselines, training-slice extraction | noise_check.json, baselines_summary.csv |
| train_denoiser_ct.py [--depth 17 --steps 20000 --out ...] | noise-conditioned DnCNN on LoDoPaB slices | dncnn_ct.pt, dncnn_ct_d17.pt |
| run_stage3_step3.py (superseded), run_stage3_step3b.py | PnP vs TV at K=60, variants (TV init, annealing, TV+denoise, deeper net) | step3b_paired.csv |
| run_stage3_step4.py, run_stage3_step4b.py, run_stage3_step4c_compare.py | lesion test at K=60 (CT and phantoms), paired statistics, threshold sweep, gallery; final version in s4c/ | s4c/step4_lesion_summary.csv, step4b_paired.csv, step4b_theta_sweep.csv, step4b_gallery_*.png |
| run_stage3_step5.py [--angles K --doses f] | dose x angle grid (K = 30, 60, 120; dose 1, 0.25, 0.1): tuning, lesion-free accuracy, detection, false blobs (resumable) | s5/step5_K*_dose*.npz, step5_params_*.json |
| run_stage3_step5_summary.py | paired statistics and grid figure | s5/step5_means.csv, step5_paired.csv, step5_grid.png |
| run_stage3_step5_shift.py, run_stage3_step5_shift_summary.py | shift to phantoms at K=30, 120 with the step-5 parameters; absolute false-blob counts | s5/shift/step5shift_paired.csv, step5shift_false_absolute.csv |

`run_stage3_step3.py` is the superseded first comparison and is kept only for
the record; its numbers are not used.

Tests: `python -m pytest tests -q` (68 tests). Data: LoDoPaB-CT zip files in
`data/` (needs `pip install h5py`).