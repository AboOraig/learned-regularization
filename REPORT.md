# Results Companion

Compact record of every experiment (E1–E19) behind the paper *Learned Regularization for Robust Image Reconstruction*.
The paper (`paper/main.tex`) carries the method, the discussion and the full statistics; this file gives, for each experiment,
the setting, the key numbers, the verdict, the caveat and how to re-run it. The overview and the setup are in [README.md](README.md).

**Conventions.** PSNR in dB. "Gap" = PnP minus TV (positive: PnP better). Digits: mean ± 1.96 s.e. over 200 test images unless stated.
Real CT: paired differences, 95 % percentile bootstrap over 40 patients; `*` = interval excludes 0. No multiplicity correction anywhere.
TV-ADMM = unconstrained TV; TV-box = TV with 0 ≤ x ≤ 1 (Chambolle–Pock); Stage 3 TV is always TV-box; TV-long = TV with twice the iterations and its own tuned weight.
All figures below are existing outputs in `results/`; nothing here is drawn by hand.

## Contents

- [Key numbers](#key-numbers) · [Stage 1: 28×28 digits, Gaussian noise (E1–E10)](#stage-1-28×28-digits-gaussian-noise) ·
  [Stage 2: Poisson noise, matrix-free operator (E11–E13)](#stage-2-poisson-noise-and-a-matrix-free-operator) ·
  [Stage 3: real LoDoPaB-CT (E14–E19)](#stage-3-real-lodopab-ct)
- [Appendix A: tuned hyper-parameters](#appendix-a-tuned-hyper-parameters-stage-3) · [B: Stage 3 data and simulator](#appendix-b-stage-3-data-and-simulator-check) ·
  [C: analysis history](#appendix-c-analysis-history) · [D: Stage 3 command sequence](#appendix-d-stage-3-command-sequence) · [E: errata](#appendix-e-errata-changes-with-respect-to-the-earlier-report)

## Key numbers

| Claim | Number | Experiment |
|---|---|---|
| PnP vs box-TV, MNIST, σ = 0.05, 20 angles | +3.36 dB (+3.84 vs TV-ADMM) | E4, E8 |
| MNIST-trained prior on Fashion-MNIST / Shepp–Logan, vs TV-ADMM | −3.10 ± 0.30 / −4.33 ± 0.12 dB (vs box: −3.13 / −4.55) | E8 |
| Fashion-trained prior on Fashion-MNIST | +2.21 ± 0.01 dB | E8 |
| Share of excess error under shift in the weakly determined subspace | 93 % (Fashion), 89 % (phantom) | E7 |
| In-distribution paired AUC, PnP vs TV (digits) | 0.69 vs 0.91 | E9 |
| PnP − TV-long, real CT, 9 conditions | −0.93 to +0.33 dB; SSIM lower in 8 of 9 (7 significant) | E16 |
| Detection AUC, PnP-const − TV-long | +0.03 to +0.16 (6 of 9 significant); never significantly above FBP-Hann | E17 |
| TV-long vs FBP-Hann in detection | worse in 9 of 9 (7 significant), despite +0.8–2.3 dB PSNR | E17 |
| Phantom shift (K = 30/60 × 3 doses) | PSNR below TV-long in 6 of 6 (5 significant); detection advantage kept in 1 of 6 | E19 |

---

## Stage 1: 28×28 digits, Gaussian noise

Dense Radon operator (800×784 at 20 angles), noise-conditioned DnCNN trained on MNIST (σ_max 0.3), seeds and hyper-parameters tuned on validation images.

### E1–E2. In-distribution accuracy versus noise and undersampling

- **Setting.** Inverse-crime data, 200 test digits, σ ∈ {0 … 0.2} at 20 angles; 5–90 angles at σ = 0.05; also crime-free (bicubic ×5) data and TV-box.
- **Key result** (PnP − TV, inverse-crime data, 20 angles):

| σ | 0 | 0.01 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| vs TV-ADMM | −6.43 | +0.07 | **+3.86** | +3.01 | +1.03 |
| vs TV-box | −12.93† | −0.75 | **+3.33** | +2.65 | +0.65 |

  Angle sweep at σ = 0.05: +2.13 / +3.03 / +3.83 / +3.04 dB at 5 / 10 / 40 / 90 angles (vs TV-ADMM, 20 angles: +3.86); PnP is best at every angle count. † the TV-box optimum lies below the λ grid, so 42.5 dB is a data-model artefact.
- **Figure.** `results/comparison_noise.png` (PSNR/SSIM vs σ), `comparison_angles.png`, `fig_recon_grid.png` (first four test images, no selection).
- **Verdict.** The advantage is largest at moderate noise, shrinks at both ends and reverses without noise. The box constraint lowers the gap by 0.4–1.0 dB at σ ≥ 0.05 but changes the sign only at σ = 0.01 (inverse crime).
- **Caveat.** Error bars reflect image variability only, not tuning or training variability.
- **Run.** `run_baselines.py`, `run_pnp.py`, `make_plots.py`, `make_figures.py`; box-TV: `run_tv_box_control.py --data fine5c`. **Outputs.** `baselines_*.csv`, `control/`.

### E3. Forward-model mismatch (angle offset and jitter)

- **Setting.** Every projection angle shifted by δ (constant offset) or perturbed with s.d. δ (jitter); hyper-parameters frozen at nominal values.
- **Key result** (PnP − TV-ADMM, dB): offset 0° +3.86, 1° +3.58, 2° +2.90, 4° +1.49; jitter 4° +2.98. PnP is still higher in 96–100 % of images, but loses more than TV in absolute terms (4.0 vs 1.6 dB at a 4° offset).
- **Rotation check.** A constant offset is a rigid image rotation: PnP against the rotated truth scores 24.14 / 24.23 / 24.14 dB at 1° / 2° / 4° (nominal 24.00), so much of the "loss" is rotation, not reconstruction failure.
- **Figure.** `results/robust_mismatch.png`. **Verdict.** The margin shrinks with mismatch but does not reverse in this range.
- **Caveat.** Frozen, not re-tuned; TV-box not run for this experiment.
- **Run.** `run_robustness.py --exp mismatch`, `run_e3_rotation_check.py`. **Outputs.** `control/e3_offset_rotation.csv`.

### E4. Distribution shift (one denoiser)

- **Setting.** Pipeline frozen at MNIST values; σ = 0.05, 20 angles, inverse-crime data.

| Test set | TV | PnP | Gap (wins) | SSIM TV / PnP |
|---|---|---|---|---|
| MNIST | 20.27 | 23.97 | +3.70 ± 0.16 (100 %) | 0.742 / 0.944 |
| EMNIST letters | 20.09 | 22.03 | +1.94 ± 0.18 (92 %) | 0.842 / 0.938 |
| Fashion-MNIST | 19.61 | 16.57 | −3.05 ± 0.31 (11 %) | 0.680 / 0.675 |
| Shepp–Logan | 19.19 | 15.03 | −4.16 ± 0.19 (0 %) | 0.722 / 0.517 |

- **Figures.** `results/robust_shift.png`, `fig_shift_grid.png` (first test image of each set, no selection: on Fashion and the phantom PnP adds spotted, stroke-like fragments; TV blurs but does not).
- **Verdict.** The advantage holds for stroke-like shift (EMNIST) and reverses for Fashion-MNIST and the phantom.
- **Caveat.** One denoiser; the phantom is one image with 20 noise draws, so its interval describes noise draws, not images. This is the single-seed version of E8.
- **Run.** `run_robustness.py --exp shift`.

### E5. Why PnP selects few iterations

- **Setting.** PnP run to k = 400 at four denoiser strengths; data residual and error in bands of s_i/s_1.
- **Key result.** PSNR peaks at k ≈ 5 and then falls (MNIST 23.97 → 22.09 at k = 400; Fashion 16.57 → 13.80; phantom 15.03 → 12.40) while the data residual stays at the noise level (0.994 → 1.019 of σ√m); 74 % of the error growth lies in bands 0.1 ≤ s/s_1 < 0.3. Even the best (σ_d, k) chosen on the test images (optimistic) leaves Fashion at 17.86 and the phantom at 16.36, still below TV (19.61, 19.19).
- **Figure.** `results/fig_convergence.png`, `iteration_control_fine5.png`.
- **Verdict.** Early stopping acts as regularisation; the pattern is consistent with semi-convergence.
- **Caveat.** Semi-convergence is a hypothesis; a noise-free iteration curve would test it and was not run. One noise draw per point.
- **Run.** `run_spectral.py`, `make_fig_e5_iterations.py`. **Outputs.** `iteration_control*.csv`, `spectral_bands.csv`.

### E6. Sensitivity to the inverse crime

- **Setting.** Four data models: inverse crime, nearest, bilinear, bicubic (fine grid ×5), same 200 digits.
- **Key result.** For σ ≥ 0.05 the curves nearly coincide (gap +3.86 inverse crime vs +4.02 bicubic at σ = 0.05). At σ = 0 TV falls from 36.0 to 33.2 dB while PnP gains 0.3 dB, so the noise-free ranking depends on the data model (gap −6.43 vs −3.29 dB); the changes at low noise reach 5.5 dB across data models and controls.
- **Figure.** `results/crime_comparison.png`. **Verdict.** Report the data model with every low-noise comparison.
- **Caveat.** The mean model error of the bicubic data grows with the angle count (0.05 → 0.17 for 5 → 90 angles). TV-box only for inverse-crime and bicubic data.
- **Run.** `run_nullspace.py --fine_factor 5 --up_mode bicubic`, `compare_crime.py`. **Outputs.** `crime_comparison.csv`.

### E7. Error decomposition along singular vectors

- **Setting.** W_0.05 = directions with s_i/s_1 ≥ 0.05 (dim 319 of 784); W⊥ (465) includes the exact null space (79). Bicubic data, σ = 0.05.

| Test set | error ratio PnP/TV in W | in W⊥ | injection ι (TV / PnP) | alignment c (TV / PnP) |
|---|---|---|---|---|
| MNIST | 0.48 | 0.40 | 0.58 / 0.76 | 0.45 / 0.76 |
| EMNIST | 0.70 | 0.64 | 1.22 / 1.55 | 0.43 / 0.67 |
| Fashion-MNIST | 1.27 | 2.32 | 0.29 / 0.80 | 0.36 / 0.37 |
| Shepp–Logan | 2.59 | 2.64 | 0.44 / 3.27 | 0.09 / 0.23 |

- **Figure.** `results/nullspace_signature_fine5c.png`.
- **Verdict.** 93 % (Fashion) and 89 % (phantom) of PnP's excess error under shift is in W⊥, but the "loud and misaligned" signature (ι ≫ 1, low c) appears only on the phantom. "Null-space hallucination" is an incomplete description. The ratios are insensitive to the choice of subspace (exact null space, W_0.05, W_0.10).
- **Caveat.** The decomposition is descriptive; it does not isolate a cause.
- **Run.** `run_nullspace.py`. **Outputs.** `nullspace_*.csv`.

### E8. Denoiser seeds and the matched-prior control

- **Setting.** Three independently trained denoisers per prior (m03 = MNIST σ_max 0.3, f03 = Fashion-MNIST 0.3; m05 = MNIST 0.5 for the noise sweep); each prior's pipeline tuned on its own validation set; bicubic data, σ = 0.05, 20 angles.

| Test set | MNIST prior vs ADMM | vs TV-box | Fashion prior vs ADMM | vs TV-box |
|---|---|---|---|---|
| MNIST | **+3.84 ± 0.03** | +3.36 | +1.38 ± 0.09 | +1.44 |
| EMNIST | +1.79 ± 0.09 | +1.72 | +0.44 ± 0.09 | +0.38 |
| Fashion-MNIST | **−3.10 ± 0.30** | −3.13 | **+2.21 ± 0.01** | +2.17 |
| Shepp–Logan | **−4.33 ± 0.12** | −4.55 | −0.18 ± 0.08 | −0.05 |

- **Figure.** `results/step4/step4_summary.png`.
- **Verdict.** The reversal is prior mismatch, not a property of PnP: with a Fashion-trained prior the same pipeline beats TV on Fashion by 2.2 dB. A broader prior shrinks the phantom deficit from −4.3 to −0.2 dB. The seed alone moves the Fashion gap between −2.83 and −3.43 dB.
- **Caveat.** Both priors are DnCNNs trained on small digit sets; TV-box only for m03 and f03.
- **Run.** `run_step4.py`, `summarize_step4.py`, `run_tv_box_shift.py`. **Outputs.** `step4/`, `control/tv_box_shift_seeds.csv`.

### E9. Lesion insertion

- **Setting.** Gaussian lesion (σ_l px, contrast c) added at a known location; lesion-present/absent images reconstructed from independent noise draws; location-known observer. Eight lesion types, 200 test images, three denoisers per prior, PSNR-tuned settings.
- **Key result** (σ_l/c = 0.75/0.5; paired AUC / false lesions per lesion-absent image):

| Prior → test set | TV | TV-box | PnP |
|---|---|---|---|
| MNIST → MNIST | 0.91 / 0.005 | 0.90 / 0.010 | 0.69 / 0.000 |
| MNIST → Fashion | 0.91 / 0.030 | 0.92 / 0.090 | 0.82 / 0.430 |
| Fashion → Fashion | 0.91 / 0.030 | 0.90 / 0.025 | 0.75 / 0.005 |

- **Figure.** `results/step5/step5_summary.png`.
- **Verdict.** In distribution PnP is about 4 dB better in PSNR but its detection AUC is lower, because it attenuates lesions (retention 0.28 vs 0.59). Under prior mismatch it keeps or exaggerates lesions (retention 0.99) and invents lesion-like blobs (0.43 vs 0.03 per image). PSNR does not predict detectability.
- **Caveat.** Additive Gaussian lesions and a location-known observer; no reader study.
- **Run.** `run_lesions.py`, `summarize_step5.py`, `run_tv_box_lesions.py`. **Outputs.** `step5/`, `control/tv_box_lesions_*.csv`.

### E10. Can re-tuning recover lesion detectability?

- **Setting.** Sweep of σ_d, k (PnP) and λ (TV); operating points chosen on held-out validation images by PSNR, by lesion AUC, or by an AUC rule with a PSNR constraint.
- **Verdict.** Re-tuning trades erasure against invention. For the broader Fashion-trained prior a competitive operating point exists; for the MNIST-trained prior it does not.
- **Figure.** `results/step5b/oppoint_m03_s0.png` (and the f03 counterpart). **Caveat.** The operating-point rules were fixed after seeing the first sweep.
- **Run.** `run_oppoint.py`. **Outputs.** `step5b/oppoint_*.csv/.png`.

---

## Stage 2: Poisson noise and a matrix-free operator

### E11. The matrix-free operator reproduces the dense pipeline

- **Setting.** 200 images, fixed untuned settings, one GPU run, exact adjoint.
- **Key result.** Relative difference between the dense and the matrix-free reconstruction: Tikhonov 4.6×10⁻⁶, TV-ADMM 9.7×10⁻⁶, PnP (k = 25) 2.8×10⁻⁴ (same PSNR: 16.36 / 19.01 / 23.10). Peak GPU memory 184 MB. The randomized SVD agrees with the dense one for the error split (`run_e11_svd_check.py`).
- **Verdict.** The two pipelines are numerically equivalent; Stages 2–3 can use the matrix-free operator.
- **Caveat.** Single-precision GPU run; timings are for this machine only (TV-ADMM is slower matrix-free: 7.5 vs 0.16 s).
- **Run.** `run_stage2.py --exp equiv`, `run_e11_svd_check.py`. **Outputs.** console, `control/e11_svd_check.csv`.

### E12. Poisson dose sweep

- **Setting.** Photon count I₀ ∈ {1000, 300, 100, 30, 10} per ray; hyper-parameters re-tuned at each dose; MNIST-trained prior.
- **Key result** (PnP − TV PSNR, dB):

| I₀ | 1000 | 300 | 100 | 30 | 10 |
|---|---|---|---|---|---|
| MNIST | +2.87 | +3.68 | +3.65 | +2.39 | +0.82 |
| EMNIST | +1.40 | +1.59 | +1.82 | +0.77 | +0.62 |
| Fashion-MNIST | −2.45 | −3.05 | −2.74 | −3.01 | −1.98 |
| Shepp–Logan | −4.41 | −6.05 | −5.05 | −5.49 | −4.75 |

- **Figure.** `results/stage2/dose_sweep.png`.
- **Verdict.** Same pattern as Gaussian noise: the gain peaks at moderate dose and the reversal under shift persists at every dose. SSIM gap: +0.09 to +0.28 on MNIST and +0.04 to +0.17 on EMNIST; on Fashion −0.014 to +0.054; on the phantom −0.19 to −0.31.
- **Caveat.** Phantom: one image. A Poisson-weighted PnP variant was run and is in the figure.
- **Run.** `run_stage2.py --exp dose`. **Outputs.** `stage2/dose_sweep.csv`.

### E13. Lesion insertion under Poisson noise

- **Setting.** As E9 with I₀ = 100 and 30, bicubic data, each method tuned on its own prior's validation distribution.
- **Verdict.** The lesion results are similar to E9 (see the paper, E13); per-condition numbers are in `results/stage2/lesions_I*/`.
- **Caveat.** Same observer limits as E9. **Run.** `run_lesions_poisson.py --I0 100 --out results/stage2/lesions_I100` (and `--I0 30`), `summarize_step5.py --dir ...`.

---

## Stage 3: real LoDoPaB-CT

362×362 slices, patient-disjoint split, K = 30 / 60 / 120 of 1000 angles, doses 1 / 0.25 / 0.1 of N₀ = 4096 photons per ray (exact Poisson thinning of the real counts), 40 test patients, 20 validation slices per condition, one CT denoiser trained on 8,960 of 35,820 slices.
Variants: PnP-const, PnP-TVinit (started from the TV solution), PnP-anneal, TV+denoise, d17 (17-layer denoiser). Data, simulator check and tuned values: Appendices A–B.

### E14. Real data, noise model and simulator check

- **Setting.** Geometry fitted to the real observations (16 variants), Poisson dose model, simulated vs real TV reconstructions.
- **Key result.** Selected geometry: relative residual 0.038 (best alternative 0.166), pixel size 1.0044× nominal. Real vs simulated TV PSNR over the nine conditions differs by at most 0.16 dB (Appendix B).
- **Figure.** `results/stage3/geometry_check.png`.
- **Verdict.** The crime-free simulator is adequate for the TV baseline and is used for the lesion and phantom experiments.
- **Caveat.** Validated on 8 slices (K = 60, TV and FBP-Hann only); not validated for PnP.
- **Run.** `run_stage3_step1.py`, `run_stage3_step2.py`. **Outputs.** `stage3/noise_check.json`.

### E15. Learned prior versus TV on real slices, K = 60

- **Setting.** 40 test slices, doses 1 and 0.25, 8- and 17-layer denoisers.
- **Key result** (PSNR − TV, d8): PnP-const −0.17* (dose 1), −0.10* (0.25); PnP-TVinit +0.03, +0.21*; SSIM of PnP-const 0.773 vs 0.786 and 0.734 vs 0.750.
- **Verdict.** The learned prior roughly ties TV in PSNR and is lower in SSIM; the deeper denoiser is not better.
- **Caveat.** The TV-long comparison of E16 uses independent noise draws, so numbers differ slightly (−0.28 vs −0.17 at dose 1). PnP-TVinit and the other variants were added after the first result.
- **Run.** `run_stage3_step3b.py`. **Outputs.** `stage3/step3b_*.csv`.

### E16. Accuracy over the dose × angle grid

| PSNR (dB) | K=30, 1 | 30, 0.25 | 30, 0.1 | 60, 1 | 60, 0.25 | 60, 0.1 | 120, 1 | 120, 0.25 | 120, 0.1 |
|---|---|---|---|---|---|---|---|---|---|
| FBP-Hann | 29.02 | 27.72 | 26.63 | 30.59 | 28.66 | 27.11 | 31.70 | 29.18 | 27.94 |
| TV-long | 31.31 | 29.21 | 27.72 | 32.12 | 29.88 | 27.98 | 32.50 | 30.11 | 29.21 |
| PnP-const | 30.38 | 29.05 | 27.86 | 31.85 | 29.68 | 28.31 | 32.64 | 30.23 | 28.86 |
| PnP-TVinit | 31.15 | 29.22 | 27.96 | 32.09 | 30.04 | 28.47 | 32.92 | 30.46 | 29.26 |

- **Figure.** `results/stage3/s5/step5_grid.png` (corrected title: PnP-const − TV-long in the first panel).
- **Verdict.** PnP-const − TV-long ranges from −0.93* to +0.33* dB (3 significantly positive, 4 significantly negative, 2 not significant); SSIM is lower in 8 of 9 (7 significant). The best variant, PnP-TVinit, is highest in 7 of 9, by at most 0.49 dB over TV-long, but it was introduced after the first result and at K = 120, dose 1 its gain does not come from the denoiser. Everything beats FBP-Hann by 0.9–1.4 dB.
- **Caveat.** One noise draw per condition; TV-long weight grid has steps of a factor three; no multiplicity correction.
- **Run.** `run_stage3_step5.py`, `run_stage3_step5_summary.py`. **Outputs.** `stage3/s5/step5_*.csv`.

### E17. Lesion detection on real slices

- **Setting.** Lesions σ_l ∈ {2, 4} px, c = 0.06, location-known observer, paired AUC over 40 patients, nine conditions.
- **Key result.** PnP-const − TV-long: +0.03 to +0.16 (6 of 9 significant); PnP-const − FBP-Hann never significantly positive. TV-long − FBP-Hann is negative in 9 of 9 (7 significant), while TV-long gains 0.8–2.3 dB PSNR.
- **Figure.** `results/stage3/s5/step5_grid.png` (bottom row).
- **Verdict.** On real CT the sign of the detection difference is opposite to the digits (E9); a PSNR-tuned TV is worse than doing nothing at detection. The cause was not isolated.
- **Caveat.** AUC uncertainty about ±0.05 per cell; lesions are synthetic.
- **Run.** `run_stage3_step5.py`, `run_stage3_step4*.py`. **Outputs.** `stage3/s5/`, `stage3/s4c/`.

### E18. False structures on real slices

- **Setting.** Lesion-free slices; false blobs counted above a threshold in units of the nominal lesion response.
- **Key result.** 0.24–1.38 false blobs per slice for PnP-const, fewer than FBP-Hann in every condition examined. The strongest blobs (galleries) lie at the same place in FBP-Hann and TV: they come from the data, not from the prior.
- **Figure.** `results/stage3/step4b_gallery_*.png`.
- **Verdict.** On real slices PnP does not invent more structure than the baselines.
- **Caveat.** The blob definition and threshold sweep were added after seeing results. The gallery suptitle overlaps the first row (cosmetic; needs a GPU rerun).
- **Run.** `run_stage3_step4b.py --gallery 1`.

### E19. Shift to non-anatomical phantoms

- **Setting.** 20 piecewise-constant phantoms; 6 conditions; same tuned settings as for real slices.
- **Key result.** PSNR below TV-long in 6 of 6 (5 significant), SSIM in all 4 stored; detection advantage kept in 1 of 6; false blobs relative to TV change sign with K.
- **Verdict.** The advantage disappears under shift on CT as on digits. This shift is deliberately TV-friendly.
- **Caveat.** One kind of shift; K = 60 SSIM not stored.
- **Run.** `run_stage3_step5_shift.py`, `run_stage3_step5_shift_summary.py`, `run_e19_k60_psnr_ci.py`. **Outputs.** `stage3/s5/shift/`, `control/e19_k60_psnr_ci.csv`.

---

## Appendix A: tuned hyper-parameters (Stage 3)

Chosen on 20 validation slices per condition. Hann = cut-off; λ = TV weight (TV-long used the same weight as TV in every condition); (σ_d, k) for PnP. "edge" = optimum on the grid boundary.

| K | dose | Hann | TV λ | PnP-const (σ_d, k) | PnP-TVinit (σ_d, k) |
|---|---|---|---|---|---|
| 30 | 1 | 0.2 | 0.003 | (0.02, 12) | (0.02, 4) |
| 30 | 0.25 | 0.16 | 0.01 | (0.05, 8) | (0.08, 4) |
| 30 | 0.1 | 0.12 | 0.03 | (0.08, 12) | (0.05, 8) |
| 60 | 1 | 0.3 | 0.003 | (0.02, 8) | (0.02, 2) |
| 60 | 0.25 | 0.2 | 0.01 | (0.08, 4) | (0.02, 8) |
| 60 | 0.1 | 0.12 | 0.03 | (0.05, 12) | (0.02, 12) |
| 120 | 1 | 0.4 | 0.003 | (0.05, 2) | (0.0039, 8) edge |
| 120 | 0.25 | 0.2 | 0.01 | (0.02, 12) | (0.0062, 8) |
| 120 | 0.1 | 0.16 | 0.01 | (0.05, 8) | (0.128, 1) edge |

Digits: σ_d = 0.1, k = 5, λ_TV = 5.6×10⁻³ for the matched-prior control; only the denoiser weights differ between priors.

## Appendix B: Stage 3 data and simulator check

- **Split.** Train 35,820 slices / 632 patients; validation 3,522 / 60; test 3,553 / 60; patient ids disjoint. Test results use 40 patients.
- **Dose model.** N₀ = 4096, μ_max = 81.35858 m⁻¹, zero counts replaced by 0.1; counts integral in 100 % of rays; dispersion 1.06 / 1.25 / 1.81 at the three doses.
- **Real vs simulated TV PSNR, nine conditions** (order K = 30, 60, 120 × dose 1, 0.25, 0.1): real 29.34 / 27.95 / 26.57 / 30.15 / 28.40 / 26.73 / 30.61 / 28.54 / 27.89; simulated 29.41 / 27.90 / 26.52 / 30.27 / 28.45 / 26.72 / 30.76 / 28.55 / 27.85.
- **40-slice check, K = 60** (real vs simulated): TV 31.91 vs 32.04 and 29.82 vs 29.83; FBP-Hann 30.65 vs 30.59 and 28.73 vs 28.66.

## Appendix C: analysis history

1. The first K = 60 comparison used 6 validation slices; PnP was 0.85 and 0.33 dB below TV.
2. With 20 validation slices PnP-const rose by 1.06 and 0.32 dB, TV by 0.38 and 0.09, and the gap became −0.17 and −0.10 dB.
3. The lesion experiments were redone on 20 slices with extended grids and a separately tuned TV-long. The paired statistics, threshold sweep and false-blob definition were added after seeing results.

Nothing was pre-specified. Treat the sign pattern across conditions as the evidence, not individual intervals.

## Appendix D: Stage 3 command sequence

```
python run_stage3_step1.py --root data
python run_stage3_step2.py --exp noise
python run_stage3_step2.py --exp baselines --angles 60 --doses 1,0.25 --n_val 6 --n_test 40
python run_stage3_step2.py --extract_train 70
python train_denoiser_ct.py
python run_stage3_step3b.py --models results/stage3/dncnn_ct.pt --doses 1
python train_denoiser_ct.py --depth 17 --steps 20000 --out results/stage3/dncnn_ct_d17.pt
python run_stage3_step3b.py --models results/stage3/dncnn_ct.pt,results/stage3/dncnn_ct_d17.pt --doses 1,0.25
python run_stage3_step4.py --doses 1 --out results/stage3/s4c --n_val 20 --auto_extend 2 --hann 0.12,0.16,0.2,0.3,0.4,0.5 --tv_lams 1e-3,3e-3,1e-2,3e-2 --sd_grid 0.01,0.02,0.03,0.05,0.08 --ck 2,4,8,12,20,30,45
python run_stage3_step4b.py --doses 1 --out results/stage3/s4c --gallery 1
python run_stage3_step4.py --doses 0.25 --out results/stage3/s4c --n_val 20 --auto_extend 2 --hann 0.08,0.12,0.16,0.2,0.3 --tv_lams 1e-3,3e-3,1e-2,3e-2 --sd_grid 0.01,0.02,0.03,0.05,0.08 --ck 1,2,4,8,12,20,30,45
python run_stage3_step4b.py --doses 0.25 --out results/stage3/s4c --gallery 1
python run_stage3_step4c_compare.py
python run_stage3_step5.py            # all 9 conditions (resumable; --angles K --doses f for pieces)
python run_stage3_step5_summary.py
python run_stage3_step5_shift.py
python run_stage3_step5_shift_summary.py
python -m pytest tests -q             # 68 tests
```

`run_stage3_step3.py` is the superseded first comparison; its numbers are not used.

## Appendix E: errata (changes with respect to the earlier report)

| Earlier statement | Now |
|---|---|
| Learned prior is 3.8–3.9 dB better than TV | About 3.4 dB against the box-constrained TV (+3.36 E8; +3.84 vs TV-ADMM) |
| Gap +3.85 dB (E3) | +3.86 (rounding) |
| Data residual rises with iterations | Stays at the noise level (0.994 → 1.019) |
| Only a matched prior avoids the reversal | A broader (Fashion-trained) prior is −0.18 dB on the phantom instead of −4.3 |
| PnP SSIM lower than TV in seven of nine conditions | Eight of nine (seven significant) |
| "Finding 1–4" labels | Replaced by experiment numbers E1–E19 |
| `step5_grid.png` panel title swapped red/blue | Fixed and regenerated |