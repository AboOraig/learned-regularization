# Learned Regularization for Robust Image Reconstruction (sparse-view CT)

Setup: `pip install -r requirements.txt`

Smoke test (no download): `python run_baselines.py --exp noise --synthetic --n_val 30 --n_test 50`
Real run: `python run_baselines.py --exp noise` then `--exp angles`

## Status
- [x] Radon operator (explicit matrix, exact adjoint, angle offset for mismatch)
- [x] FBP, Tikhonov, TV-ADMM baselines; validation-only tuning; PSNR/SSIM/rel. error
- [x] DnCNN (sigma-conditioned) + PnP-PGD + run_pnp.py + make_plots.py (written, run order: train_denoiser -> run_pnp -> make_plots)
- [x] E3 (run_robustness.py --exp mismatch), E4 (--exp shift), figures (make_figures.py) - written
- [ ] Analysis + write-up
