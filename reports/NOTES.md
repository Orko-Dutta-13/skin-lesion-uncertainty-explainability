# Project notes (numbers from real runs only)

Last updated: 2026-11-18. Config: EfficientNet-B0 (ImageNet-pretrained), 224 px, dropout 0.3, AdamW lr 3e-4,
wd 1e-4, batch 64, sqrt-inverse-frequency class weights, early stopping (patience 5) on val balanced accuracy,
about 33 s/epoch on an RTX 3070 Laptop GPU. Seeds: 42 (`runs/baseline`), 1, 2.

## Step 1: data, leakage-safe evaluation, baseline, Grad-CAM

### Data (ISIC 2019 training, CC BY-NC 4.0)
- 25,331 images, 13,931 lesion groups; 8.23 % of images have no lesion_id (treated as singleton groups).
- Largest lesion group: 31 images; 5,059 lesions have more than one image.
- Sources inferred from image size: BCN 12,414 / HAM 10,015 / MSK_other 2,902. Checked against lesion_id
  prefixes: all 10,015 HAM and 12,413 of 12,414 BCN match; MSK_other = 819 MSK + 2,083 without lesion_id.
- Split (lesion-grouped, stratified): train 18,093 / val 3,619 / test 3,619.
- Class imbalance: NV 12,875 vs DF 239 and VASC 253. Val/test have only about 34 DF and 36 VASC images each.
- Grouping is lesion-level, NOT patient-level (ISIC 2019 has no patient ID).

### Leakage check (val/test images sharing a lesion with a train image)
| split column | test | val |
|---|---|---|
| split (grouped) | 0 / 3,619 (0 %) | 0 / 3,619 (0 %) |
| split_naive (random images) | 2,094 / 3,619 (57.9 %) | 2,056 / 3,619 (56.8 %) |
| split_shift | shift_test 0 / 12,414 | val 0 / 1,846 |

### Baseline results (grouped split, 3 seeds, mean ± sample SD)
| Metric | Val (best epoch) | Test |
|---|---|---|
| Balanced accuracy | 0.651 ± 0.016 | **0.604 ± 0.013** |
| Macro AUROC | 0.940 | 0.938 ± 0.006 |
| ECE | about 0.072 | 0.063 ± 0.008 |
| NLL | | 0.757 ± 0.012 |
| Accuracy | | 0.733 |

- Per-seed test balanced accuracy: 0.620 / 0.596 / 0.597. Val numbers are optimistic (checkpoint picked on a noisy metric).
- Test recall per class (seed 42): MEL 0.61, NV 0.85, BCC 0.65, AK 0.52, BKL 0.58, DF 0.77, VASC 0.64, SCC 0.34.
  Melanoma recall across seeds 0.61 to 0.64; SCC 0.24 to 0.41; AK 0.40 to 0.52. DF/VASC recall swings widely (about 35 test images).
- Overfitting: train loss keeps falling, val NLL is lowest around epoch 3 to 5, val ECE rises from about 0.02 to about 0.13.

### Leakage ablation (naive random image split, single run)
- Test: balanced acc 0.814, AUROC 0.978, NLL 0.478, ECE 0.056, accuracy 0.848.
- Grouped vs naive: balanced accuracy 0.604 vs 0.814 = about 0.21 inflation (about 15 x the seed SD).
- Caveats: the naive result is one run; the two protocols use different test images.

### Grad-CAM (32 hand-sampled test images, 2 per class, seed 0)
- Heat mostly on the lesion; no obvious ruler/hair/border shortcut in these samples. Not a systematic audit.
- Errors look like lesion-level confusions. Some are confidently wrong (SCC predicted as BKL at 0.98, AK as BCC at 0.92).
- Limits: 7 x 7 final feature map is coarse; many maps are centred blobs (lesions are centred anyway).
- TODO: sanity check with a randomised-weights model; check black circular border vs source.

## Step 2.1: temperature scaling (done)
T fitted on val logits only, applied unchanged to test. Predictions and balanced accuracy are identical before/after.

| Run | T | ECE before -> after | NLL before -> after | Brier before -> after |
|---|---|---|---|---|
| seed 42 | 1.331 | 0.0669 -> 0.0213 | 0.7686 -> 0.7405 | 0.3868 -> 0.3789 |
| seed 1 | 1.356 | 0.0539 -> 0.0280 | 0.7445 -> 0.7153 | 0.3720 -> 0.3667 |
| seed 2 | 1.341 | 0.0691 -> 0.0121 | 0.7567 -> 0.7306 | 0.3783 -> 0.3705 |
| mean ± SD | 1.343 ± 0.013 | 0.0633 ± 0.0082 -> **0.0205 ± 0.0080** | 0.7566 -> 0.7288 | 0.3790 -> 0.3720 |

- Macro AUROC 0.9379 -> 0.9386 (essentially unchanged). Reliability diagram: `docs/figures/reliability_baseline.png`.
- Model is over-confident (T > 1). The weighted loss also shapes calibration; mention in the write-up.

## Step 2.2: MC Dropout, ensemble, selective prediction (done)
MC Dropout = Dropout + 16 StochasticDepth layers kept random at test time, BatchNorm in eval, 20 passes. Ensemble = mean of the 3 seeds (free deep ensemble). Test set, mean ± SD over 3 runs (ensemble = one model, no SD).

| Method (score) | ECE | NLL | AURC | acc@80% cov | MEL silently missed @80% |
|---|---|---|---|---|---|
| single (maxprob) | 0.063 | 0.757 | 0.111 | 0.803 | 0.269 |
| single + TS | 0.020 | 0.729 | 0.110 | 0.803 | 0.275 |
| MC Dropout (maxprob) | 0.042 | 0.728 | 0.107 | 0.806 | 0.267 |
| MC Dropout + TS | 0.018 | 0.720 | 0.107 | 0.806 | 0.272 |
| 3-seed ensemble (maxprob) | 0.015 | 0.654 | 0.089 | 0.832 | 0.237 |
| ensemble + TS | 0.021 | 0.653 | 0.088 | 0.832 | 0.238 |

- Ensemble is clearly best (AURC ~0.089 vs 0.110 ± 0.006, about 3 SD; NLL 0.654 vs 0.757). Ensemble is already calibrated, TS adds nothing.
- MC Dropout: better calibration than single (ECE 0.042 vs 0.063), ranking gain is within noise (AURC 0.107 vs 0.110). Calibration gain is about the same as TS alone.
- TS fixes calibration but does not change ranking (AURC unchanged).
- Mutual information is a worse error ranker than maxprob/entropy (MC AURC 0.121, ensemble 0.108); few members/passes.
- Melanoma: even at 80% coverage 24-29% of melanomas are kept but not called MEL; only 16-22% are deferred. Uncertainty alone does not make the model safe.
- Figure: `docs/figures/selective_prediction_baseline.png`. Raw: `reports/selective/`.
- Caveat: ensemble has one instance (3 seeds); no CI. Test set N=3619 lesions.
- Not yet run: head-only MC ablation (`--mode head`).

## Step 2.3: Grad-CAM randomisation sanity check (done)
Cascading weight randomisation (Adebayo et al. 2018): re-initialise Conv/Linear weights from the classifier down to features.0, recompute Grad-CAM for the originally predicted class, Spearman vs original map. 16 test images (2 per class, seed 0), baseline seed-42 model. BatchNorm statistics kept.

- Spearman falls 1.00 -> 0.15 ± 0.43 after the classifier alone, and stays about 0 (-0.10 to +0.32) for the rest; 0.03 ± 0.27 with everything randomised. Maps lose the lesion focus and scatter to image borders (`docs/figures/gradcam_sanity_baseline.png`).
- Reading: Grad-CAM depends on the trained weights, i.e. it is not just an edge/centre detector. Passed.
- Caveats: only 16 images, per-image SD about 0.3-0.4. Constant (all-zero) maps are excluded from the mean: 6 of 16 at features.0, so the last point is the least reliable. Randomised BN statistics were kept, which is one of several possible choices. Run on seed-42 model only.
- Does not show that the explanations are medically correct, only that they are model-dependent.

## Step 2.4: SHAP vs Grad-CAM (done)
SHAP = expected gradients (`shap.GradientExplainer`, shap 0.54.1), 100 samples, 50 training images as background, baseline seed-42 model. 32 test images (4 per class, random, seed 0; 18 correctly and 14 wrongly classified). Explained class = predicted class. Figure: `docs/figures/shap_gradcam_baseline.png`; per-image numbers: `runs/baseline/shap_test.csv`.

- Agreement Grad-CAM vs |SHAP| (Spearman on 14x14 pooled maps): mean 0.44, median 0.47, range -0.05 to 0.86 (SD 0.25). Moderate, not strong.
- Qualitatively: both methods put most weight on the lesion. Grad-CAM = one smooth, centred blob (7x7 map). SHAP = fine-grained, fragmented peaks, sometimes on small dark spots, hairs or the image border / dermoscope rim, and it can leave large uniform lesions mostly blank. Lowest agreement (rho about 0) on two AK images with texture across the whole frame.
- Correct vs wrong predictions: mean rho 0.38 vs 0.52. Small n and noisy, not interpreted as a finding.
- Additivity (sum of attributions vs f(x) - E f(bg)): mean abs gap 0.67 (about 16% of attribution total), larger for confident predictions (0.86 vs 0.43). This is sampling error of expected gradients; attributions are approximate.
- Caveats: 32 images, one model; no randomisation sanity check for SHAP; |SHAP| summed over colour channels loses sign; 14x14 pooling is a crude comparison. Neither method is validated against clinical annotations, so these are model explanations, not evidence of clinically correct reasoning.

## Step 3.2: source shift, HAM+MSK -> BCN (done)
Protocol `split_ood` (data/splits.py `add_ood_split`): train/val/test = HAM + MSK only (9226 / 1845 / 1846 images, grouped by lesion); `shift_test` = all 12,414 BCN images. 3 seeds of EfficientNet-B0 (`configs/shift.yaml`), temperature fitted on ID val only. ID-test accuracy / ECE / NLL / AURC are weighted to the BCN class mix (BCN has 34 % NV vs 67 % in the ID test set; BA, per-class recall and AUROC do not depend on class mix). Figure: `docs/figures/shift_overview.png`; tables: `reports/shift/`.

| | ID test (BCN mix) | BCN shift | BCN, trained with BCN (reference) |
|---|---|---|---|
| single, balanced accuracy | 0.679 ± 0.003 | 0.341 ± 0.008 | 0.533 ± 0.012 |
| single + TS, ECE | 0.061 ± 0.006 | 0.165 ± 0.039 | 0.039 ± 0.004 |
| ensemble + TS, balanced accuracy | 0.702 | 0.348 | 0.560 |
| ensemble + TS, ECE / NLL | 0.044 / 0.642 | 0.131 / 1.491 | 0.043 / 0.855 |
| ensemble + TS, acc at 80 % coverage | 0.826 | 0.596 | 0.755 |
| ensemble + TS, AURC (lower is better) | 0.087 | 0.292 | 0.159 |

- Balanced accuracy falls by 0.35 (bootstrap 95 % CI of the ensemble change: -0.40 to -0.31; image-level, optimistic). Macro AUROC 0.96 -> 0.82. Same pattern in all 3 seeds.
- Not all of that is source shift: models trained WITH BCN (the baseline runs, 2x the training images) only reach BA 0.53-0.56 on BCN, so BCN is also a harder set. A fairer reading is about 0.19 attributable to the shift (0.53 -> 0.34) plus about 0.15 for BCN being harder / different. The reference is not perfectly controlled (different training set size and different test images).
- Per class (ensemble + TS; BCN recall is reliable, ID recall is noisy for AK 19 / DF 16 / VASC 21 / SCC 28 test images): NV 0.91 -> 0.85, MEL 0.59 -> 0.56 hold up; BCC 0.85 -> 0.25, AK 0.53 -> 0.05, DF 0.81 -> 0.15, VASC 1.00 -> 0.39, BKL 0.68 -> 0.45, SCC 0.25 -> 0.08 collapse.
- Calibration: temperature scaling fitted on ID data does not carry over. ECE on BCN 0.165 (single) / 0.131 (ensemble) vs 0.039 when the model has seen BCN. The model is over-confident everywhere on BCN (about 83 % accurate at 95 % confidence).
- Ensemble under shift: better NLL/AURC (AURC 0.292 vs 0.325 single + TS), balanced accuracy about the same (0.348 vs 0.341, within the seed spread).
- Selective prediction does not fix it: ensemble error among answered images is 0.30 at 50 % coverage on BCN vs 0.06 on ID.
- Detection: uncertainty is a weak shift detector. AUROC for BCN vs ID images: max-prob 0.67, entropy 0.69 (ensemble), mutual information 0.68 (0.5 = no signal). Many BCN images get confident predictions.
- Caveats: single source pair (HAM+MSK -> BCN); BCN differs in class mix, acquisition and probably image quality, and these are not separated beyond reweighting; the ID test set is small for rare classes; seed spread only from 3 runs.

## Step 3.3: drift monitoring with Evidently (done)
`python -m reliable_derm.drift --ckpt runs/shift/best.pt` (seed-42 shift model, Evidently 0.7.23, pandas pinned to 3.0.x because 3.1 breaks Evidently). Reference = 1800 ID validation images; current = 1800 images each. Label-free columns only: 8 image statistics (brightness, contrast, saturation, sharpness, dark-pixel share, mean R/G/B), 8 PCA components of the 1280-d penultimate embedding (PCA fitted on the reference only), max-prob and entropy. Figure: `docs/figures/drift_overview.png`; Evidently HTML reports in `reports/drift/` (not committed).

| Current data vs ID validation | Columns flagged by Evidently | Domain-classifier AUROC: image stats / embedding / model output |
|---|---|---|
| control: ID test images | 0 % (0 of 18) | 0.51 / 0.49 / 0.51 |
| BCN, natural class mix | 94 % | 0.92 / 0.91 / 0.70 |
| BCN, sampled to the same class mix | 100 % | 0.94 / 0.91 / 0.65 |

- The monitor does not cry wolf: no column flagged on the control, classifier at chance.
- Shift is easy to see from the inputs: a logistic regression on 8 image statistics separates BCN from HAM/MSK images with AUROC 0.92-0.94, versus 0.65-0.69 when using the model's own confidence (Step 3.2). Monitoring inputs / embeddings detects this shift far better than monitoring uncertainty.
- Class-matching changes nothing, so this is an appearance difference, not a class-mix effect.
- Most drifted columns: dark-pixel share, mean red, contrast, saturation, brightness. BCN images are on average darker and less red (mean R about 0.65 vs 0.80), and a larger part of them have large dark areas (borders / vignettes). Cause not established: camera, lighting, skin tone, or the resize + centre crop applied to different original image sizes.
- Caveats: Evidently's default thresholds flag any systematic difference once n = 1800, so "flagged" does not mean "accuracy will drop by X". Drift tells you that the data changed, not how much performance suffers. One model, one source pair, one random reference sample (seed 0).

## Step 3.4: Docker (done)
`docker build -t reliable-derm .` (python:3.11-slim, CPU torch) then `docker run --rm reliable-derm` runs `make smoke-all` (fake data: prepare, train, evaluate, Grad-CAM, calibration, MC Dropout, selective, shift split, shift eval, drift) end to end without error on the user's Windows laptop, 2026-11-18. Smoke numbers are meaningless (284 fake images, 2 epochs).

## Open items
- 2.2 ablation: head-only MC Dropout (`--mode head`) vs full.
- Bootstrap confidence intervals for per-class metrics (DF, VASC, SCC).
- Shift: class mix handled by reweighting (metrics) and class-matched sampling (drift); BCN also appears harder than a fair reference (see 3.2). Not done: other source pairs, drift on a model trained on all sources.
- Step 4: README results + figures, CV bullet, cover-letter paragraph, first git commit.
