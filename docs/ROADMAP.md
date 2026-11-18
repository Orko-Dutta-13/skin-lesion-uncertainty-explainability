# Roadmap (today = Tue 17 Nov 2026, target = Tue 15 Dec)

Rule: every step ends with an artefact you could show (a table, a figure, a committed notebook/script)
and a note in `reports/NOTES.md` on what surprised you. Those notes become the write-up.

## Step 1: data, leakage-safe split, baseline, Grad-CAM  (17 to 23 Nov)
| Day | Do | Done when |
|---|---|---|
| 1 | `pip install -r requirements.txt`, `make test`, `make smoke`. Create GitHub repo, first commit. | smoke passes on your machine |
| 1-2 | Download ISIC 2019 (docs/DATA.md), `make data` | `data_report.json` exists; leakage = 0% for `split`, clearly >0% for `split_naive`; 3 sources look plausible |
| 2 | Read `data_report.json`: % missing lesion_id, images per lesion, class counts per source | 5 lines in NOTES.md |
| 3-4 | `make train` (EfficientNet-B0, 224px, ~15 epochs), watch MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) | best val balanced acc logged |
| 5 | `make eval-val`, then `make eval-test` ONCE. Same train with `data.split_col=split_naive out_dir=runs/naive` | baseline table + leakage-inflation number |
| 6-7 | `make gradcam` (correct and wrong). Look at them critically: lesion or ruler/hair/border/ink? | figure + 5 sentences |

## Step 2: explanations + uncertainty  (24 Nov to 30 Nov)
- SHAP: GradientExplainer (expected gradients) or KernelSHAP on SLIC superpixels for ~30 images. DeepExplainer may
  fail additivity checks on EfficientNet ops; ResNet50 is better supported. Decide after trying 1 image.
- Grad-CAM sanity check: model-parameter randomisation test (CAM of trained vs random-weight model should differ).
- UQ: temperature scaling fitted on val (logits are already saved by evaluate.py), MC Dropout (head dropout alone gives
  weak variance; also test keeping stochastic depth active), optional 3-seed deep ensemble.
- Compare ECE/NLL/Brier, reliability diagram, risk-coverage curve (accuracy when rejecting the least-confident x%).

## Step 3: distribution shift + monitoring  (1 Dec to 8 Dec)
- Train on `split_shift` train, evaluate on `shift_test` (BCN). Report the drop in balanced acc and calibration.
- Does uncertainty flag the shift? AUROC of max-prob / entropy / MC variance for "in-dist vs BCN".
- Evidently: no native image drift. Compute embeddings (penultimate layer, PCA) + simple image statistics
  (brightness, contrast, mean colour, sharpness) as tabular features, run a drift report. Pin the Evidently version.
- Docker image + `make` targets reproducing everything.

## Step 4: polish  (9 Dec to 15 Dec)
README with figures and honest limitations, CV bullet, cover-letter paragraph, 2-minute project talk track.

## Claims to keep accurate in the CV
Say "lesion-level grouped splits" (not "patient-level"). Say "calibration (ECE, NLL)" only after you measured it.
