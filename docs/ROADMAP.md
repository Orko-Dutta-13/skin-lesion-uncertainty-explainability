# Future work

Planned extensions, not yet done. Results so far are in the README and `reports/NOTES.md`.

## Evaluation
- Per-class bootstrap confidence intervals, in particular for the rare classes DF, VASC and SCC (about 35 or fewer
  test images each), to quantify how noisy their recall is.
- Image-level bootstrap intervals are optimistic; a lesion-level (cluster) bootstrap would be more conservative.

## Uncertainty
- Head-only MC Dropout (`python -m reliable_derm.mc_dropout --mode head`) compared with the full stochastic mode.
- More ensemble members, so the ensemble gets a spread estimate.

## Explanations
- Check whether the black circular dermoscope border, which is more common in some sources, drives Grad-CAM or SHAP attributions.
- A randomisation sanity check for SHAP (done for Grad-CAM only), and more images per class.

## Distribution shift
- Semantic OOD: hold out one diagnosis, retrain and test whether the OOD detectors (kNN, Mahalanobis) flag it as unknown.
- Tune k and the covariance shrinkage of the OOD detectors on a validation shift, and investigate why relative Mahalanobis fails here.
- Other source pairs (for example leave-one-source-out over HAM, BCN and MSK).
- Drift monitoring for a model trained on all sources.
- Separating acquisition effects (colour, borders, resize and crop) from class mix and image difficulty.

## Data
- Patient-level grouping, if a dataset with patient identifiers is used (ISIC 2019 only has lesion identifiers).
