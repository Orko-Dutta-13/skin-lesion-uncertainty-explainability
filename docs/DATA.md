# Getting ISIC 2019

The data cannot be fetched by scripts without accepting the licence, so download it by hand.

1. Open <https://challenge.isic-archive.com/data/> and go to the **2019** table.
2. Download these three files (all from the *training* columns):
   - **Task 1, Training Data** (9.1 GB): `ISIC_2019_Training_Input.zip`, 25,331 JPEGs
   - **Task 1, Training Ground Truth** (1 MB): `ISIC_2019_Training_GroundTruth.csv`
   - **Task 2, Training Data** (1 MB): `ISIC_2019_Training_Metadata.csv` (age, sex, anatomic site, lesion identifier)

   Not needed: the test data and test ground truth (different set, no lesion identifier in its
   metadata) and the Task 2 images (same images as Task 1).
3. Put them here and extract the images zip in `data/raw` (the folder name inside the zip does not
   matter; images are found recursively):
   ```
   data/raw/ISIC_2019_Training_Input/ISIC_0000000.jpg ...
   data/raw/ISIC_2019_Training_GroundTruth.csv
   data/raw/ISIC_2019_Training_Metadata.csv
   ```
   Plan for roughly 25 GB of free disk space (zip + extracted images + resized cache). You can
   delete the zip after extracting.
4. Build `data/processed/master.csv` and a resized image cache:
   - Linux/macOS: `make data`
   - Windows (no `make`): `python -m reliable_derm.data.prepare --raw-dir data\raw --out-dir data\processed --resize 256`

Licence: ISIC 2019 is CC BY-NC 4.0 (non-commercial). Fine for a portfolio project; do not commit the
data (`.gitignore` already excludes it) and attribute it as below.

## What the data is

ISIC 2019 training = HAM10000 (Vienna) + BCN20000 (Barcelona) + MSK. Eight classes:
MEL, NV, BCC, AK, BKL, DF, VASC, SCC (the 9th "unknown" class exists only in the challenge test set).

## Two things this repo does that you must be able to explain

**Grouping key.** The metadata provides a common lesion identifier but no patient ID. We group by
`lesion_id` (several images per lesion exist, notably in HAM10000) and treat images with a missing
`lesion_id` as singleton groups. So the split is *lesion-level*, not patient-level. Say that in the
write-up. The prepare step prints the share of images without a lesion_id: check it and report it.

**Source inference.** The CSVs do not say which hospital an image came from. We infer it from the
original pixel size: 600x450 = HAM10000, 1024x1024 = BCN20000, anything else = MSK/other. This is a
heuristic. Check the printed source counts look sensible (three non-trivial groups) and say it is a
heuristic. It powers the `split_shift` protocol (train on HAM+MSK, test on BCN).

## Citations (required by the CC BY-NC 4.0 licence)

"ISIC 2019: Training" is an aggregate and must be attributed as:
- BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona
- HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161
- MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368

Full references:

[1] Tschandl P., Rosendahl C. & Kittler H. The HAM10000 dataset, a large collection of multi-source
dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018).
doi:10.1038/sdata.2018.161

[2] Codella N.C.F., Gutman D., Celebi M.E., Helba B., Marchetti M.A., Dusza S.W., Kalloo A.,
Liopyris K., Mishra N., Kittler H., Halpern A. Skin Lesion Analysis Toward Melanoma Detection: A
Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the
International Skin Imaging Collaboration (ISIC). arXiv:1710.05006 (2017).

[3] Hernández-Pérez C., Combalia M., Podlipnik S., Codella N.C., Rotemberg V., Halpern A.C.,
Reiter O., Carrera C., Barreiro A., Helba B., Puig S., Vilaplana V., Malvehy J. BCN20000:
Dermoscopic lesions in the wild. Scientific Data 11(1):641 (2024).
