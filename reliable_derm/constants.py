"""Shared constants. Class order follows the ISIC 2019 ground-truth CSV header."""

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
}
NUM_CLASSES = len(CLASSES)

# Dermoscopy image sizes that identify the source dataset (see docs/DATA.md).
# HAM10000: 600x450, BCN20000: 1024x1024, everything else (MSK): various sizes.
SOURCE_BY_SIZE = {(600, 450): "HAM", (1024, 1024): "BCN"}
SOURCE_OTHER = "MSK_other"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
