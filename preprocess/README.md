# Preorocess

## How to use
- **Split WSI into patches**: \
    single-scale patch: Run openslide_wsi.py \
    multi-scale patch: Run openslide_wsi_multi.py
- **Split WSIs into some folds for cross-validation**: \
    Run split_dataset_cv.py

## Files
- **openslide_wsi.py**: \
    Split WSI into single-scale patches. Check the patch can be used by checking the corresponding area of ground truth.
- **openslide_wsi_multi.py**: \
    Split WSI into multi-scale patches.
- **split_dataset_cv.py**: \
    Split WSIs into some folds for cross-validation.