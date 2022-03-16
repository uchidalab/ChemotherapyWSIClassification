# annotation

## How to use
Just run main.py. \
Preprocess and visualize the annotation data (.xml) from original WSI (.ndpi)


## Files
- **main.py**: \
    Run this code if you want to preprocess and visualize annotation data (.xml).
- **main_bg_mask.py**: \
    To make tissue mask (gray-scale image). It divides tissue area and background area.
- **main_gt_from_xml.py**: \
    To make groud truth image from annnotation data (.xml).
- **make_overlaid_mask.py**: \
    To make colored groud truth image.
- **util.py**: \
    Some functions that are used in this directory.