# Classification of Chemotherapy Pathological Image
2022年卒-備瀬研-荒木健吾

---
## Task
- 3-class classification
- Calculate residual tumor ratio of each WSI \
residual tumor ratio: Residual tumor/ (Residual turmor + Tumor bed)

## Dataset
Chemotherapy Whole Slide Image (WSI)

**Class:**
- Non-Neoplasm
- Tumor bed
- Residual tumor

**Extension:** \
WSI (.ndpi), Annotation file (.xml)

**Number of samples:** \
(March 9th, 2022)
- WSI: 112
- Case (Patient): 55

**About ndpi/xml name:** \
e.g., H〇〇-〇〇〇〇〇_■.ndpi/xml
- H〇〇-〇〇〇〇〇: Case name
- ■: Serial number of the case 


## Directory
- **analysis**: \
    To calculate residual tumor ratio
- **annotation**: \
    To preprocess and visualize annotated area from original WSI (.ndpi)annotation file (.xml)
- **config**: \
    Config files of experiments (src & src_multi)
- **environment**: \
    Environment files for this project (Anaconda)
- **preprocess**: \
    Preprocess for an experiment. Split into patches basis on its annotated class. 
- **src**: \
    Experimet files for single-scale model
- **src_multi**: \
    Experiment files for multi-scale model
- **visualize**: \
    To modify a figure
---