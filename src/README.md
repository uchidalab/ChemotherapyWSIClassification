# src

## How to use
- **To train the model**: \
    Run train.py
- **To test the model**: \
    Run test.py or test2.py
- **To make segmentation map (prediction map)**: \
    Run predict_faster.py. (Exectuion time of predict.py is much slower than predict_faster.py)

## Files:
- **dataset.py**: \
    torch.utils.data.Dataset for WSI. It is used in Dataloader.
- **eval.py**: \
    Evaluation functions that are used at training/validation/test time.
- **metrics.py**: \
    Metrics for an experiment.
- **model.py**: \
    Models for an experiment.
- **predict_faster.py**: \
    To make segmentation (prediction) map by using the trained model.
- **predict.py**: \
    This is the old version of predict_faster.py. Exectution time of this code is much slower than predict_faster.py. So please use predict_faster.py.
- **test.py**: \
    To test the trained model.
- **test2.py**: \
    To test the trained model. It can calculate cross-validation results.
- **train.py**: \
    To train a classification model.
- **util.py**: \
    Some functions that are used in this directory.