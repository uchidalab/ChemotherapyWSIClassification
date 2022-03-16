import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score


# TODO: 動作確認

class Metric(object):
    def __init__(self, argmax=False):
        self.argmax = argmax

    def preprocess(self, preds, labels):
        preds = preds.cpu().detach().numpy().copy()
        labels = labels.cpu().detach().numpy().copy()

        if self.argmax:
            preds = preds.argmax(dim=1)

        assert preds.shape == labels.shape, \
            f"inputs' shape does not match, (preds: {preds.shape}, labels: {labels.shape})"
        return preds, labels

    def accuracy(self, preds, labels):
        preds, labels = self.preprocess(preds, labels)
        return accuracy_score(labels, preds)

    def precision(self, preds, labels):
        preds, labels = self.preprocess(preds, labels)
        return precision_score(labels, preds)

    def recall(self, preds, labels):
        preds, labels = self.preprocess(preds, labels)
        return recall_score(labels, preds)

    def f1_score(self, preds, labels):
        preds, labels = self.preprocess(preds, labels)
        return f1_score(labels, preds)

    def jaccard_score(self, preds, labels):
        preds, labels = self.preprocess(preds, labels)
        return jaccard_score(labels, preds)


class evalMet(object):
    def Accuracy(self, cm):
        len_label = len(cm)
        inter_all = 0
        for idx in range(len_label):
            inter_all += cm[idx][idx]
        return inter_all / np.sum(cm)

    def Precision(self, cm):
        len_label = len(cm)
        iou = 0
        for idx in range(len_label):
            inter = cm[idx][idx]
            iou += inter / np.sum(cm[:, idx])
        return iou / len_label

    def Recall(self, cm):
        len_label = len(cm)
        iou = 0
        for idx in range(len_label):
            inter = cm[idx][idx]
            iou += inter / np.sum(cm[idx, :])
        return iou / len_label

    def F1(self, cm):
        precision = self.Precision(cm)
        recall = self.Recall(cm)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def mIoU(self, cm):
        len_label = len(cm)
        iou = 0
        for idx in range(len_label):
            inter = cm[idx][idx]
            iou += inter / (np.sum(cm[idx, :]) + np.sum(cm[:, idx]) - inter)
        return iou / len_label
