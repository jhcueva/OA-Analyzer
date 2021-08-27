import os
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_squared_error, precision_score

class RunningMetric:
    def __init__(self):
        self.S = 0
        self.N = 0

    def update(self, val, size):
        self.S += val
        self.N += size

    def __call__(self):
        return self.S/float(self.N)

    def accuracy(self, preds, names):
        correct_prediction = 0
        for i in range(len(names)):
            name, ext = os.path.splitext(names[i])
            name = name[:-2]
            if [*map(name.count, "_")][0] == 2:
                if int(name[0]) == preds[i] or int(name[2]) == preds[i]:
                    correct_prediction += 1
            elif int(name[0]) == preds[i]:
                correct_prediction += 1
        return correct_prediction

    def cohen_kappa(self, knee_true, knee_pred):
        cohen_score = cohen_kappa_score(knee_true, knee_pred, weights="quadratic")
        return print("\t Cohen kappa: {}".format(round(cohen_score, 4)))

    def mean_square(self, knee_true, knee_pred):
        mean_square = mean_squared_error(knee_true, knee_pred)
        return print("\t Mean square: {}".format(round(mean_square, 4)))

    def acc(self, knee_true, knee_pred):
        acc = accuracy_score(knee_true, knee_pred, normalize=False)
        return print("\t Accuracy Score: {}".format(round(acc, 4)))

    def precision_none(self, knee_true, knee_pred):
        precision = precision_score(knee_true, knee_pred, average=None)
        return print("\t Precision_none: {}".format(precision))

    def precision_micro(self, knee_true, knee_pred):
        precision = precision_score(knee_true, knee_pred, average='micro')
        return print("\t Precision_micro: {}".format(precision))