import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sn
import pandas as pd


class Graph:
    def __init__(self, new_path, optimizer, lr, num_epoch=1, start_time=None, end_time=None):
        self.new_path = new_path
        self.optimizer = optimizer
        self.lr = lr
        self.batchsize = 64
        self.num_epoch = num_epoch
        self.start_time = start_time
        self.end_time = end_time
        self.text = "ft:7-filter_data-ResNet50-bs64"

    def confusion_matrixxN(self, knee_true, knee_pred, name):
        classes = ('KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4')

        cf_matrix = confusion_matrix(knee_true, knee_pred, normalize='all')
        df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis],
                             index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True, cmap="Greens", vmin=0.0, vmax=1.0)
        plt.ylabel("True Label")
        plt.xlabel("Predicted label")
        plt.savefig(os.path.join(self.new_path, str(name) + '.png'),
                    dpi=200)
        plt.close()

    def confusion_matrixx(self, knee_true, knee_pred, name):
        classes = ('KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4')

        cf_matrix = confusion_matrix(knee_true, knee_pred)
        df_cm = pd.DataFrame(cf_matrix.astype('float'),
                             index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True, cmap="Greens")
        plt.ylabel("True Label")
        plt.xlabel("Predicted label")
        plt.savefig(os.path.join(self.new_path, str(name) + '.png'),
                    dpi=200)
        plt.close()

    def ROC_curve(self, knee_true, knee_pred):
        y = label_binarize(knee_pred, classes=[0, 1, 2, 3, 4])
        n_classes = y.shape[1]
        lw = 2
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(knee_pred[:, i], knee_true[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        colors = cycle(['blue', 'red', 'green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class data')
        plt.legend(loc="lower right")
        plt.show()

    def acc_graph(self, train_acc, val_acc):
        plt.plot(range(self.num_epoch), train_acc, label='Train Acc')
        plt.plot(range(self.num_epoch), val_acc, label="Val Acc")
        plt.legend(loc='best')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.suptitle('Accuracy | {} | {} | {}'.format(type(self.optimizer).__name__, self.batchsize, self.lr))
        plt.title('Duration: {} | Dist = {} | Max = {}'.format(
            self.end_time - self.start_time,
            round(abs(train_acc[-1] - val_acc[-1]), 3),
            max(train_acc)))
        plt.savefig(os.path.join(self.new_path, 'Accuracy - lr_{} - bs_{} - ep_{} {}.png'.format(
            # type(self.optimizer).__name__,
            self.lr,
            self.batchsize,
            self.num_epoch,
            self.text)),
                    dpi=200)
        plt.close()

    def loss_graph(self, train_loss, val_loss):
        plt.plot(range(self.num_epoch), train_loss, label="Train Loss")
        plt.plot(range(self.num_epoch), val_loss, label='Loss Val')
        plt.legend(loc='best')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.suptitle('Loss | {} | {} | {}'.format(type(self.optimizer).__name__, self.batchsize, self.lr))
        plt.title('Duration: {} | Dist = {} | Min = {}'.format(
            self.end_time - self.start_time,
            round(abs(train_loss[-1] - val_loss[-1]), 3),
            min(train_loss)))
        plt.savefig(os.path.join(self.new_path, 'Loss - lr_{} - bs_{} - ep_{} {}.png'.format(
            # type(self.optimizer).__name__,
            self.lr,
            self.batchsize,
            self.num_epoch,
            self.text)),
                    dpi=200)
        plt.close()