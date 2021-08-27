import os
from datetime import datetime

import torch

from Metrics import RunningMetric
from Graph import Graph
from CMdata import confusion_matrixx_data


DIR = r"C:\Users\Humberto\Desktop\resultado"
def save_model(state, filename='checkpoint.pth'):
  print("=> Saving model")
  torch.save(state, os.path.join(DIR, filename))

def train_and_evaluate_siamese(model, optimizer, criterion, dataloader, device, num_epoch=3, lr = 0.001):
    start_time = datetime.now()
    knee_pred = []
    knee_true = []
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []

    for epoch in range(num_epoch):
        print('Epoch {}/{} | lr = {}'.format(epoch + 1, num_epoch, optimizer.param_groups[0]['lr']))

        cohen_kappa = RunningMetric()
        mean_square = RunningMetric()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss_train = RunningMetric()
            running_acc_train = RunningMetric()
            running_loss_val = RunningMetric()
            running_acc_val = RunningMetric()
            predcat = torch.empty((64)).to(device)
            targetcat = torch.empty((64)).to(device)
            for lateral, medial, target, file, name in dataloader[phase]:
                lateral, medial, target = lateral.to(device), medial.to(device), target.to(device)  # Pasamos a la GPU
                optimizer.zero_grad()
                batch_size = lateral.size()[0]
                with torch.set_grad_enabled(phase == 'train'):
                    output1, output2= model(lateral, medial)
                    _, predl = torch.max(output1, 1)
                    _, predm = torch.max(output2, 1)
                    loss1 = criterion(output1, target)
                    loss2 = criterion(output2, target)
                    loss = (loss1 + loss2) / 2
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                        pred1 = torch.sum(predl == target).float()
                        pred2 = torch.sum(predm == target).float()
                        pred = (pred1 + pred2) / 2
                        running_loss_train.update(loss.item() * batch_size, batch_size)
                        running_acc_train.update(pred, batch_size)
                if phase == "val":
                    # print(phase)
                    pred1 = torch.sum(predl == target).float()
                    pred2 = torch.sum(predm == target).float()
                    pred = (pred1 + pred2) / 2
                    running_loss_val.update(loss.item() * batch_size, batch_size)
                    running_acc_val.update(pred, batch_size)
                    if epoch + 1 == num_epoch:
                        predcat = torch.cat((predcat, predl), 0)
                        targetcat = torch.cat((targetcat, target), 0)

            if phase == "train":
                acc_train.append(round(running_acc_train().item(), 4))
                loss_train.append(round(running_loss_train(), 4))

            else:
                acc_val.append(round(running_acc_val().item(), 4))
                loss_val.append(round(running_loss_val(), 4))

            if phase == 'val' and epoch + 1 == num_epoch:
                g = predcat.size()[0]-64
                predcat = predcat.narrow(0, 64, g)
                targetcat = targetcat.narrow(0, 64, g)
                knee_pred,  knee_true= confusion_matrixx_data(predcat, targetcat)

        ep = epoch + 1
        save_points = [1 / 4, 2 / 4, 3 / 4, 1]
        checkpoints = [int(num_epoch * save_points[i]) for i in range(len(save_points))]
        if len([*filter(lambda x: ep == x, checkpoints)]) != 0:
            checkpoint = {
                'epoch': num_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val[-1]
            }
            save_model(checkpoint, filename= str(ep)+'_model_RN50_13k_ff.pth')

        print("Lateral Train | Acc = {} - Loss = {}".format(acc_train[-1], loss_train[-1]))
        print("Lateral Val   | Acc = {} - Loss = {}".format(acc_val[-1], loss_val[-1]))

    end_time = datetime.now()

    graph = Graph(DIR, optimizer, lr, num_epoch, start_time, end_time)
    graph.acc_graph(acc_train, acc_val)
    graph.loss_graph(loss_train, loss_val)
    cohen_kappa.cohen_kappa(knee_true, knee_pred)
    mean_square.mean_square(knee_true, knee_pred)
    graph.confusion_matrixx(knee_true, knee_pred, "Confusion matrix")

    return model