import torch
import os
from tqdm import tqdm
from Metrics import RunningMetric
from Graph import Graph
from datetime import datetime
from CMdata import confusion_matrixx_data


DIR = r"C:\Users\Humberto\Desktop\resultado"
def save_model(state, filename='checkpoint.pth'):
  # print("=> Saving model")
  torch.save(state, os.path.join(DIR, filename))

def train_and_evaluate_siamese(model, optimizer, criterion, dataloader, device, num_epoch=3, lr = 0.001):
    start_time = datetime.now()
    knee_pred = []
    knee_true = []
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    acc = [0.30]
    loss = []

    for epoch in range(num_epoch):
        # print('Epoch {}/{} | lr = {}'.format(epoch + 1, num_epoch, lr))
        print('Epoch {}/{} | lr = {}'.format(epoch + 1, num_epoch, optimizer.param_groups[0]['lr']))
        # torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1).step()


        cohen_kappa = RunningMetric()
        mean_square = RunningMetric()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss_train = RunningMetric()  # Perdida
            running_acc_train = RunningMetric()  # Precision
            running_loss_val = RunningMetric()  # Perdida
            running_acc_val = RunningMetric()  # Precision
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
                    # print(phase)
                    if phase == "train":
                        # print(phase)
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
                # print(predcat)
                targetcat = targetcat.narrow(0, 64, g)
                # print(targetcat)
                knee_pred,  knee_true= confusion_matrixx_data(predcat, targetcat)


        ep = epoch + 1
        # save_points = [1 / 4, 2 / 4, 3 / 4, 1]
        # checkpoints = [int(num_epoch * save_points[i]) for i in range(len(save_points))]
        # if len([*filter(lambda x: ep == x, checkpoints)]) != 0:
        #     checkpoint = {
        #         'epoch': num_epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss_val[-1]
        #     }
        #     save_model(checkpoint, filename= str(ep)+'_model_RN50_13k_ff.pth')

        # if epoch+1 == 1:
        #   checkpoint = {
        #           'epoch': num_epoch,
        #           'model_state_dict': model.state_dict(),
        #           'optimizer_state_dict': optimizer.state_dict(),
        #           'loss': loss_val[-1]
        #   }
        #   save_model(checkpoint, filename= str(1)+'_model_RN50_13k_ff.pth')

        print("Lateral Train | Acc = {} - Loss = {}".format(acc_train[-1], loss_train[-1]))
        print("Lateral Val   | Acc = {} - Loss = {}".format(acc_val[-1], loss_val[-1]))
        print(acc)
        print(acc_val[-1])
        if acc_val[-1] > max(acc) :
            checkpoint = {
                  'epoch': num_epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss_val[-1]
            }

            save_model(checkpoint, filename= str(ep)+'_model_RN50_13k_ff.pth')

        acc.append(acc_val[-1])

    end_time = datetime.now()

    graph = Graph(DIR, optimizer, lr, num_epoch, start_time, end_time)
    graph.acc_graph(acc_train, acc_val)
    graph.loss_graph(loss_train, loss_val)
    cohen_kappa.cohen_kappa(knee_true, knee_pred)
    mean_square.mean_square(knee_true, knee_pred)
    # graph.confusion_matrixx(knee_true, knee_pred, "Confusion matrix")

    return model

def train_and_evaluate(model, optimizer, criterion, dataloader, device, num_epoch=3, lr = 0.001):
    start_time = datetime.now()
    lateral_loss_train = []
    medial_loss_train = []
    lateral_acc_train = []
    medial_acc_train = []
    lateral_loss_val = []
    medial_loss_val = []
    lateral_acc_val = []
    medial_acc_val = []

    confusion_matrix = RunningMetric()
    cohen_kappa = RunningMetric()
    mean_square = RunningMetric()

    # for epoch in range(num_epoch):
    for epoch in tqdm(range(num_epoch), desc="Progress"):
        # print('Epoch {}/{} | lr = {}'.format(epoch + 1, num_epoch, optimizer.param_groups[0]['lr']))
        # lr_scheduler.step()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            lateral_running_loss_train = RunningMetric()  # Perdida
            lateral_running_acc_train = RunningMetric()  # Precision
            lateral_running_loss_val = RunningMetric()  # Perdida
            lateral_running_acc_val = RunningMetric()  # Precision

            medial_running_loss_train = RunningMetric()  # Perdida
            medial_running_acc_train = RunningMetric()  # Precision
            medial_running_loss_val = RunningMetric()  # Perdida
            medial_running_acc_val = RunningMetric()  # Precision

            for lateral, medial, target, file_name, name in dataloader[phase]:
                lateral, medial, target = lateral.to(device), medial.to(device), target.to(device)  # Pasamos a la GPU
                optimizer.zero_grad()  # Llevar a cero el optimizado
                with torch.set_grad_enabled(phase == 'train'):
                    lateral_output, medial_output = model(lateral), model(medial)
                    _, lateral_preds = torch.max(lateral_output, 1)
                    _, medial_preds = torch.max(medial_output, 1)
                    lateral_loss = criterion(lateral_output, target)
                    medial_loss = criterion(medial_output, target)

                    if phase == 'train':
                        lateral_loss.backward()  #Gradientes calculados automaticamente
                        medial_loss.backward()  #Gradientes calculados automaticamente
                        #Gradients updated
                        # for name, param in model.named_parameters():
                        #     print(name, param.grad)
                        optimizer.step()  #Actualiza las periillas con los parametros

                        batch_size = lateral.size()[0]
                        lateral_running_loss_train.update(lateral_loss.item() * batch_size, batch_size)
                        medial_running_loss_train.update(medial_loss.item() * batch_size, batch_size)
                        lateral_running_acc_train.update(torch.sum(lateral_preds == target).float(), batch_size)
                        medial_running_acc_train.update(torch.sum(medial_preds == target).float(), batch_size)

                batch_size = lateral.size()[0]
                lateral_running_loss_val.update(lateral_loss.item() * batch_size, batch_size)
                medial_running_loss_val.update(medial_loss.item() * batch_size, batch_size)
                lateral_running_acc_val.update(torch.sum(lateral_preds == target), batch_size)
                medial_running_acc_val.update(torch.sum(medial_preds == target), batch_size)

            if phase == 'train':
                lateral_acc_train.append(round(lateral_running_acc_train().item(), 4))
                medial_acc_train.append(round(medial_running_acc_train().item(), 4))
                lateral_loss_train.append(round(lateral_running_loss_train(), 4))
                medial_loss_train.append(round(medial_running_loss_train(), 4))
            else:
                lateral_acc_val.append(round(lateral_running_acc_val().item(), 4))
                medial_acc_val.append(round(lateral_running_acc_val().item(), 4))
                lateral_loss_val.append(round(lateral_running_loss_val(), 4))
                medial_loss_val.append(round(lateral_running_loss_val(), 4))

            if phase == 'val' and epoch+1 == num_epoch:
                lateral_knee_pred,  lateral_knee_true= confusion_matrixx_data(lateral_preds, target)
                # medial_knee_true,  medial_knee_pred= confusion_matrixx_data(medial_preds, target)
        # torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1).step()

        ep = epoch + 1
        save_points = [1 / 4, 2 / 4, 3 / 4, 1]
        checkpoints = [int(num_epoch * save_points[i]) for i in range(len(save_points))]
        if len([*filter(lambda x: ep == x, checkpoints)]) != 0:
            checkpoint = {
                'epoch': num_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': lateral_loss_val[-1]
            }
            save_model(checkpoint, filename= str(ep)+'_model.pth')

        # print("Lateral Train | Acc = {} - Loss = {}".format(lateral_acc_train[-1], lateral_loss_train[-1]))
        # print("Medial  Train | Acc = {} - Loss = {}".format(medial_acc_train[-1], medial_loss_train[-1]))
        # print("Lateral Val   | Acc = {} - Loss = {}".format(lateral_acc_val[-1], lateral_loss_val[-1]))
        # print("Medial  Val   | Acc = {} - Loss = {}".format(medial_acc_val[-1], medial_loss_val[-1]))

    end_time = datetime.now()

    graph = Graph(DIR, optimizer, lr, num_epoch, start_time, end_time)
    graph.acc_graph(lateral_acc_train, lateral_acc_val)
    graph.loss_graph(lateral_loss_train, lateral_loss_val)
    cohen_kappa.cohen_kappa(lateral_knee_true, lateral_knee_pred)
    mean_square.mean_square(lateral_knee_true, lateral_knee_pred)
    graph.confusion_matrixx(lateral_knee_true, lateral_knee_pred, "Confusion matrix")

    return model