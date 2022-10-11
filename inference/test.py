import os
from PIL import ImageFile

import torch
import torch.nn as nn
from torchvision import models, transforms

import torch.optim as optim
from torch.utils.data import DataLoader


ImageFile.LOAD_TRUNCATED_IMAGES = True
from train.Model import ResNet

from train.Metrics import RunningMetric
from train.Load_Dataset import LoadDataset
from train.CMdata import confusion_matrixx_data
from train.Graph import Graph


def no_doubles(lista):
    return list(set(lista))


if __name__ == "__main__":
    PATH = r"C:\Users\Humberto\Desktop\Inference\model.pth"
    DIR = r"C:\Users\Humberto\Desktop\metrics\metrics" #test_manual_224
    IMG_DIR = r"C:\Users\Humberto\Desktop\metrics"

    batchsize = 16
    lr = 1e-4
    cuda = True
    n_cpu = 4
    num_classes = 5

    feature_extract = False

    device = torch.device('cuda' if cuda else 'cpu')

    model = ResNet(num_classes)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)

    if cuda:
        model.to(device)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = LoadDataset(DIR, transform=transform, split='test_Norm_min')
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batchsize, num_workers=n_cpu)

    values = []

    # for _ in tqdm(range(5), desc="Progress"):
    lateral_ck = RunningMetric()
    medial_ck = RunningMetric()
    lateral_ms = RunningMetric()
    medial_ms = RunningMetric()
    lateral_cm = RunningMetric()
    medial_cm = RunningMetric()
    lateral_acc = RunningMetric()
    medial_acc = RunningMetric()

    correct = 0
    prediction = []
    label = []
    total = 0
    file_correct = []
    file_incorrect = []

    model.eval()
    # with torch.no_grad():
    predcat = torch.empty(batchsize).to(device)
    targetcat = torch.empty(batchsize).to(device)
    for lateral , medial, target, file, name in test_loader:
        lateral, medial, target = lateral.to(device), medial.to(device), target.to(device)
        lateral_pred, medial_pred = model(lateral, medial)
        pred = lateral_pred + medial_pred
        _, preds = torch.max(pred, 1)

        total += target.size()[0]
        label.extend(target.cpu().numpy())
        prediction.extend(preds.cpu().numpy())

        predcat = torch.cat((predcat, preds), 0)
        targetcat = torch.cat((targetcat, target), 0)

        correct += torch.sum(preds == target).float()

        for i in range(target.size()[0]):
            # print(type(preds.cpu().numpy()[i]))
            # print(name[i])
            # print(type(name[i][0]))
            if preds.cpu().numpy()[i] == int(name[i][0]):
                file_correct.append(name[i])
            else:
                file_incorrect.append(name[i])


        # for i in range(target.size()[0]):
        #     ax = plt.gca()
        #     ax.axis('off')
        #     image = plt.imread(os.path.join(r"C:\Users\Humberto\Desktop\data\34k_train_eval\test",name[i]))
        #     plt.suptitle("Pred: {} - Label: {}".format(preds.cpu().numpy()[i], target.cpu().numpy()[i]),
        #                  color= color_title(preds.cpu().numpy()[i], name[i]))
        #     plt.title("Filename: {}".format(name[i]))
        #     plt.imshow(image)
        #     # plt.show()
        #     plt.savefig(os.path.join(IMG_DIR, name[i]),
        #                 dpi=200,
        #                 pad_inches=0,
        #                 bbox_inches='tight')
        #     plt.close()

    val = predcat.size()[0]-batchsize
    predcat = predcat.narrow(0, batchsize, val)
    targetcat = targetcat.narrow(0, batchsize, val)

    knee_pred,  knee_true= confusion_matrixx_data(predcat, targetcat)
    graph = Graph(DIR, None, lr, None, None, None)
    cohen_kappa = RunningMetric()
    mean_square = RunningMetric()
    accuracy = RunningMetric()
    precision_none = RunningMetric()
    precision_micro = RunningMetric()

    correct = correct.cpu().numpy()
    print(correct)
    print(len(file_correct))
    print(sorted(file_correct))
    print(sorted(file_incorrect))

    acc = round((correct/total), 2)

    values.append(acc)

    print("No sorted: {} - Dist: {}".format(no_doubles(values), (sorted(no_doubles(values))[-1] - sorted(no_doubles(values))[0])))
    print("Sorted: {} - Dist: {}".format(sorted(no_doubles(values)), (sorted(no_doubles(values))[-1] - sorted(no_doubles(values))[0])))

    graph.confusion_matrixx(knee_true, knee_pred, "Confusion matrix")
    graph.confusion_matrixxN(knee_true, knee_pred, "Confusion matrixN")
    cohen_kappa.cohen_kappa(knee_true, knee_pred)
    mean_square.mean_square(knee_true, knee_pred)
    accuracy.acc(knee_true, knee_pred)
    precision_none.precision_none(knee_true, knee_pred)
    precision_micro.precision_micro(knee_true, knee_pred)
