import os
from matplotlib import pyplot as plt
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from inference.model import ResNet
import torch.optim as optim
from inference.loadDataset import LoadDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchvision import transforms
from captum.attr import GuidedGradCam


class Inference:
    def __init__(self, directory):
        self.directory = directory

    def Knee(self):
        lr = 1e-4
        cuda = True
        n_cpu = 2
        num_classes = 5

        device = torch.device('cuda' if cuda else 'cpu')

        model = ResNet(num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        if cuda:
            model.to(device)

        checkpoint = torch.load(os.path.join(os.path.abspath(os.getcwd()), "model.pth"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.BILINEAR),
                transforms.Grayscale(num_output_channels=1),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        test_dataset = LoadDataset(self.directory, split="analyzed", transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=False, num_workers=n_cpu)

        classes = []
        names = []
        model.eval()
        for lateral, medial, file, name in test_loader:
            names.append(name[0])
            lateral, medial = lateral.to(device), medial.to(device)
            lateral_pred, medial_pred = model(lateral), model(medial)
            output1, l_preds = torch.max(lateral_pred, 1)
            output2, m_preds = torch.max(medial_pred, 1)
            # output = lateral_pred.add(medial_pred)
            output = (lateral_pred + medial_pred) / 2
            output = output.cpu()
            output = output.detach().numpy().squeeze()
            classes.append(output)
            guided_gc1 = GuidedGradCam(model, model.model2)
            guided_gc2 = GuidedGradCam(model, model.model2)
            attribution1 = guided_gc1.attribute(lateral, l_preds)
            attribution2 = guided_gc2.attribute(medial, m_preds)

            heatmap1 = torch.mean(attribution1, dim=1)
            heatmap2 = torch.mean(attribution2, dim=1)
            heatmap1, _ = torch.max(heatmap1, 0)
            heatmap2, _ = torch.max(heatmap2, 0)
            heatmap1 /= torch.max(heatmap1)
            heatmap2 /= torch.max(heatmap2)
            heatmap1, heatmap2 = heatmap1.cpu(), heatmap2.cpu()
            heatmap1, heatmap2 = heatmap1.detach().numpy(), heatmap2.detach().numpy()

            img = cv2.imread(file[0])
            width, height, _ = img.shape
            pad = (width // 2) + 16
            lateral = img[0: height, 0: pad]
            medial = img[0: height, (pad - 16): width]
            heatmap1 = cv2.resize(heatmap1, (lateral.shape[1], lateral.shape[0]))
            heatmap2 = cv2.resize(heatmap2, (medial.shape[1], medial.shape[0]))
            heatmap1, heatmap2 = np.uint8(255 * heatmap1), np.uint8(255 * heatmap2)
            heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
            heatmap2 = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)
            heatmap2 = cv2.flip(heatmap2, 1)
            overlay_img1 = cv2.addWeighted(lateral, 0.8, heatmap1, 0.3, 0)
            overlay_img2 = cv2.addWeighted(medial, 0.8, heatmap2, 0.3, 0)
            knee_heatmap = cv2.hconcat([overlay_img1, overlay_img2])
            cv2.imwrite(os.path.join(os.path.join(self.directory, "analyzed"), "heatmap_" + name[0]), knee_heatmap)

        return classes, names

    def bar_graph(self, values, name):
        kl_predicted = values
        kl_classes = ["K0", "K1", "K2", "K3", "K4"]

        plt.bar(kl_classes, kl_predicted, width=1, color='#FFE5CC')

        for i in range(len(kl_predicted)):
            plt.annotate(str(kl_predicted[i]), xy=(kl_classes[i], kl_predicted[i]), ha='center', va='bottom', fontsize=23)

        plt.box(on=None)  # Delete black frame of the image
        plt.yticks([])  # Delete the Y axis
        plt.xticks(fontsize=20)
        plt.tight_layout()
        plt.ylim(0, 1)
        plt.savefig(os.path.join(os.path.join(self.directory, "analyzed"), "stats_"+name),
                    dpi=200,
                    pad_inches=0,
                    bbox_inches='tight',
                    transparent="True")
        plt.close()

    def plot_prediction(self, predictions, names):
        print(predictions)
        if len(predictions) > 1:
            self.bar_graph([round(value, 2) for value in predictions[0]], names[0])
            self.bar_graph([round(value, 2) for value in predictions[1]], names[1])
        else:
            self.bar_graph([round(value, 2) for value in predictions[0]], names[0])