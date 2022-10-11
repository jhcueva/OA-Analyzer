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
            output = (lateral_pred + medial_pred) / 2
            print("Output: ", output)
            output = output.cpu()
            output = output.detach().numpy().squeeze()
            classes.append(output)
            guided_gc1 = GuidedGradCam(model, model.model2)
            guided_gc2 = GuidedGradCam(model, model.model2)
            attribution1 = guided_gc1.attribute(lateral, l_preds)
            attribution2 = guided_gc2.attribute(medial, m_preds)

            heatmapLateral = torch.mean(attribution1, dim=1)
            heatmapMedial = torch.mean(attribution2, dim=1)
            heatmapLateral, _ = torch.max(heatmapLateral, 0)
            heatmapMedial, _ = torch.max(heatmapMedial, 0)
            heatmapLateral /= torch.max(heatmapLateral)
            heatmapMedial /= torch.max(heatmapMedial)
            heatmapLateral, heatmapMedial = heatmapLateral.cpu(), heatmapMedial.cpu()
            heatmapLateral, heatmapMedial = heatmapLateral.detach().numpy(), heatmapMedial.detach().numpy()

            img = cv2.imread(file[0])
            width, height, _ = img.shape
            pad = (width // 2) + 16
            lateral = img[0: height, 0: pad]
            # lateral = img[height//4: (height//4)+pad, 0: pad]
            medial = img[0: height, (pad - 16): width]
            # medial = img[height//4: (height//4)+pad, pad: width]
            heatmapLateral = cv2.resize(heatmapLateral, (lateral.shape[1], lateral.shape[0]))
            heatmapMedial = cv2.resize(heatmapMedial, (medial.shape[1], medial.shape[0]))
            heatmapLateral, heatmapMedial = np.uint8(255 * heatmapLateral), np.uint8(255 * heatmapMedial)
            heatmapLateral = cv2.applyColorMap(heatmapLateral, cv2.COLORMAP_JET)
            heatmapMedial = cv2.applyColorMap(heatmapMedial, cv2.COLORMAP_JET)
            heatmapMedial = cv2.flip(heatmapMedial, 1)
            overlay_img1 = cv2.addWeighted(lateral, 0.8, heatmapLateral, 0.3, 0)
            overlay_img2 = cv2.addWeighted(medial, 0.8, heatmapMedial, 0.3, 0)
            knee_heatmap = cv2.hconcat([overlay_img1, overlay_img2])


            # x_offset = 0
            # y_offset = height//4
            # print(knee_heatmap.shape)
            # x_end = x_offset + knee_heatmap.shape[1]
            # y_end = y_offset + knee_heatmap.shape[0]
            # print(y_offset, y_end, x_offset, x_end)
            # img[y_offset: y_end, x_offset, x_end-1] = knee_heatmap
            # cv2.imshow(img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(os.path.join(self.directory, "analyzed"), "heatmap_" + name[0]), knee_heatmap)

        return classes, names

    def bar_graph(self, values, name):
        print("Values: ", values)
        kl_predicted = values
        kl_classes = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]

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