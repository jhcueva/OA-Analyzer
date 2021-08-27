import secrets
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms
from Load_Dataset import LoadDataset
from torch.utils.data import DataLoader
import PIL.Image as Image

# dir = r'C:\Users\Humberto\Desktop\data\train_eval'

contrast = [i/10 for i in range(3,7)]
factor = [i/100 for i in range(11, 23)]
brightness_factor = secrets.choice(factor)
contrast_factor = secrets.choice(contrast)
saturation_factor = secrets.choice(factor)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(degrees= (-7, 7)),
    transforms.Grayscale(3),
    transforms.ColorJitter(brightness= brightness_factor, contrast= contrast_factor, saturation=saturation_factor, hue=.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.Grayscale(3),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# train_dataset = LoadDataset(dir,transform= transform)
# dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

def show_batch_lateral(dl):
    for lateral, medial, target, _ in dl:
        out = make_grid(lateral)
        plt.imshow(out.permute(1,2,0))
        plt.show()
        out = make_grid(medial)
        plt.imshow(out.permute(1, 2, 0))
        plt.show()

# show_batch_lateral(dataloader)

def show_lateral(dl):
    print(len(dl[0]))
    for lateral, medial, target, file_name in dl:
        print(target)
        print(file_name)

# show_lateral(train_dataset)