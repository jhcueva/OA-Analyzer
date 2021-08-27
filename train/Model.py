import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class ResNet(nn.Module):  # Layer 4, part 3
    def __init__(self, num_classes):
        """Load pretrained Resnet and replace the 4th layer, and fc layer"""
        super(ResNet, self).__init__()
        resnet = models.resnet34(pretrained=True)
        model1 = [*list(resnet.children())][:4]
        self.model1 = nn.Sequential(*model1)
        model2 = [*list(resnet.children())][4:-2]
        self.model2 = nn.Sequential(*model2)
        self.avg_pool = resnet.avgpool
        self.ftrs_in = resnet.fc.in_features
        self.fc1 = nn.Sequential(nn.Linear(self.ftrs_in, 256),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5))

        self.fc3 = nn.Linear(128, num_classes)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x

    def get_activations_before(self, x):

        x = self.model1(x)
        x = self.model2(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    def once_forward(self, x):
        x = self.model1(x)
        x = self.model2(x)

        # if x.requires_grad:
        #   x.register_hook(self.activations_hook)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        if x.requires_grad:
            # x = x.view(-1, 128)
            x.register_hook(self.activations_hook)

        x = self.fc3(x)

        return x

    def forward(self, input1, input2):
        o1 = self.once_forward(input1)
        o2 = self.once_forward(input2)

        return o1, o2
