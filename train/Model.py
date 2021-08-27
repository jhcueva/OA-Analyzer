import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

num_classes = 5
model_name = "resnet"
feature_extract = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes: int, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        # print(type(model.layer4[:]))
        # layer4 = [*list(models.resnet34(pretrained=use_pretrained).layer4)]
        # # layer3 = [*list(models.resnet50(pretrained=use_pretrained).layer3)[:]]
        # # print(t)
        # # print(type(*list(model.layer4.children())))
        # # layer3 = list(model.children())[6:-3]
        # # model.layer3 = nn.Sequential(*layer3)
        # # layer4 = list(models.resnet50(pretrained=use_pretrained).children())[7:-2]
        # # model.layer3 = nn.Sequential(*layer3)
        # model.layer4 = nn.Sequential(*layer4)
        # layer2 = [*list(models.resnet101(pretrained=True).layer2)]
        # layer3 = [*list(models.resnet101(pretrained=True).layer3)]
        # layer4 = [*list(models.resnet101(pretrained=True).layer4)]
        # model.layer2 = nn.Sequential(*layer2)
        # model.layer3 = nn.Sequential(*layer3)
        # model.layer4 = nn.Sequential(*layer4)
        num_ftrs_fc = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs_fc, 256),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(256, num_classes))

    return model


class ResNetS(nn.Module): #Layer 4, part 3
    def __init__(self, num_classes):
        """Load pretrained Resnet and replace the 4th layer, and fc layer"""
        super(ResNetS, self).__init__()
        # original_model = models.resnet50(pretrained=True)
        # model = [*list(original_model.children())][:-1]
        # self.resnet = nn.Sequential(*model)
        # self.fc = nn.Sequential(nn.Linear(512, 256),
        #                          nn.ReLU(),
        #                          nn.Dropout(p=0.5),
        #                          nn.Linear(256, num_classes))
        # self.fc1 = nn.Linear(512, 256)
        # self.fc2 = nn.Linear(256, num_classes)
        self.model = initialize_model("resnet", num_classes, feature_extract, use_pretrained=True)

    # def activations_hook(self, grad):
    #     self.gradients = grad
    #
    # def get_activations_gradient(self):
    #     return self.gradients
    #
    # def get_activations(self, x):
    #     model = models.resnet34(pretrained=True)
    #     model = [*list(model.children())][:-1]
    #     self.mod = nn.Sequential(model)
    #
    #     x = self.mod(x)
    #
    #     return x
    def once_forward(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        o1 = self.once_forward(input1)
        o2 = self.once_forward(input2)
        return o1, o2


# class ResNet(nn.Module): #Layer 4, part 3
#     def __init__(self, num_classes):
#         """Load pretrained Resnet and replace the 4th layer, and fc layer"""
#         super(ResNet, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#         model_1 = [*list(resnet.children())][:5]
#         self.model_1 = nn.Sequential(*model_1)
#         model_2 = [*list(resnet.children())][5:-1]
#         self.model_2 = nn.Sequential(*model_2)
#         self.ftrs_in = resnet.fc.in_features
#         self.fc1 = nn.Sequential(nn.Linear(self.ftrs_in, 256),
#                                  nn.ReLU(),
#                                  nn.Dropout(p=0.5))
#         self.fc2 = nn.Linear(256, num_classes)
#
#         self.gradients = None
#
#     def activations_hook(self, grad):
#         self.gradients = grad
#
#     def get_activations_gradient(self):
#         return self.gradients
#
#     def get_activations(self, x):
#         self.model_1(x)
#         self.model_2(x)
#         return x
#
#     def get_activations_before(self, x):
#         self.model_1(x)
#         self.model_2(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#
#         return x
#
#     def once_forward(self, x):
#         x = self.model_1(x)
#         x = self.model_2(x)
#         x = x.view(x.size(0), -1)
#
#         x = self.fc1(x)
#
#         x.register_hook(self.activations_hook)
#
#         x = self.fc2(x)
#
#         return x
#
#     def forward(self, input1, input2):
#         o1 = self.once_forward(input1)
#         o2 = self.once_forward(input2)
#
#         return o1, o2

    # def forward(self, input1):
    #     o1 = self.once_forward(input1)
    #
    #     return o1

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
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = x.view(x.size(0), -1)

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

if __name__ == "__main__":
    num_classes = 5
    cuda = True

    feature_extract = True

    device = torch.device('cuda' if cuda else 'cpu')

    # resnet = torchvision.models.resnet34(pretrained=True)
    # print(resnet)
    # model = ResNet(num_classes)
    # print(model)

    # for param in model.parameters():
    #     param.requires_grad = False

    # model = ResNet(resnet, num_classes=num_classes)
    # model = ResnetX(num_classes)

    # if cuda:
    #     model.to(device)

    # params_to_update = model.parameters()
    # print("Params to learn")
    # if feature_extract:
    #     params_to_update = []
    #     for name, param in model.named_parameters():
    #         if param.requires_grad == True:
    #             params_to_update.append(param)
    #             print("\t", name)
    # else:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t", name)
