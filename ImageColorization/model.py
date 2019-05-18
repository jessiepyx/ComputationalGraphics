import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # resnet that deals with gray images
        resnet = models.resnet18(num_classes=365)
        resnet.conv1.weight = nn.Parameter(torch.Tensor(64, 1, 7, 7))

        # layer1, output_shape: [None, 128, img_h, img_w]
        self.layer1 = nn.Sequential(*list(resnet.children())[0:6])

        # define colorization operators

        # layer 2
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        # layer 3
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()

        # layer 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()

        # layer 5
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.Sigmoid()

        # layer 6
        self.conv6 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, l_input):
        """
        l_input: [None, 1, img_h, img_w]
        """
        # layer 1
        features = self.layer1(l_input)

        # set colorization operations

        # layer 2
        x = F.interpolate(self.act2(self.bn2(self.conv2(features))), scale_factor=2)
        # layer 3
        x = self.act3(self.bn3(self.conv3(x)))
        # layer 4
        x = F.interpolate(self.act4(self.conv4(x)), scale_factor=2)
        # layer 5
        x = self.act5(self.conv5(x))
        # layer 6
        x = F.interpolate(self.conv6(x), scale_factor=2)
        return x
