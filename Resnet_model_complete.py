from Resnet_block import *
import torch.nn as nn
import torch

#Resnet for captcha detection complete model

class ResNetCaptcha(nn.Module):
    def __init__(self, layers, num_classes=36, captcha_length=6):
        super(ResNetCaptcha, self).__init__()
        self.in_channels = 64
        self.captcha_length = captcha_length
        self.dropout = nn.Dropout(0.3)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        feature_dim = 512 * Resblock.expansion
        self.classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_classes) for _ in range(captcha_length)
        ])

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Resblock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Resblock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Resblock.expansion)
            )

        layers = [Resblock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * Resblock.expansion

        for _ in range(1, blocks):
            layers.append(Resblock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.dropout(x)

        outputs = [classifier(x) for classifier in self.classifiers]
        return torch.stack(outputs, dim=1)
