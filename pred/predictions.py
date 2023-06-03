import torch
import torch.nn as nn

import numpy as np
import cv2


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(6)

        self.conv_layer2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(9)

        self.conv_layer3 = nn.Conv2d(in_channels=9, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(12)

        self.conv_layer4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(16)

        self.conv_layer5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(32)

        self.fully_connected1 = nn.Linear(8 * 8 * 32, 256)
        self.fully_connected2 = nn.Linear(256, 64)
        self.fully_connected3 = nn.Linear(64, 2)

        self.dropout_layer = nn.Dropout(p=0.5)

        self.relu_layer = nn.ReLU()
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2)

    def forward(self, input):
        output = self.conv_layer1(input)
        output = self.batchnorm1(output)
        output = self.relu_layer(output)
        output = self.maxpool_layer(output)

        output = self.conv_layer2(output)
        output = self.batchnorm2(output)
        output = self.relu_layer(output)
        output = self.maxpool_layer(output)

        output = self.conv_layer3(output)
        output = self.batchnorm3(output)
        output = self.relu_layer(output)
        output = self.maxpool_layer(output)

        output = self.conv_layer4(output)
        output = self.batchnorm4(output)
        output = self.relu_layer(output)
        output = self.maxpool_layer(output)

        output = self.conv_layer5(output)
        output = self.batchnorm5(output)
        output = self.relu_layer(output)
        output = self.maxpool_layer(output)

        output = output.view(-1, 8 * 8 * 32)

        output = self.fully_connected1(output)
        output = self.relu_layer(output)
        output = self.dropout_layer(output)

        output = self.fully_connected2(output)
        output = self.relu_layer(output)
        output = self.dropout_layer(output)

        output = self.fully_connected3(output)

        return output

def prediction(path):
    trained_model = Network()
    trained_model.load_state_dict(torch.load('model/model.pth'))
    trained_model.eval()
    print()

    class_names=['NOT WEED' ,'WEED']
    image = cv2.imread(str(path))
    image = cv2.resize(image, (256, 256))

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.array(image, dtype=np.float32)
    image = torch.from_numpy(image)
    image = image / 255

    c = trained_model(image)
    a=torch.sigmoid(c)
    _, predictionss = torch.max(c, 1)
    return class_names[int(predictionss)],a
