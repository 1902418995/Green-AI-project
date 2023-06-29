# @Time : 2023-6-26
# @Author : Bangguo Xu
# @Version：V 0.1
# @File : model
# @desc : This is a file to build the model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

    def get_loss(self, output, label):
        return self.loss(output, label)


if __name__ == '__main__':
    model = Net()
    print(model)
    summary(model, input_size=(1, 1, 28, 28))
