# @Time : 2020-04-19 13:14 
# @Author : Ben 
# @Version：V 0.1
# @File : nets.py
# @desc : define the student net and teacher net

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
class new_TeacherNet(nn.Module):
    def __init__(self, hidden_dims):
        super(new_TeacherNet, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Conv2d(in_channels=1, out_channels=hidden_dims[i], kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels=hidden_dims[i-1], out_channels=hidden_dims[i], kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.cnn = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_dims[-1]*3*3, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
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

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)




    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.relu(self.fc3(x))
        return output

if __name__ == '__main__':
    hidden_dims = [32, 64, 128]
    model_teacher = TeacherNet()
    print(model_teacher)
    summary(model_teacher, input_size=(1, 1, 28, 28))
    print(model_student)
    summary(model_student, input_size=(1, 1, 28, 28))