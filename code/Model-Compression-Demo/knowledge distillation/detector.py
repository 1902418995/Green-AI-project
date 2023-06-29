# @Time : 2023-6-26
# @Author : Bangguo Xu
# @Version：V 0.1
# @File : detector.py
# @desc :Test the performance difference between the teacher network and the student network

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
from kd.nets import TeacherNet, StudentNet
from torch import nn
import time


class Detector:
    def __init__(self, net_path, isTeacher=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if isTeacher:
            self.net = TeacherNet().to(self.device)
        else:
            self.net = StudentNet().to(self.device)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                      transform=self.trans)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                                     transform=self.trans)
        self.train_data = DataLoader(train_set, batch_size=64, shuffle=True)
        self.test_data = DataLoader(test_set, batch_size=100, shuffle=False)
        self.net.load_state_dict(torch.load(net_path, map_location=self.device))
        self.net.eval()

    def detect(self):
        correct = 0
        start = time.time()
        with torch.no_grad():
            for data, label in self.test_data:
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()
        end = time.time()
        print(f"total time:{end - start}")

        print('Test: average  accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(self.test_data.dataset),
                                                                  100. * correct / len(self.test_data.dataset)))


if __name__ == '__main__':
    print("teacher_net")
    detector = Detector("models/teacher_net.pth")
    detector.detect()
    print("student_net_with_kd")
    detector = Detector("models/student_net(with_kd).pth", False)
    detector.detect()
    print("student_net_without_kd")
    detector = Detector("models/student_net(without_kd).pth", False)
    detector.detect()
