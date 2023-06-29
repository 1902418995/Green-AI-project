# @Time : 2023-6-26
# @Author : Bangguo Xu
# @Version：V 0.1
# @File : trainer.py
# @desc : Train the teacher network and the student network
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
from kd.nets import TeacherNet, StudentNet
from torch import nn
from torch.functional import F


class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher_net = TeacherNet().to(self.device)
        self.student_net = StudentNet().to(self.device)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
           transforms.Normalize([0.1307], [0.3081])
        ])
        # self.train_data = DataLoader(datasets.MNIST("../datasets/", train=True, transform=self.trans, download=False),
        #                              batch_size=1000, shuffle=True, drop_last=True)
        # self.test_data = DataLoader(datasets.MNIST("../datasets/", False, self.trans, download=False), batch_size=10000,
        #                             shuffle=True)
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                      transform=torchvision.transforms.ToTensor())
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                                     transform=torchvision.transforms.ToTensor())
        self.train_data = DataLoader(train_set, batch_size=64, shuffle=True)
        self.test_data = DataLoader(test_set, batch_size=64, shuffle=False)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.KLDivLoss = nn.KLDivLoss()
        self.teacher_optimizer = torch.optim.Adam(self.teacher_net.parameters())
        self.student_optimizer = torch.optim.Adam(self.student_net.parameters())
        self.epochs = 10

    def train_teacher(self):
        self.teacher_net.train()
        # calculate the loss and accuracy
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(self.epochs):
            total = 0
            train_loss = 0
            train_accuracy = 0
            for i, (data, label) in enumerate(self.train_data):
                data, label = data.to(self.device), label.to(self.device)
                output = self.teacher_net(data)
                loss = self.CrossEntropyLoss(output, label)
                self.teacher_optimizer.zero_grad()
                loss.backward()
                self.teacher_optimizer.step()
                train_loss += loss.item()
                train_accuracy += (output.argmax(dim=1) == label).sum().item()
                total += len(data)
                progress = math.ceil(i / len(self.train_data) * 50)
                print("\rTrain teacher_net epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')
            torch.save(self.teacher_net.state_dict(), "models/teacher_net.pth")
            train_losses.append(train_loss / len(self.train_data))
            train_accuracies.append(train_accuracy / len(self.train_data.dataset))
            # print the train loss and accuracy
            print("\nTrain teacher_net epoch %d: loss %.4f, accuracy %.4f" %
                    (epoch, train_loss / len(self.train_data), train_accuracy / len(self.train_data.dataset)))
            test_accuracy, test_loss = self.evaluate(self.teacher_net, "teacher")
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        return train_losses, train_accuracies, test_losses, test_accuracies

    def train_student_without_distillation(self):
        self.student_net.train()
        # calculate the loss and accuracy
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        for epoch in range(self.epochs):
            total = 0
            train_loss = 0
            train_accuracy = 0
            for i, (data, label) in enumerate(self.train_data):
                data, label = data.to(self.device), label.to(self.device)
                output = self.student_net(data)
                loss = self.CrossEntropyLoss(output, label)
                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()
                train_loss += loss.item()
                train_accuracy += (output.argmax(dim=1) == label).sum().item()
                total += len(data)

                progress = math.ceil(i / len(self.train_data) * 50)
                print("\rTrain student_net_without_distillation epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')
            torch.save(self.student_net.state_dict(), "models/student_net(without_kd).pth")
            train_losses.append(train_loss / len(self.train_data))
            train_accuracies.append(train_accuracy / len(self.train_data.dataset))
            # print the train loss and accuracy
            print("\nTrain teacher_net epoch %d: loss %.4f, accuracy %.4f" %
                  (epoch, train_loss / len(self.train_data), train_accuracy / len(self.train_data.dataset)))
            test_accuracy, test_loss = self.evaluate(self.student_net, "student")
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        return train_losses, train_accuracies, test_losses, test_accuracies


    def train_student_with_distillation(self):
        self.student_net.train()
        # calculate the loss and accuracy
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        for epoch in range(self.epochs):
            total = 0
            train_loss = 0
            train_accuracy = 0
            for i, (data, label) in enumerate(self.train_data):
                data, label = data.to(self.device), label.to(self.device)
                teacher_output = self.teacher_net(data)
                student_output = self.student_net(data)
                teacher_output = teacher_output.detach()
                loss = self.distillation(student_output, label, teacher_output, temp=5.0, alpha=0.7)
                # loss = self.KLDivLoss(student_output, teacher_output)
                # loss = self.CrossEntropyLoss(output, label)
                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()
                train_loss += loss.item()
                train_accuracy += (student_output.argmax(dim=1) == label).sum().item()

                total += len(data)
                progress = math.ceil(i / len(self.train_data) * 50)
                print("\rTrain student_net_with_distillation epoch %d: %d/%d, [%-51s] %d%%" %
                      (epoch, total, len(self.train_data.dataset),
                       '-' * progress + '>', progress * 2), end='')
            torch.save(self.student_net.state_dict(), "models/student_net(with_kd).pth")
            train_losses.append(train_loss / len(self.train_data))
            train_accuracies.append(train_accuracy / len(self.train_data.dataset))
            # print the train loss and accuracy
            print("\nTrain teacher_net epoch %d: loss %.4f, accuracy %.4f" %
                  (epoch, train_loss / len(self.train_data), train_accuracy / len(self.train_data.dataset)))
            test_accuracy, test_loss = self.evaluate(self.student_net, "student")
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        return train_losses, train_accuracies, test_losses, test_accuracies

    def evaluate(self, net, model_flag):
        if model_flag == "teacher":
            self.teacher_net.eval()
        else:
            self.student_net.eval()
        test_loss = 0
        test_accuracy = 0
        for data, label in self.test_data:
            data, label = data.to(self.device), label.to(self.device)
            output = net(data)
            test_loss += self.CrossEntropyLoss(output, label).item()
            test_accuracy += (output.argmax(dim=1) == label).sum().item()
        test_loss /= len(self.test_data)
        test_accuracy /= len(self.test_data.dataset)
        print(f"\nTest loss:{test_loss:.4f}, Test accuracy:{test_accuracy * 100:.2f}%")
        return test_accuracy, test_loss

    def distillation(self, y, labels, teacher_scores, temp, alpha):
        return self.KLDivLoss(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


if __name__ == '__main__':
    trainer = Trainer()
    teacher_train_losses, teacher_train_accuracies, teacher_test_losses, teacher_test_accuracies = trainer.train_teacher()
    student_train_losses_without_distillation, student_train_accuracies_without_distillation, student_test_losses_without_distillation, student_test_accuracies_without_distillation = trainer.train_student_without_distillation()
    student_train_losses, student_train_accuracies, student_test_losses, student_test_accuracies = trainer.train_student_with_distillation()
    # plot the teacher train accuracy, student train accuracy and student train accuracy without distillation in one figure
    plt.figure()
    plt.plot(teacher_train_accuracies, label="teacher_train_accuracy")
    plt.plot(student_train_accuracies, label="student_train_accuracy_with_distillation")
    plt.plot(student_train_accuracies_without_distillation, label="student_train_accuracy_without_distillation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("train accuracy")
    # Set the unit length of the x-axis to 1
    plt.xticks(np.arange(0, 10, 1))
    # save the figure
    plt.savefig("train_accuracy_linear.png")
    plt.show()
    # plot the teacher test accuracy, student test accuracy and student test accuracy without distillation in one figure
    plt.figure()
    plt.plot(teacher_test_accuracies, label="teacher_test_accuracy")
    plt.plot(student_test_accuracies, label="student_test_accuracy_with_distillation")
    plt.plot(student_test_accuracies_without_distillation, label="student_test_accuracy_without_distillation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("test accuracy")
    # Set the unit length of the x-axis to 1
    plt.xticks(np.arange(0, 10, 1))
    plt.savefig("test_accuracy_linear.png")
    plt.show()
    # save the figure
    # plot the teacher train loss, student train loss and student train loss without distillation in one figure
    plt.figure()
    plt.plot(teacher_train_losses, label="teacher_train_loss")
    plt.plot(student_train_losses, label="student_train_loss_with_distillation")
    plt.plot(student_train_losses_without_distillation, label="student_train_loss_without_distillation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss")
    # Set the unit length of the x-axis to 1
    plt.xticks(np.arange(0, 10, 1))
    # save the figure
    plt.savefig("train_loss_linear.png")
    plt.show()
    # plot the teacher test loss, student test loss and student test loss without distillation in one figure
    plt.figure()
    plt.plot(teacher_test_losses, label="teacher_test_loss")
    plt.plot(student_test_losses, label="student_test_loss_with_distillation")
    plt.plot(student_test_losses_without_distillation, label="student_test_loss_without_distillation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("test loss")
    # Set the unit length of the x-axis to 1
    plt.xticks(np.arange(0, 10, 1))
    # save the figure
    plt.savefig("test_loss_linear.png")
    plt.show()



