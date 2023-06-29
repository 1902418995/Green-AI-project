# @Time : 2023-6-26
# @Author : Bangguo Xu
# @Version：V 0.1
# @File : datas
# @desc : This is a file to load the data
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from arg_parse import parse_args

args = parse_args()

train_data = torchvision.datasets.FashionMNIST(
    root='../data/',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=True
)

val_data = torchvision.datasets.FashionMNIST(
    root='../data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=True
)

train_loader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=True)

if __name__ == '__main__':
    print(train_data.data[0].shape)
