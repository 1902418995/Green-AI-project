# @Time : 2023-6-26
# @Author : Bangguo Xu
# @Version：V 0.1
# @File : main
# @desc : This is a file to compare the original model and the pruned model
import numpy as np
import torch
from torch import nn
from arg_parse import parse_args
from visual import draw
from Prune.model import Net
import train
import prune
import datas

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = Net().to(device)
    model, history = train.main(model, mode=1)
    model.load_state_dict(torch.load("./pts/origin_model.pth"))
    model_prune, masks = prune.pruning_main(model)
    # 这部分有些冗余，后续修改
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train.test(model_prune, device=device, val_loader=datas.val_loader, loss_func=loss_func)
    _, prune_history = train.main(model_prune, mode=2)

    history = np.array(torch.tensor(history, device='cpu'))
    prune_history = np.array(torch.tensor(prune_history, device='cpu'))
    draw(history, prune_history, args.epoch)
