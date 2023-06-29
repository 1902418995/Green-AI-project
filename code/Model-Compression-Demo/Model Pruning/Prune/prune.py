# @Time : 2023-6-26
# @Author : Bangguo Xu
# @Version：V 0.1
# @File : prune
# @desc : This is a file to prune the model
import torch
from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from arg_parse import parse_args
from torchinfo import summary
args = parse_args()


def pruning(model, device, config_list):
    print('==========================before pruning==========================')
    print(model)
    summary(model, (1,1,28,28))
    print('==================================================================')
    pruner = L1NormPruner(model, config_list)
    model, masks = pruner.compress()
    # 展示每层结构的掩码稀疏度(此时还没尽兴剪枝)
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
    pruner._unwrap_model()
    ModelSpeedup(model, torch.rand(1, 1, 28, 28).to(device), masks).speedup_model()

    print('==========================after pruning==========================')
    print(model)
    summary(model, (1,1,28,28))
    print('=================================================================')
    # pruner.export_model(model_path="./pts/prune.pth", mask_path="./pts/mask.pth")
    return model, masks


def pruning_main(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_prune, masks = pruning(model, device, args.config_list)
    return model_prune, masks
