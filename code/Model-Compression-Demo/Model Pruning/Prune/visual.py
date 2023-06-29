# @Time : 2023-6-26
# @Author : Bangguo Xu
# @Version：V 0.1
# @File : visual
# @desc : This is a file to plot the loss and accuracy of the original model and the pruned model
import matplotlib.pyplot as plt


def mask_vis(masks):
    for _, mask in masks.items():
        mask = mask['weight'].detach().cpu().numpy()
    print("sparsity:{}".format(mask.sum() / mask.size))
    plt.imshow(mask)


def draw(history, prune_history, epochs):
    # 三个模型的loss和acc分析
    x = list(range(1, epochs + 1))

    plt.subplot(1, 1, 1)
    plt.plot(x, [history[i][1] for i in range(epochs)], label='original model')
    plt.plot(x, [prune_history[i][1] for i in range(epochs)], label='pruned model')
    plt.title('Test accuracy')
    plt.legend()
    plt.savefig("visual0.95.png")
