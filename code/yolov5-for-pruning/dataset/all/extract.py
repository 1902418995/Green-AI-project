# 读取train.txt中的图像文件名，在labels文件夹中找到对应的标签文件，复制到train文件夹中

import os
import shutil

# 读取train.txt中的图像文件名
with open('val.txt', 'r') as f:
    train_list = f.readlines()
    train_list = [i.strip() for i in train_list]
    # 去掉文件名后缀
    train_list = [i[:-4] for i in train_list]
    print(train_list)


# 在labels文件夹中找到对应的标签文件，复制到train文件夹中
for i in train_list:
    shutil.copyfile('./labels/'+i[8:]+'.txt', './val/labels/'+i[8:]+'.txt')

