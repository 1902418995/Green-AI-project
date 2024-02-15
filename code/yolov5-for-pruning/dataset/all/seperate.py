# 读取train.txt文件，将图片按照类别分别存放在dataset\all\train\下

import os
import shutil

# 读取train.txt文件
with open('train.txt', 'r') as f:
    lines = f.readlines()

# 将读取到的文件名对应的图片存放在dataset\all\train\下

for line in lines:

    # 读取图片路径
    path = os.path.join(line)
    print(path)
    # 如果train文件夹不存在，则创建
    if not os.path.exists('./train'):
        os.makedirs('./train')
    # 将对应的图片复制到dataset\all\train\下
    path = path.strip()
    shutil.copy(path, './train')

# 读取train.txt文件，将每一条数据原本的路径./image/ysm_319.jpg修改为./train/ysm_319.jpg
# with open('val.txt', 'r') as f:
#     lines = f.readlines()
#
# # 将修改后的数据重新写入train.txt文件
# with open('val.txt', 'w') as f:
#     for line in lines:
#         line = line.replace('./image', './val')
#         f.write(line)



