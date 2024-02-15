 # 读取images文件夹下的图片，将每张图片的绝对路径保存到train.txt文件中

import os
import random

# 读取images文件夹下的图片
images_path = './train/images'
# images_path = '/content/data/yolov5-6.0/data/val/images'
images = os.listdir(images_path)
txt_path = './train.txt'
# txt_path = '/content/data/yolov5-6.0/data/val/val.txt'
# 将每张图片的绝对路径保存到train.txt文件中
with open(txt_path, 'w') as f:
    for image in images:
        f.write(os.path.join(os.path.abspath(images_path), image) + '\n')

images_path = './val/images'
# images_path = '/content/data/yolov5-6.0/data/val/images'
images = os.listdir(images_path)
txt_path = './val.txt'
# txt_path = '/content/data/yolov5-6.0/data/val/val.txt'
# 将每张图片的绝对路径保存到train.txt文件中
with open(txt_path, 'w') as f:
    for image in images:
        f.write(os.path.join(os.path.abspath(images_path), image) + '\n')

images_path = './test/images'
# images_path = '/content/data/yolov5-6.0/data/val/images'
images = os.listdir(images_path)
txt_path = './test.txt'
# txt_path = '/content/data/yolov5-6.0/data/val/val.txt'
# 将每张图片的绝对路径保存到train.txt文件中
with open(txt_path, 'w') as f:
    for image in images:
        f.write(os.path.join(os.path.abspath(images_path), image) + '\n')

print('Done!')