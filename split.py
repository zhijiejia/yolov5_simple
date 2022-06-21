import os
from PIL import Image
from sklearn.model_selection import train_test_split
from shutil import copy

data_root = 'data/new_clean/'
image_root = os.path.join(data_root, 'images')
label_root = os.path.join(data_root, 'labels')

dst_root = 'data/hw'

file_list = []
for filename in os.listdir(image_root):
    if filename[-3:] == 'jpg':
        file_list.append(filename)

file_list.sort(key=lambda x: int(x.split('.')[0]))

index_train, index_val = train_test_split(list(range(0, len(file_list))), test_size=0.2, random_state=2022)

for index in index_train:
    index = file_list[index].split('.')[0]
    src_image_path = os.path.join(image_root, '{0}.jpg'.format(index))
    dst_image_path = os.path.join(dst_root, 'images/train','{0}.jpg'.format(index))

    src_label_path = os.path.join(label_root, '{0}.txt'.format(index))
    dst_label_path = os.path.join(dst_root, 'labels/train','{0}.txt'.format(index))

    image = Image.open(src_image_path).convert("RGB")
    image.save(dst_image_path)

    copy(src_label_path, dst_label_path)

for index in index_val:
    index = file_list[index].split('.')[0]
    src_image_path = os.path.join(image_root, '{0}.jpg'.format(index))
    dst_image_path = os.path.join(dst_root, 'images/val','{0}.jpg'.format(index))

    src_label_path = os.path.join(label_root, '{0}.txt'.format(index))
    dst_label_path = os.path.join(dst_root, 'labels/val','{0}.txt'.format(index))

    image = Image.open(src_image_path).convert("RGB")
    image.save(dst_image_path)

    copy(src_label_path, dst_label_path)

file_handle=open(os.path.join(dst_root, 'train.txt'),mode='w')

train_image_root = os.path.join(dst_root, 'images/train')

for filename in os.listdir(train_image_root):
    path = os.path.join(train_image_root, filename)
    file_handle.write(path+'\n')

file_handle=open(os.path.join(dst_root, 'val.txt'),mode='w')

val_image_root = os.path.join(dst_root, 'images/val')

for filename in os.listdir(val_image_root):
    path = os.path.join(val_image_root, filename)
    file_handle.write(path+'\n')


