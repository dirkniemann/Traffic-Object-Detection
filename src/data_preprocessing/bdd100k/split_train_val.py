"""
This script splits a dataset of images and labels into training and validation sets.
"""
import os
import shutil
import random

data_dir = 'data/bdd100k'
val_ratio=0.2

def split_train_val(data_dir):
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    train_images_dir = os.path.join(data_dir, 'images', 'train')
    train_labels_dir = os.path.join(data_dir, 'labels', 'train')
    val_images_dir = os.path.join(data_dir, 'images', 'val')
    val_labels_dir = os.path.join(data_dir, 'labels', 'val')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and os.path.isfile(os.path.join(images_dir, f))]
    random.shuffle(images)

    val_size = int(len(images) * val_ratio)
    val_images = images[:val_size]
    train_images = images[val_size:]

    for image in train_images:
        shutil.move(os.path.join(images_dir, image), os.path.join(train_images_dir, image))
        label = image.replace('.jpg', '.txt')
        shutil.move(os.path.join(labels_dir, label), os.path.join(train_labels_dir, label))

    for image in val_images:
        shutil.move(os.path.join(images_dir, image), os.path.join(val_images_dir, image))
        label = image.replace('.jpg', '.txt')
        shutil.move(os.path.join(labels_dir, label), os.path.join(val_labels_dir, label))

if __name__ == "__main__":
    split_train_val(data_dir)