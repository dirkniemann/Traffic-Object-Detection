"""
This script first moves the images and labels from the traffic_signs and audi_a2d2 datasets into the finetuning folder.
Then, it splits these data into 80% for training and 20% for validation.
"""
import os
import shutil
import random

def split_train_val(data_dir, val_ratio):
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
        if os.path.exists(os.path.join(labels_dir, label)):
            shutil.move(os.path.join(labels_dir, label), os.path.join(train_labels_dir, label))
        else:
            print(f'Label not found for image {image}', flush=True)
    for image in val_images:
        shutil.move(os.path.join(images_dir, image), os.path.join(val_images_dir, image))
        label = image.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(labels_dir, label)): 
            shutil.move(os.path.join(labels_dir, label), os.path.join(val_labels_dir, label))
        else:
            print(f'Label not found for image {image}', flush=True)

def move_files(source_dirs, target_dir_images, target_dir_labels):
    if not os.path.exists(target_dir_images):
        os.makedirs(target_dir_images)
    if not os.path.exists(target_dir_labels):
        os.makedirs(target_dir_labels)

    for source_dir in source_dirs:
        images_dir = os.path.join(source_dir, 'images')
        labels_dir = os.path.join(source_dir, 'labels')

        for image in os.listdir(images_dir):
            if image.endswith('.jpg'):
                shutil.move(os.path.join(images_dir, image), target_dir_images)
                label = image.replace('.jpg', '.txt')
                if os.path.exists(os.path.join(labels_dir, label)):
                    shutil.move(os.path.join(labels_dir, label), target_dir_labels)
                else:
                    print(f'Label not found for image {image}', flush=True)

if __name__ == "__main__":
    source_dirs = ['data/audi_a2d2',
                   'data/traffic_signs']
    target_dir_images = 'data/finetuning/images'
    target_dir_labels = 'data/finetuning/labels'
    target_data_dir = 'data/finetuning'
    move_files(source_dirs, target_dir_images, target_dir_labels)
    split_train_val(target_data_dir, val_ratio=0.2)
