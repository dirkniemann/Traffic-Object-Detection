"""
This script converts PPM images to JPG and scales them down from 1360x800 to 640x640 using padding.
Additionally, it saves a TXT file with the bounding boxes for each image and scales these boxes to the corresponding padded image size.
"""
import os
import shutil
import random
from PIL import Image, ImageOps

# Define paths
data_dir = 'data/traffic_signs'
gt_file = 'data/traffic_signs/gt.txt'
readme_file = 'data/traffic_signs/ReadMe.txt'
images_dir = os.path.join(data_dir, 'images')
labels_dir = os.path.join(data_dir, 'labels')

# Convert .ppm images to .jpg and resize to 640x640 with padding
for item in os.listdir(data_dir):
    if item.endswith('.ppm'):
        ppm_path = os.path.join(data_dir, item)
        jpg_path = os.path.join(data_dir, item.replace('.ppm', '.jpg'))
        with Image.open(ppm_path) as img:
            img = ImageOps.pad(img, (640, 640), color=(0, 0, 0))
            img.save(jpg_path, 'JPEG')
        os.remove(ppm_path)

# Delete all subfolders under data/traffic_signs
for item in os.listdir(data_dir):
    item_path = os.path.join(data_dir, item)
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)

# Create the folder structure
os.makedirs(os.path.join(images_dir, 'train'))
os.makedirs(os.path.join(images_dir, 'val'))
os.makedirs(os.path.join(labels_dir, 'train'))
os.makedirs(os.path.join(labels_dir, 'val'))

# Read the bounding box information from gt.txt
annotations = {}
with open(gt_file, 'r') as f:
    for line in f:
        parts = line.strip().split(';')
        if len(parts) == 6:
            filename, x_min, y_min, x_max, y_max, label = parts
            filename = filename.replace('.ppm', '.jpg')
            if filename not in annotations:
                annotations[filename] = []
            annotations[filename].append((int(x_min), int(y_min), int(x_max), int(y_max), int(label)))

# Get all .jpg files in the data directory
all_jpg_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

# Split the images into 90% for training and 10% for validation
random.shuffle(all_jpg_files)
split_index = int(0.9 * len(all_jpg_files))
train_files = all_jpg_files[:split_index]
val_files = all_jpg_files[split_index:]

def save_annotations(files, subset):
    for filename in files:
        img_src = os.path.join(data_dir, filename)
        img_dst = os.path.join(images_dir, subset, filename)
        label_dst = os.path.join(labels_dir, subset, filename.replace('.jpg', '.txt'))

        # Move the image to the corresponding folder
        shutil.move(img_src, img_dst)

        # Create the YOLO-format .txt file
        with open(label_dst, 'w') as f:
            if filename in annotations:
                for x_min, y_min, x_max, y_max, label in annotations[filename]:
                    # Calculate new bounding box coordinates for 640x640 image
                    scale_x = 640 / 1360
                    scale_y = 376 / 800 # in this case because of the padding, the new height of the picture in the 640x640 picture is 376
                    new_x_min = int(x_min * scale_x)
                    new_y_min = int(y_min * scale_y)
                    new_x_max = int(x_max * scale_x)
                    new_y_max = int(y_max * scale_y)

                    width = new_x_max - new_x_min
                    height = new_y_max - new_y_min
                    x_center = new_x_min + width / 2
                    y_center = new_y_min + height / 2 + 132 # 132 is the padding added to the top of the image
                    f.write(f"{label} {x_center / 640} {y_center / 640} {width / 640} {height / 640}\n")

# Save the annotations for training and validation
save_annotations(train_files, 'train')
save_annotations(val_files, 'val')