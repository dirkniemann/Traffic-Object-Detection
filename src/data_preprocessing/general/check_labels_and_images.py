"""
This script checks for mismatched image and label files in the Cityscapes dataset directories and moves any image files that do not have a corresponding label file to a separate directory to check it manually.
It also deletes any label files that do not have a corresponding image file.
"""
import os
import shutil

# Define the directories for images, labels, and unmatched images
images_dir = 'data/cityspaces/images'
labels_dir = 'data/cityspaces/labels/detection_label'
unmatched_images_dir = 'data/cityspaces/images/unmatched'

# Create the unmatched images directory if it doesn't exist
os.makedirs(unmatched_images_dir, exist_ok=True)

# Get list of image files and corresponding txt files
image_files = set(os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.jpg'))
label_files = set(os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt'))

# Find image files without corresponding txt files
images_without_labels = image_files - label_files

# Find txt files without corresponding image files
labels_without_images = label_files - image_files

# Move image files without corresponding txt files
for image_file in images_without_labels:
    image_path = os.path.join(images_dir, image_file + '.jpg')
    new_image_path = os.path.join(unmatched_images_dir, image_file + '.jpg')
    shutil.move(image_path, new_image_path)
    print(f"Moved image file without corresponding text file: {image_path} to {new_image_path}")

# Delete JStxtON files without corresponding image files
for label_file in labels_without_images:
    label_path = os.path.join(labels_dir, label_file + '.txt')
    os.remove(label_path)
    print(f"Deleted text file without corresponding image: {label_path}")

print("Cleanup complete. Each image now has a corresponding text file and vice versa.")