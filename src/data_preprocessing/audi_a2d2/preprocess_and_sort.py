"""
This script prepares the data from the Audi_A2D2 dataset for further processing.
It extracts all relevant images from the dataset and deletes unnecessary data.
"""
import os
import shutil
from PIL import Image

root_directory = 'data/audi_a2d2'

def move_files_and_cleanup(root_dir):
    for dir_name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, dir_name)) and dir_name not in ['images', 'labels']:
            print(f"Processing {dir_name}", flush=True)
            lidar_path = os.path.join(root_dir, dir_name, 'lidar')
            if os.path.exists(lidar_path):
                shutil.rmtree(lidar_path)
            images_path = os.path.join(root_dir, dir_name, 'camera', 'cam_front_center')
            if os.path.exists(images_path):
                print(f"Processing images", flush=True)
                for file_name in os.listdir(images_path):
                    if file_name.endswith('.png'):
                        src_file = os.path.join(images_path, file_name)
                        dest_dir = os.path.join(root_dir, 'images')
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_file = os.path.join(dest_dir, file_name)
                        shutil.move(src_file, dest_file)
            labels_path = os.path.join(root_dir, dir_name, 'label', 'cam_front_center')
            if os.path.exists(labels_path):
                print(f"Processing labels", flush=True)
                for file_name in os.listdir(labels_path):
                    if file_name.endswith('.png'):
                        src_file = os.path.join(labels_path, file_name)
                        dest_dir = os.path.join(root_dir, 'labels')
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_file = os.path.join(dest_dir, file_name)
                        shutil.move(src_file, dest_file)
            shutil.rmtree(os.path.join(root_dir, dir_name))
            print(f"Deleted {dir_name}", flush=True)
    print("Done", flush=True)

move_files_and_cleanup(root_directory)
