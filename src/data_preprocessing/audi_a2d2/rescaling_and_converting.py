"""
This script processes all images by converting them from PNG to JPG and scaling them down from 1920x1208 to 640x640 using padding to maintain consistency with other data.
Additionally, it adjusts the bounding boxes to the new scale, taking the padding into account.
"""
import os
from PIL import Image, ImageOps

def load_bounding_boxes(file_path):
    # Load bounding boxes from a file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        boxes = [line.strip().split() for line in lines]
    return boxes

def pad_and_convert_image(image_path, target_size=(640, 640)):
    image = Image.open(image_path)
    rgb_image = image.convert('RGB')
    scaled_image = ImageOps.pad(rgb_image, target_size, color=(0, 0, 0))
    image_jpg_path = image_path.replace('.png', '.jpg')
    scaled_image.save(image_jpg_path, 'JPEG')
    os.remove(image_path)
    image.close()
    rgb_image.close()
    scaled_image.close()

def scale_bounding_box(old_box):
    idx, x, y, w, h = map(float, old_box)
    idx = int(idx)
    old_x = x * 1920
    old_y = y * 1208
    old_w = w * 1920
    old_h = h * 1208
    scale_x = 640 / 1920
    scale_y = 403 / 1208
    new_x = int(old_x * scale_x)
    new_y = int(old_y * scale_y) + 119
    new_w = int(old_w * scale_x)
    new_h = int(old_h * scale_y)
    scaled_bbox = (idx, new_x / 640, new_y / 640, new_w / 640, new_h / 640)
    return scaled_bbox

def process_images(image_folder, txt_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    total_images = len(image_files)
    
    for idx, filename in enumerate(image_files):
        print(f"Processing image {idx + 1}/{total_images}", flush=True)
        
        image_path = os.path.join(image_folder, filename)
        txt_path = os.path.join(txt_folder, filename.replace('.png', '.txt'))
        pad_and_convert_image(image_path)

        if os.path.exists(txt_path):
            new_box = []
            old_boxes = load_bounding_boxes(txt_path)
            for old_box in old_boxes:
                box = scale_bounding_box(old_box)
                new_box.append(box)
            os.remove(txt_path)
            with open(txt_path, 'w') as file:
                for box in new_box:
                    line = ' '.join(map(str, box)) + '\n'
                    file.write(line)
        else:
            print(f"Annotation file not found for {filename}", flush=True)
            continue
        

if __name__ == "__main__":
    image_folder = 'data/audi_a2d2/images'
    txt_folder = 'data/audi_a2d2/labels'
    process_images(image_folder, txt_folder)