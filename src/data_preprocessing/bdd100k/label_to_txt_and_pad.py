"""
This script processes image and label files in the BDD100K dataset and converts the labels to a text format. Additionally, it pads the images to a square shape and resizes them to 640x640 pixels. 
Note: This script may take a long time to run, depending on the number of images and labels in the dataset. For me it took over 2 hours for 100000 images. 
      If you have access to multiple CPU cores you can parallelize the processing to speed it up.
"""
import os
import json
from PIL import Image, ImageOps, UnidentifiedImageError

# Define the directories for images and labels
images_dir = 'data/bdd100k/images'
json_dir = 'data/bdd100k/labels'

# Define the label mapping
label_mapping = {
    "pedestrian": 0,
    "car": 1,
    "truck": 2,
    "bicycle": 3,
    "traffic light": 4,
    "traffic sign": 5
}

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

# Iterate over all files in the directory
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(json_dir, filename)
        with open(file_path, 'r') as f:
            print(f"Processing {file_path}")
            data = json.load(f)
            for item in data:
                txt_content = []
                obj = item.get('name')
                labels = item.get('labels', {})
                for idx in labels:
                    category = idx.get('category', {})
                    if category not in label_mapping:
                        continue
                    index = label_mapping[category]
                    bbox = idx.get('box2d', {})
                    xmin = bbox.get('x1')
                    ymin = bbox.get('y1')
                    xmax = bbox.get('x2')
                    ymax = bbox.get('y2')
                    if xmin is not None and ymin is not None and xmax is not None and ymax is not None:
                        # Scale the bounding box coordinates
                        scale_x = 640 / 1280
                        scale_y = 360 / 720
                        new_x_min = int(xmin * scale_x)
                        new_y_min = int(ymin * scale_y)
                        new_x_max = int(xmax * scale_x)
                        new_y_max = int(ymax * scale_y)

                        # Calculate relative coordinates
                        width = new_x_max - new_x_min
                        height = new_y_max - new_y_min
                        x_center = new_x_min + width / 2
                        y_center = new_y_min + height / 2 + 140 # Add 140 to y_center to center the object in the image because of padding

                        new_bbox = (x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2)
                        ignore_bbox = False
                        for line in txt_content:
                            _, x_c, y_c, w, h = map(float, line.split())
                            old_x_min = int((x_c - w / 2) * 640)
                            old_y_min = int((y_c - h / 2) * 640)
                            old_x_max = int((x_c + w / 2) * 640)
                            old_y_max = int((y_c + h / 2) * 640)
                            old_bbox = (old_x_min, old_y_min, old_x_max, old_y_max)
                            if calculate_iou(new_bbox, old_bbox) > 0.8:
                                ignore_bbox = True
                                break

                        if not ignore_bbox:
                            txt_content.append(f"{index} {x_center / 640} {y_center / 640} {width / 640} {height / 640}")
                # Write the label to a .txt file
                if txt_content:
                    txt_filename = obj.replace('.jpg', '.txt')
                    txt_path = os.path.join(json_dir, txt_filename)
                    with open(txt_path, 'w') as txt_file:
                        txt_file.write('\n'.join(txt_content))
                    # Find and scale the corresponding image
                    image_filename = obj
                    image_path = os.path.join(images_dir, image_filename)
                    if os.path.exists(image_path):
                        try:
                            with Image.open(image_path) as img:
                                img = ImageOps.pad(img, (640, 640), color=(0, 0, 0))
                                img.save(image_path)  # Overwrite the original image
                        except UnidentifiedImageError:
                            print(f"Image {image_filename} is corrupted and will be deleted.")
                            os.remove(image_path)
                            if os.path.exists(txt_path):
                                os.remove(txt_path)
                    else:
                        print(f"Image {image_filename} not found.")
                        if os.path.exists(txt_path):
                            os.remove(txt_path)

        # Delete the JSON file after processing
        os.remove(file_path)
        print(f"Processed {file_path}")

print("Conversion complete.")