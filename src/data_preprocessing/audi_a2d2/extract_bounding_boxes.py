"""
This script is used to create bounding boxes from PNG images of the Audi dataset with semantic segmentation. 
It first creates a mask with the corresponding color encodings for the labels and uses dilate and erode to close gaps. 
Then, it creates the bounding boxes from the mask using the finde contours function and saves them in YOLO format.
"""
import cv2
import numpy as np
from collections import defaultdict
import os

color_to_class = {
    "#ff0000": "Car",
    "#c80000": "Car",
    "#960000": "Car",
    "#800000": "Car",
    "#b65906": "Bicycle",
    "#963204": "Bicycle",
    "#5a1e01": "Bicycle",
    "#5a1e1e": "Bicycle",
    "#cc99ff": "Pedestrian",
    "#bd499b": "Pedestrian",
    "#ef59bf": "Pedestrian",
    "#ff8000": "Truck",
    "#c88000": "Truck",
    "#968000": "Truck",
    "#0080ff": "Traffic signal",
    "#1e1c9e": "Traffic signal",
    "#3c1c64": "Traffic signal",
    "#00ffff": "Traffic sign",
    "#1edcdc": "Traffic sign",
    "#3c9dc7": "Traffic sign"
}

def save_bounding_boxes(image_file, bounding_boxes, output_dir):
    # Save bounding boxes in YOLO format
    output_file_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + '.txt')
    with open(output_file_path, 'w') as f:
        for box in bounding_boxes:
            class_id, x_center, y_center, width, height = box
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def iou(box1, box2):
    # Filter overlapping boxes
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def hex_to_bgr(hex_color):
    # Convert hex color to BGR
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# Directory containing labeled images
image_dir = 'data/audi_a2d2/labels_png'
output_dir = 'data/audi_a2d2/labels/labels_semantic'

class_name_to_id = {
    "Pedestrian": 0,
    "Car": 1,
    "Truck": 2,
    "Bicycle": 3,
    "Traffic signal": 4,
    "Traffic sign": 5
}

# Process each image in the directory
for image_file in os.listdir(image_dir):
    if image_file.endswith('.png'):
        image_path = os.path.join(image_dir, image_file)
        
        # Load the labeled image
        label_image = cv2.imread(image_path)
        # Extract image dimensions
        height, width, _ = label_image.shape
        if height != 1208 or width != 1920:
            print(f"Image {image_file} has incorrect dimensions: {width}x{height}")
        # Create a mask for each class
        bounding_boxes = []
        for hex_color, class_name in color_to_class.items():
            bgr_color = hex_to_bgr(hex_color)
            mask = cv2.inRange(label_image, bgr_color, bgr_color)
            kernel = np.ones((11, 11), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.erode(mask, kernel, iterations=1)

            # Extract bounding boxes for each mask separately
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if class_name not in ["Traffic sign", "Traffic light"] and w * h > 350 and w > 10 and h > 10:  # Filter out small boxes
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_n = w / width
                    h_n = h / height
                    class_id = class_name_to_id[class_name]
                    bounding_boxes.append((class_id, x_center, y_center, w_n, h_n))
                elif class_name in ["Traffic light"] and w * h > 200 and w > 5 and h > 5:
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_n = w / width
                    h_n = h / height
                    class_id = class_name_to_id[class_name]
                    bounding_boxes.append((class_id, x_center, y_center, w_n, h_n))
                elif class_name in ["Traffic sign"] and w * h > 100 and w > 5 and h > 5:
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_n = w / width
                    h_n = h / height
                    class_id = class_name_to_id[class_name]
                    bounding_boxes.append((class_id, x_center, y_center, w_n, h_n))
        if len(bounding_boxes) == 0:
            continue
        else:
            filtered_boxes = []
            for i, box1 in enumerate(bounding_boxes):
                keep = True
                for j, box2 in enumerate(bounding_boxes):
                    if i != j and iou(box1, box2) > 0.75:
                        keep = False
                        break
                if keep:
                    filtered_boxes.append(box1)
            save_bounding_boxes(image_file, filtered_boxes, output_dir)