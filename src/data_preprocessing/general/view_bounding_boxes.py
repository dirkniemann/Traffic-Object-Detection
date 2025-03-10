"""
This script visualizes bounding boxes on images based on label files.
"""
import os
import random
from PIL import Image, ImageDraw

# Define paths
images_dir = 'test/images'
labels_dir = 'test/labels'
output_dir = 'test'

# Define label mapping
label_mapping = {
    0: "pedestrian",
    1: "car",
    2: "truck",
    3: "bicycle",
    4: "traffic light",
    5: "speed limit 20 (prohibitory)",
    6: "speed limit 30 (prohibitory)",
    7: "speed limit 50 (prohibitory)",
    8: "speed limit 60 (prohibitory)",
    9: "speed limit 70 (prohibitory)",
    10: "speed limit 80 (prohibitory)",
    11: "restriction ends 80 (other)",
    12: "speed limit 100 (prohibitory)",
    13: "speed limit 120 (prohibitory)",
    14: "no overtaking (prohibitory)",
    15: "no overtaking (trucks) (prohibitory)",
    16: "priority at next intersection (danger)",
    17: "priority road (other)",
    18: "give way (other)",
    19: "stop (other)",
    20: "no traffic both ways (prohibitory)",
    21: "no trucks (prohibitory)",
    22: "no entry (other)",
    23: "danger (danger)",
    24: "bend left (danger)",
    25: "bend right (danger)",
    26: "bend (danger)",
    27: "uneven road (danger)",
    28: "slippery road (danger)",
    29: "road narrows (danger)",
    30: "construction (danger)",
    31: "traffic signal (danger)",
    32: "pedestrian crossing (danger)",
    33: "school crossing (danger)",
    34: "cycles crossing (danger)",
    35: "snow (danger)",
    36: "animals (danger)",
    37: "restriction ends (other)",
    38: "go right (mandatory)",
    39: "go left (mandatory)",
    40: "go straight (mandatory)",
    41: "go right or straight (mandatory)",
    42: "go left or straight (mandatory)",
    43: "keep right (mandatory)",
    44: "keep left (mandatory)",
    45: "roundabout (mandatory)",
    46: "restriction ends (overtaking) (other)",
    47: "restriction ends (overtaking (trucks)) (other)"
}

# Collect all .txt files and their corresponding .jpg files
all_files = []

for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        image_path = os.path.join(images_dir, filename.replace('.txt', '.jpg'))
        label_path = os.path.join(labels_dir, filename)
        all_files.append((image_path, label_path))

# Select 10 random files
random_files = random.sample(all_files, 1)

def draw_bounding_boxes(image_path, label_path):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    label, x_center, y_center, width, height = map(float, parts)
                    x_center *= img.width
                    y_center *= img.height
                    width *= img.width
                    height *= img.height
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    
                    # Set color based on label
                    if int(label) == 0:
                        color = "darkred"
                    elif int(label) == 1:
                        color = "darkblue"
                    elif int(label) == 2:
                        color = "yellow"
                    elif int(label) == 3:
                        color = "purple"
                    elif int(label) == 4:
                        color = "darkorange"
                    else:
                        color = "darkgreen"
                    
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                    draw.text((x_min, y_min-35), label_mapping[int(label)], fill=color, font_size=30, stroke_width=0.5)
        return img

# Save the images with bounding boxes
for image_path, label_path in random_files:
    img_with_boxes = draw_bounding_boxes(image_path, label_path)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    img_with_boxes.save(output_path)