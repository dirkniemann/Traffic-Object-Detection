"""
This script uses the YOLOv11n model, trained on BDD100k, to detect objects in the GTSDB dataset.
It saves the bounding boxes in YOLO format as TXT files.
"""
import os
import cv2
from collections import defaultdict
from ultralytics import YOLO

# Pfade definieren
input_folder = 'data/traffic_signs/images/val'
output_folder = 'data/traffic_signs/labels/object_detection'
model_path = 'models/YoloV11n_BDD100k.pt'


# YOLOv11 Modell laden
model = YOLO(model_path)

# Eingabeordner durchlaufen und Dateien zählen
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg'))]
print(f'Gefundene Dateien: {len(image_files)}')

# Dictionary zur Zählung der Labels
label_count = defaultdict(int)

# Funktion zum Speichern der Bounding Boxes im YOLO-Format
def save_yolo_format(file_path, boxes):
    with open(file_path, 'w') as f:
        for box in boxes:
            label, x_center, y_center, width, height = box
            f.write(f"{label} {x_center} {y_center} {width} {height}\n")

# Bilder verarbeiten
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)
    
    # Modellvorhersage
    results = model.predict(source=img, save=False, show=False, conf=0.5, classes=[0, 1, 2, 3, 4])
    
    # Bounding Boxes extrahieren
    extracted_boxes = []
    for results in results:
        boxes = results.boxes
        for box in boxes:
            x_center, y_center, width, height = box.xywhn[0] 
            label = int(box.cls)
            extracted_boxes.append((label, x_center, y_center, width, height))
            label_count[label] += 1
    
    # Wenn Bounding Boxes gefunden wurden, speichern
    if extracted_boxes:
        output_file_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.txt')
        save_yolo_format(output_file_path, extracted_boxes)

# Ausgabe der Label-Zählung sortiert nach Label
print("Gefundene Bounding Boxes pro Label:")
for label in sorted(label_count.keys()):
    print(f"Label {label}: {label_count[label]}")
