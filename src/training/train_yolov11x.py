"""
This script trains a YOLOv11x model on a traffic signs dataset using the Ultralytics YOLO library.
"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")

# Train the model
train_results = model.train(
    data="config/train_traffic_signs.yaml",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    imgsz=640,  # training image size
    batch=80,         # Batch-Größe
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    workers=14,        # CPU-Worker für Datenvorbereitung
    optimizer="AdamW",  # Optimierungsalgorithmus
    patience=150,      # Frühes Beenden bei keiner Verbesserung
    project="results",  # Ergebnisse-Ordner
    name="YoloV11x_GTSDB_AdamW",  # Experimentname
    save=True,         # Modelle regelmäßig speichern
    plots=True,
    deterministic=False,
    save_period=100,   # Speicherintervall
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
)