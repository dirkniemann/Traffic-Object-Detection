"""
This script trains a YOLOv11n model on the BDD100K dataset using the Ultralytics YOLO library.
"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="config/train_bdd100k.yaml",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    imgsz=640,  # training image size
    batch=300,         # Batch-Größe
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    workers=14,        # CPU-Worker für Datenvorbereitung
    optimizer="SGD",  # Optimierungsalgorithmus
    patience=150,      # Frühes Beenden bei keiner Verbesserung
    project="results",  # Ergebnisse-Ordner
    name="Yolov11n_BDD100k",  # Experimentname
    save=True,         # Modelle regelmäßig speichern
    plots=True,
    save_period=100,   # Speicherintervall
    lr0=0.01,
    lrf=0.01,
)

