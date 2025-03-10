"""
This script trains a pretrained YOLOv11n model on the Audi_A2D2 dataset using the Ultralytics YOLO library.
"""
from ultralytics import YOLO

# Load a model
model = YOLO("models/YoloV11n_BDD100k.pt")

# Train the model
train_results = model.train(
    data="config/train_audi.yaml",  # path to dataset YAML
    epochs=5000,  # number of training epochs
    imgsz=640,  # training image size
    batch=300,         # Batch-Größe
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    workers=14,        # CPU-Worker für Datenvorbereitung
    optimizer="SGD",  # Optimierungsalgorithmus
    patience=150,      # Frühes Beenden bei keiner Verbesserung
    project="results",  # Ergebnisse-Ordner
    name="Yolov11n_Audi_without_freeze",  # Experimentname
    save=True,         # Modelle regelmäßig speichern
    plots=True,
    save_period=100,   # Speicherintervall
    lr0=0.01,
    lrf=0.01,
)

