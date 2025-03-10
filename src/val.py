from ultralytics import YOLO

# Load a model
model = YOLO("models/YoloV11n_Audi_A2D2.pt")  # load a custom model

# Validate the model
metrics = model.val(plots=True, save_json=True, name='YoloV11n_val')  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category