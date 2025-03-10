from ultralytics import YOLO

# Paths and settings
model_path = "models/YoloV11m_traffic_object_detection_final.pt"
video_path = "data/Loeningen_20241228.mp4"

# Perform YOLO inference
model = YOLO(model_path)
results = model.track(source=video_path, save=True, show=False, project="runs/track", name="Video_VoloV11m_20241228", save_txt=True, save_conf=True, persist=True, conf=0.35)
print("Finished.")
