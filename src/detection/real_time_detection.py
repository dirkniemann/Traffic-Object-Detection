import cv2
import time
from ultralytics import YOLO

# Paths and settings
model_path = "models/YoloV11m_traffic_object_detection_final.pt"
video_output_path = "runs/detect/realtime/Osnabrueck/YoloV11m_20250303_Stadt.mp4"
confidence_threshold = 0.5
camera_id = 1  # Camera index
camera_fps = 5  # Frame rate, should be between 5 and 30
inference_fps = 2   # Inference frame rate 
camera_width = 1280  # Camera resolution, can be 1280x960, 1280x720, 1024x768, 800x600, 640x480, 640x360, 320x240 for Logitech C270
camera_height = 960

camera = cv2.VideoCapture(camera_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
camera.set(cv2.CAP_PROP_FPS, camera_fps)

model = YOLO(model_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_output_path, fourcc, inference_fps, (camera_width, camera_height))

while True:
    start_time = time.time()

    # Read frame from camera
    ret, frame = camera.read()
    if not ret:
        print("Could not read camera feed.")
        break

    # Perform YOLO inference
    results = model.predict(frame, save=False, show=True, conf=confidence_threshold)

    # Plot results and save video
    for i, r in enumerate(results):
        # Plot results image
        result_image = r.plot()  # BGR-order numpy array
    video_writer.write(result_image)

    # Calculate frame interval
    elapsed_time = time.time() - start_time
    frame_interval = 1.0 / inference_fps - elapsed_time
    if frame_interval > 0:
        time.sleep(frame_interval)

    # Check for user input to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Program terminated.")
