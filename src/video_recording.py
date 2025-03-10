import cv2

# Paths and settings
video_output_path = "data/Osnabrueck_20250303_Stadt.mp4"
camera_id = 1  # Camera index
camera_fps = 30  # Frame rate, should be between 5 and 30
camera_width = 1280  # Camera resolution, can be 1280x960, 1280x720, 1024x768, 800x600, 640x480, 640x360, 320x240 for Logitech C270
camera_height = 960

camera = cv2.VideoCapture(camera_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
camera.set(cv2.CAP_PROP_FPS, camera_fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_output_path, fourcc, camera_fps, (camera_width, camera_height))

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    video_writer.write(frame)
    
    cv2.imshow('Recording', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
video_writer.release()
cv2.destroyAllWindows()