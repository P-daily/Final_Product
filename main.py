import cv2
import torch
from parking_utils import detect_and_mark_red_points, draw_parking_boundary
from car_detector import detect_and_return_frame
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Check if CUDA is available and move the model to GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

car_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='car_detector_model.pt',
                                     force_reload=True).to(device)

pathlib.PosixPath = temp
# Wywołanie i pokazanie działania
video_path = "films/parking.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps * 20

scale_percent = 50
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Przycinanie góra-dół
    height, width, _ = frame.shape
    crop_top = int(height * 0.10)
    crop_bottom = int(height * 0.7)
    frame = frame[crop_top:crop_bottom, :]

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # czerwone punkty
    points = detect_and_mark_red_points(frame)
    frame = detect_and_return_frame(frame, car_detection_model, confidence_threshold=0.3)
    frame_with_marks, parking_data = draw_parking_boundary(frame, points)

    cv2.imshow("Parking Boundary", frame_with_marks)

    # Wypisuj dane co 20 sekund
    frame_counter += 1
    if frame_counter % frame_interval == 0:
        print("Rozpoznane miejsca parkingowe:")
        for data in parking_data:
            print(data)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()