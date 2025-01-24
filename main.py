import cv2
import torch
from parking_utils import detect_and_mark_red_points, draw_parking_boundary, cut_frame_and_resize, print_parking_data, \
    detect_new_car_on_entrance, detect_and_scan_license_plate, change_barrier_state, detect_new_car_on_exit
from car_detector import detect_car_and_return_frame
import pathlib
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
device = 'cuda' if torch.cuda.is_available() else 'cpu'
car_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='car_detector_model.pt',
                                     force_reload=True).to(device)
pathlib.PosixPath = temp

video_path = os.path.join("films", "parking.mp4")
cap = cv2.VideoCapture(video_path)

FRAME_INTERVAL = 5
frame_counter = 0

is_entry_barrier_down = True
is_exit_barrier_down = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % FRAME_INTERVAL == 0:
        frame = cut_frame_and_resize(frame)
        points = detect_and_mark_red_points(frame)
        frame, detections = detect_car_and_return_frame(frame, car_detection_model, confidence_threshold=0.6)
        frame_with_marks, parking_data = draw_parking_boundary(frame, points)

        is_detected = detect_new_car_on_entrance(detections, parking_data)
        if is_detected:
            is_entry_barrier_down = False
            detect_and_scan_license_plate()  # Skanuj tablicę rejestracyjną
        else:
            is_exit_barrier_down = True

        # Wykryj samochody opuszczające parking
        is_exiting = detect_new_car_on_exit(detections, parking_data)
        if is_exiting:
            is_exit_barrier_down = False
        else:
            is_exit_barrier_down = True

        # Zmień stan szlabanów
        change_barrier_state(is_entry_barrier_down, is_exit_barrier_down, frame)

        cv2.imshow("Parking", frame_with_marks)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
