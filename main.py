import cv2
import torch
from parking_utils import detect_and_mark_red_points, draw_parking_boundary, cut_frame_and_resize, \
    detect_new_car_on_entrance, call_get_license_plate_api, call_is_entrance_allowed_api, call_car_entrance_api, \
    add_new_car, update_positions_of_cars, update_parking_area, check_if_last_registered_car_is_out_of_entrance
from car_detector import draw_detections, detect_objects
import pathlib
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
device = 'cuda' if torch.cuda.is_available() else 'cpu'
car_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='first_detection_model.pt',
                                     force_reload=True).to(device)
pathlib.PosixPath = temp

video_path = os.path.join("films", "parking.mp4")
cap = cv2.VideoCapture(video_path)

FRAME_INTERVAL = 5
frame_counter = 0

is_barrier_down = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % FRAME_INTERVAL == 0:
        frame = cut_frame_and_resize(frame)
        points = detect_and_mark_red_points(frame)
        detections = detect_objects(frame, car_detection_model, confidence_threshold=0.6)
        frame_with_marks, parking_data = draw_parking_boundary(frame, points)
        update_positions_of_cars(detections)
        update_parking_area(parking_data)
        is_detected = detect_new_car_on_entrance(detections, parking_data)
        if is_detected and is_barrier_down:
            license_plate = call_get_license_plate_api()
            if license_plate:
                is_license_allowed, parking_type = call_is_entrance_allowed_api(license_plate)
                print(f"License plate: {license_plate}, allowed: {is_license_allowed}, parking type: {parking_type}")
                if is_license_allowed:
                    is_barrier_down = False
                    call_car_entrance_api(license_plate)
                    add_new_car(license_plate, detections)

                else:
                    is_barrier_down = True

        # popraw te funkcje pierwszy samochód jest git, ale drugi już fisiuje
        is_barrier_down = check_if_last_registered_car_is_out_of_entrance()


        draw_detections(frame_with_marks, detections)
        cv2.imshow("Parking", frame_with_marks)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
