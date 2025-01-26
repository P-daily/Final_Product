import time

import cv2
import torch

from parking_utils import detect_and_mark_red_points, draw_parking_boundary, cut_frame_and_resize, \
    detect_new_car_on_entrance, call_get_license_plate_api, call_is_entrance_allowed_api, call_car_entrance_api, \
    add_new_car, update_positions_of_cars, update_parking_area, check_if_last_registered_car_is_out_of_entrance, \
    reset_db, detect_car_on_exit, call_car_exit_api, call_car_parked_properly_api, call_log_api
from car_detector import draw_detections, detect_objects
import pathlib
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
device = 'cuda' if torch.cuda.is_available() else 'cpu'
car_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='jagiello_model.pt',
                                     force_reload=True).to(device)
pathlib.PosixPath = temp

video_path = os.path.join("films", "parking.mp4")
cap = cv2.VideoCapture(video_path)

FRAME_INTERVAL = 5
frame_counter = 0

is_entry_barrier_down = True
is_exit_barrier_down = True

reset_db()

exit_check_interval = 3
exit_time = time.time()
exitv2_time = time.time()
is_exit_detected = False
exit_detected_start_time = None
exit_car_license_plate = None
exiting_cars_license_plate = None

is_properly_parked_check_interval = 8
properly_parked_check_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % FRAME_INTERVAL == 0:
        frame = cut_frame_and_resize(frame)
        points = detect_and_mark_red_points(frame)
        detections = detect_objects(frame, car_detection_model, confidence_threshold=0.6)
        update_positions_of_cars(detections)
        frame_with_cars = draw_detections(frame, detections)
        frame_with_marks, parking_data = draw_parking_boundary(frame_with_cars, points)
        update_parking_area(parking_data)

        # Check for new car at the entrance
        is__entry_detected = detect_new_car_on_entrance(detections, parking_data)
        if is__entry_detected and is_entry_barrier_down:
            license_plate = call_get_license_plate_api()
            if license_plate:
                is_license_allowed, parking_type = call_is_entrance_allowed_api(license_plate)
                if is_license_allowed:
                    is_entry_barrier_down = False
                    if call_car_entrance_api(license_plate):
                        add_new_car(license_plate, detections)
                        entrance_data = [license_plate, parking_type]
                        call_log_api("car_entered", entrance_data)
                else:
                    # Not allowed carq
                    is_entry_barrier_down = True

        if check_if_last_registered_car_is_out_of_entrance():
            is_entry_barrier_down = True

        if time.time() - exit_time > exit_check_interval:
            exit_time = time.time()
            is_exit_detected, exit_car_license_plate = detect_car_on_exit(detections, parking_data)
            if is_exit_detected:
                is_exit_barrier_down = False
                exiting_cars_license_plate = exit_car_license_plate
            else:
                is_exit_barrier_down = True



        is_exit_V2_detected, _ = detect_car_on_exit(detections, parking_data, 'V2')
        print(is_exit_V2_detected, exiting_cars_license_plate)
        if is_exit_V2_detected and exiting_cars_license_plate is not None:
            is_car_deleted = call_car_exit_api(exiting_cars_license_plate)
            if is_car_deleted:
                call_log_api("car_exited", exiting_cars_license_plate)

        if time.time() - properly_parked_check_time > is_properly_parked_check_interval:
            properly_parked_check_time = time.time()
            are_all_cars_parked_properly, improperly_parked_cars = call_car_parked_properly_api()
            if not are_all_cars_parked_properly and improperly_parked_cars:
                call_log_api("not_allowed_parking (wrong privileges)", improperly_parked_cars)

        cv2.imshow("Parking", frame_with_marks)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
