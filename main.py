import time

import cv2
import torch
from numexpr.expressions import truediv_op

from parking_utils import detect_and_mark_red_points, draw_parking_boundary, cut_frame_and_resize, \
    detect_new_car_on_entrance, call_get_license_plate_api, call_is_entrance_allowed_api, call_car_entrance_api, \
    add_new_car, update_positions_of_cars, update_parking_area, check_if_last_registered_car_is_out_of_entrance, \
    reset_db, detect_car_on_exit, call_car_exit_api
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

exit_check_interval = 4
exit_time = time.time()
exit_detected_start_time = None

exit_car_license_plate = None

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
                print(f"ENTRANCE License plate: {license_plate}")
                if is_license_allowed:
                    is_entry_barrier_down = False
                    call_car_entrance_api(license_plate)
                    add_new_car(license_plate, detections)

                else:
                    # Not allowed carq
                    is_entry_barrier_down = True


        if check_if_last_registered_car_is_out_of_entrance():
            is_entry_barrier_down = True


        is_exit_detected, exit_car_license_plate = detect_car_on_exit(detections, parking_data)
        if is_exit_detected:
            is_exit_barrier_down = False
        else:
            is_exit_barrier_down = True
        is_exit_V2_detected, _ = detect_car_on_exit(detections, parking_data, 'V2')
        if is_exit_V2_detected:
            is_car_deleted = call_car_exit_api(exit_car_license_plate)
            if is_car_deleted:
                print("\n\n\n\n-------------------------\n\n\n\n\n\n\n\n\Car deleted successfully!\n\n\n\n\n\n\n\n\n------------------------------------")
            else:
                print("Car not deleted")



        cv2.imshow("Parking", frame_with_marks)


    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
