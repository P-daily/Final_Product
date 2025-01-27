import time

import cv2
import torch

from parking_utils import detect_and_mark_red_points, draw_parking_boundary, cut_frame_and_resize, \
    detect_new_car_on_entrance, call_get_license_plate_api, call_is_entrance_allowed_api, call_car_entrance_api, \
    add_new_car, update_positions_of_cars, update_parking_area, check_if_last_registered_car_is_out_of_entrance, \
    reset_db, detect_car_on_exit, call_car_exit_api, call_car_parked_properly_api, call_log_api, change_barrier_state, \
    get_cars_from_api
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

exit_check_interval = 1
exit_time = time.time()
exitv2_time = time.time()
is_exit_detected = False
exit_detected_start_time = None
exit_car_license_plate = None
exiting_cars_license_plate = None


frames_with_cars_in_the_same_position = 0
BLOCK_TIME_THRESHOLD = 15  # Czas w sekundach
POSITION_TOLERANCE = 5  # Tolerancja pozycji w pikselach
cars_blocking_road = {}

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

        cars = get_cars_from_api()
        road = next((item for item in parking_data if item["id"] == 9), None)
        # print(road)
        for car in cars:
            car_id = car['id']  # Identyfikator samochodu
            car_position = [car['center_x'], car['center_y']]  # Pozycja samochodu (np. [x, y])
            # Sprawdź, czy samochód znajduje się na drodze
            if not (road["top_left_x"] < car_position[0] and
                    road["top_left_y"] < car_position[1] and
                    road["bottom_right_x"] > car_position[0] and
                    road["bottom_right_y"] > car_position[1]):
                # Usuń samochód spoza drogi, jeśli jest w monitorowaniu
                if car_id in cars_blocking_road:
                    print(f"Samochód {car_id} opuścił drogę.\n{cars_blocking_road}")
                    cars_blocking_road.pop(car_id, None)
            else:
                # Sprawdź, czy samochód już jest monitorowany
                if car_id in cars_blocking_road:
                    print(f"Samochód {car_id} jest na drodze.\n{cars_blocking_road}")
                    previous_position = cars_blocking_road[car_id]['position']
                    blocking_start_time = cars_blocking_road[car_id]['start_time']
                    # Sprawdź, czy samochód jest w tej samej pozycji (z tolerancją)
                    if abs(car_position[0] - previous_position[0]) < POSITION_TOLERANCE and \
                            abs(car_position[1] - previous_position[1]) < POSITION_TOLERANCE:
                        # Oblicz czas blokowania
                        elapsed_time = time.time() - blocking_start_time
                        if elapsed_time >= BLOCK_TIME_THRESHOLD:
                            data_to_send = "License_plate:" + car['license_plate']
                            call_log_api("car_blocking_road", data_to_send)
                            cars_blocking_road[car_id] = {'position': car_position, 'start_time': time.time()}
                    else:
                        # Samochód zmienił pozycję, zresetuj czas blokowania
                        cars_blocking_road[car_id] = {'position': car_position, 'start_time': time.time()}
                else:
                    # Dodaj nowy samochód do monitorowania
                    cars_blocking_road[car_id] = {'position': car_position, 'start_time': time.time()}
        # Usuń samochody, które zniknęły (porównanie aktualnych ID samochodów)
        current_car_ids = [car['id'] for car in cars]
        cars_blocking_road = {car_id: data for car_id, data in cars_blocking_road.items() if
                              car_id in current_car_ids}

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


        is_exit_V2_detected, _ = detect_car_on_exit(detections, parking_data, 'V2')
        if is_exit_V2_detected and exiting_cars_license_plate is not None:
            is_car_deleted = call_car_exit_api(exiting_cars_license_plate)
            if is_car_deleted:
                call_log_api("car_exited", exiting_cars_license_plate)
                is_exit_barrier_down = True

        if time.time() - properly_parked_check_time > is_properly_parked_check_interval:
            properly_parked_check_time = time.time()
            are_all_cars_parked_properly, improperly_parked_cars = call_car_parked_properly_api()
            if not are_all_cars_parked_properly and improperly_parked_cars:
                call_log_api("not_allowed_parking (wrong privileges)", improperly_parked_cars)

        change_barrier_state(is_entry_barrier_down, is_exit_barrier_down, frame_with_marks)

        cv2.imshow("Parking", frame_with_marks)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
