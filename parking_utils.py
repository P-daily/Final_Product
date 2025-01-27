import time

import numpy as np
import cv2
import requests

API_URL = "http://127.0.0.1:5000"

last_detection_time = time.time()


def call_all_license_plates_positions_api():
    try:
        response = requests.get(f"{API_URL}/all_license_plates_positions")
        if response.status_code == 200:
            return response.json()['cars']
        else:
            return None

    except Exception as e:
        print(f"Error during API call: {e}")
        return None


def reset_db():
    try:
        response = requests.get(f"{API_URL}/reset_db")
        if response.status_code == 200:
            print("Database reset successfully!")
        else:
            print(f"Failed to reset database: {response.text}")

    except Exception as e:
        print(f"Error during API call: {e}")


# Wykrywanie granic parkingu
def detect_and_mark_red_points(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Zakres czerwonego
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Maska czerwonego
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0

        if 100 <= area <= 800 and 0.3 <= circularity <= 1.0:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # środek obszaru
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))

    return points


# Funkcja pomocnicza do rozpoznawania miejsc uprzywilejowanych
def analyze_color_in_area(frame, top_left, bottom_right):
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # niebieski
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_count = cv2.countNonZero(blue_mask)

    # żółty
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_count = cv2.countNonZero(yellow_mask)

    return blue_count, yellow_count


# Rysowanie układu i generewanie data-packu struktury parkingu
def draw_parking_boundary(frame, points):
    if not points:
        return frame, []

    points = np.array(points)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    # Wymiary parkingu
    height = y_max - y_min
    width = x_max - x_min

    # Wymiary kolumn
    entrance_width = int(0.29 * width)
    parking_width = (width - entrance_width) // 4

    # Podział na wiersze
    row_heights = [0.33 * height, 0.17 * height, 0.17 * height, 0.33 * height]
    row_starts = [y_min + int(sum(row_heights[:i])) for i in range(4)]
    row_starts.append(y_max)

    # Punkty początkowe kolumn
    col_starts = [x_min + i * parking_width for i in range(4)]
    col_starts.append(x_max - entrance_width)
    col_starts.append(x_max)

    parking_areas = []
    for row_idx in [0, 3]:
        for col_idx in range(4):
            top_left = (col_starts[col_idx], row_starts[row_idx])
            bottom_right = (col_starts[col_idx + 1], row_starts[row_idx + 1])
            blue_count, yellow_count = analyze_color_in_area(frame, top_left, bottom_right)
            parking_areas.append((blue_count, yellow_count, top_left, bottom_right))

    parking_data = []
    label = 1

    # Wyjazd za szlabanem
    exit_top_left = (col_starts[4], row_starts[2])
    exit_bottom_right = (col_starts[5], row_starts[3])
    cv2.rectangle(frame, exit_top_left, exit_bottom_right, (0, 0, 255), 1)
    cv2.putText(frame, "Wyjazd_V2",
                (exit_top_left[0] + 10, exit_top_left[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    parking_data.append({
        "id": 12,  # id wyjazdu
        "type": "EXITV2",  # Wyjazd
        "top_left": exit_top_left,
        "bottom_right": exit_bottom_right
    })

    # Wyjazd
    exit_top_left = (col_starts[2], row_starts[2])
    exit_bottom_right = (col_starts[4], row_starts[3])
    cv2.rectangle(frame, exit_top_left, exit_bottom_right, (0, 0, 255), 1)
    cv2.putText(frame, "Wyjazd",
                (exit_top_left[0] + 10, exit_top_left[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

    parking_data.append({
        "id": 11,  # id wyjazdu
        "type": "EXIT",  # Wyjazd
        "top_left": exit_top_left,
        "bottom_right": exit_bottom_right
    })

    for idx, (blue_count, yellow_count, top_left, bottom_right) in enumerate(parking_areas):
        color = (0, 255, 255)  # Żółty
        area_type = "DEFAULT"

        if label == 1:
            area_type = "MANAGER"  # Kierownictwo
            color = (0, 0, 0)  # Czarny
        elif label == 5:
            area_type = "DISABLED"  # Inwalidzi
            color = (255, 255, 255)  # Biały

        parking_data.append({
            "id": label,
            "type": area_type,
            "top_left": top_left,
            "bottom_right": bottom_right
        })

        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 1)
        cv2.putText(frame, f"{label} {area_type}",
                    (top_left[0] + 10, top_left[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        label += 1

    # Droga
    road_top_left = (x_min, row_starts[1])
    road_bottom_right = (x_max - entrance_width, row_starts[3])
    cv2.putText(frame, "Droga",
                (x_min + 10, row_starts[1] + int((row_heights[1] + row_heights[2]) // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 1)

    parking_data.append({
        "id": 9,  # id drogi
        "type": "ROAD",  # Droga
        "top_left": road_top_left,
        "bottom_right": road_bottom_right
    })

    # Wjazd
    entrance_top_left = (col_starts[-2], y_min)
    entrance_bottom_right = (col_starts[-1], y_min + (y_max - y_min) // 2)
    cv2.rectangle(frame, entrance_top_left, entrance_bottom_right, (0, 0, 255), 1)
    cv2.putText(frame, "Wjazd",
                (entrance_top_left[0] + 10, entrance_top_left[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

    parking_data.append({
        "id": 10,  # id wjazdu
        "type": "ENTRANCE",
        "top_left": entrance_top_left,
        "bottom_right": entrance_bottom_right
    })

    return frame, parking_data


def cut_frame_and_resize(frame, scale_percent=50):
    height, width, _ = frame.shape
    crop_top = int(height * 0.10)
    crop_bottom = int(height * 0.7)
    frame = frame[crop_top:crop_bottom, :]

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    return frame


def detect_new_car_on_entrance(detections, parking_data):
    global last_detection_time
    current_detection_time = time.time()
    if current_detection_time - last_detection_time < 2:
        return False
    last_detection_time = current_detection_time

    entrance_coords = parking_data[-1]
    entrance_top_left_x = entrance_coords['top_left_x']
    entrance_bottom_right_y = entrance_coords['bottom_right_y']

    for _, detection in detections.iterrows():
        if detection['xmin'] > entrance_top_left_x and detection['ymax'] < entrance_bottom_right_y:
            return True

    return False


def detect_car_on_exit(detections, parking_data, exit='V1'):
    license_plate = None
    if exit == 'V2':
        exit_coords = parking_data[0]
    else:
        exit_coords = parking_data[1]
        license_plate = call_get_license_plate_from_exit_api()

    exit_top_left_x = exit_coords['top_left_x']
    exit_top_left_y = exit_coords['top_left_y']

    exit_bottom_right_x = exit_coords['bottom_right_x']
    exit_bottom_right_y = exit_coords['bottom_right_y']

    for _, detection in detections.iterrows():
        if detection['xmin'] > exit_top_left_x and detection['ymin'] > exit_top_left_y and detection[
            'xmax'] < exit_bottom_right_x and detection['ymax'] < exit_bottom_right_y:
            return True, license_plate

    return False, None


def call_get_license_plate_api():
    try:
        response = requests.get(f"{API_URL}/license_plate_from_entrance")
        if response.status_code == 200:
            return response.json()['license_plate']
        else:
            return None

    except Exception as e:
        print(f"Error during API call: {e}")


def call_is_entrance_allowed_api(license_plate):
    try:
        print(f"Checking if entrance is allowed for license plate: {license_plate}")
        response = requests.get(f"{API_URL}/is_entrance_allowed/{license_plate}")
        if response.status_code == 200:
            # print(response.json())
            is_allowed = response.json()['is_allowed']
            parking_type = response.json()['parking_type']
            return is_allowed, parking_type
        else:
            print(f"Failed to check if entrance is allowed for license plate '{license_plate}': {response.text}")
            return False, 'unknown'

    except Exception as e:
        print(f"Error during API call: {e}")
        return False, 'unknown'


def call_car_entrance_api(license_plate):
    try:
        response = requests.post(f"{API_URL}/car_entrance", json={'license_plate': license_plate})
        if response.status_code == 201:
            return True
        else:
            print(f"Failed to send license plate '{license_plate}': {response.text}")
            return False

    except Exception as e:
        print(f"Error during API call: {e}")
        return False


def call_car_position_update_api(json_data):
    try:
        response = requests.post(f"{API_URL}/car_position", json=json_data)
        if response.status_code == 201:
            # print(f"Car position sent successfully!")
            return True
        else:
            print(f"Failed to send car position: {response.text}")
            return False

    except Exception as e:
        print(f"Error during API call: {e}")
        return False


def add_new_car(license_plate, detections):
    car_detection = detections.sort_values(by='xmin', ascending=False).iloc[0]

    # print(car_detection)

    left_top_x = int(car_detection['xmin'])
    left_top_y = int(car_detection['ymin'])
    right_bottom_x = int(car_detection['xmax'])
    right_bottom_y = int(car_detection['ymax'])

    center_x = (left_top_x + right_bottom_x) // 2
    center_y = (left_top_y + right_bottom_y) // 2

    json_data = {
        "license_plate": license_plate,
        "center_x": center_x,
        "center_y": center_y,
        "left_top_x": left_top_x,
        "left_top_y": left_top_y,
        "right_bottom_x": right_bottom_x,
        "right_bottom_y": right_bottom_y
    }

    call_car_position_update_api(json_data)


def call_license_plate_by_position_api(center_x, center_y):
    try:
        response = requests.get(f"{API_URL}/license_plate_by_position/{center_x}/{center_y}",
                                json={'center_x': center_x, 'center_y': center_y})
        if response.status_code == 200:
            return response.json()['license_plate']
        else:
            return None

    except Exception as e:
        print(f"Error during API call: {e}")
        return None


def update_positions_of_cars(detections):
    for _, detection in detections.iterrows():
        left_top_x = int(detection['xmin'])
        left_top_y = int(detection['ymin'])
        right_bottom_x = int(detection['xmax'])
        right_bottom_y = int(detection['ymax'])

        center_x = (left_top_x + right_bottom_x) // 2
        center_y = (left_top_y + right_bottom_y) // 2

        license_plate = None
        json_data = []
        # get all license plates positions
        car_positions = call_all_license_plates_positions_api()
        if car_positions is not None:
            for car_position in car_positions:
                distance = (center_x - car_position['center_x']) ** 2 + (center_y - car_position['center_y']) ** 2
                if distance < 1000:
                    license_plate = car_position['license_plate']
                    break
            else:
                license_plate = None

            if license_plate is None:
                continue
            else:
                json_data.append({
                    "license_plate": license_plate,
                    "center_x": center_x,
                    "center_y": center_y,
                    "left_top_x": left_top_x,
                    "left_top_y": left_top_y,
                    "right_bottom_x": right_bottom_x,
                    "right_bottom_y": right_bottom_y,
                })
                # print(json_data)
            call_car_position_update_api(json_data)


def update_parking_area(parking_data):
    try:
        for area in parking_data:
            area['top_left_x'] = int(area['top_left'][0])
            area['top_left_y'] = int(area['top_left'][1])
            area['bottom_right_x'] = int(area['bottom_right'][0])
            area['bottom_right_y'] = int(area['bottom_right'][1])
            area.pop('top_left')
            area.pop('bottom_right')

        response = requests.post(f"{API_URL}/parking_areas", json=parking_data)
        if response.status_code == 201:
            return True
        else:
            print(f"Failed to send parking area: {response.text}")
            return False

    except Exception as e:
        print(f"Error during API call: {e}")
        return False


def check_if_last_registered_car_is_out_of_entrance():
    try:
        response = requests.get(f"{API_URL}/is_car_out_of_entrance")
        if response.status_code == 200:
            is_car_out = response.json()['is_out']
            if is_car_out:
                # print(f"out of the entrance.")
                return True
            else:
                # print(f"still in the entrance.")
                return False
        else:
            print(f"Failed to check if car is out of the entrance: {response.text}")
            return False


    except Exception as e:
        print(f"Error during API call: {e}")
        return False


def call_car_exit_api(exit_car_license_plate):
    try:
        response = requests.delete(f"{API_URL}/car_exit/{exit_car_license_plate}")
        if response.status_code == 200:
            print(f"Exit handled successfully!")
            return True
        else:
            print(f"Failed to handle exit: {response.text}")
            return False

    except Exception as e:
        print(f"Error during API call: {e}")
        return False


def call_last_registered_license_plate_api():
    try:
        response = requests.get(f"{API_URL}/last_registered_license_plate")
        if response.status_code == 200:
            return response.json()['license_plate']
        else:
            return None

    except Exception as e:
        print(f"Error during API call: {e}")
        return None


def call_get_license_plate_from_exit_api():
    try:
        response = requests.get(f"{API_URL}/license_plate_from_exit")
        if response.status_code == 200:
            return response.json()['license_plate']
        else:
            return None

    except Exception as e:
        print(f"Error during API call: {e}")
        return None


def call_car_parked_properly_api():
    try:
        response = requests.get(f"{API_URL}/are_properly_parked")
        if response.status_code == 200:
            return response.json()['is_parked_properly'], None
        elif response.status_code == 401:
            return False, None
        else:
            return response.json()['is_parked_properly'], response.json()['improperly_parked_cars']

    except Exception as e:
        print(f"Error during API call: {e}")
        return False


def call_log_api(log_type, data):
    try:
        data_to_str = ""
        if isinstance(data, list) or isinstance(data, tuple) or isinstance(data, dict):
            for d in data:
                data_to_str += f"{d} "
        else:
            data_to_str = data
        response = requests.post(f"{API_URL}/logs", json={'type': log_type, 'log': data_to_str})
        if response.status_code == 201:
            return True
        else:
            print(f"Failed to send log: {response.text}")
            return False

    except Exception as e:
        print(f"Error during API call: {e}")
        return False


def change_barrier_state(is_entry_barrier_down, is_exit_barrier_down, frame):
    """
    Funkcja zmienia stan dwóch szlabanów (wjazdowego i wyjazdowego) i rysuje je na klatce.

    :param is_entry_barrier_down: bool, True jeśli szlaban wjazdowy ma być opuszczony, False jeśli podniesiony.
    :param is_exit_barrier_down: bool, True jeśli szlaban wyjazdowy ma być opuszczony, False jeśli podniesiony.
    :param frame: Obecna klatka wideo.
    """
    barrier_thickness = 10  # Grubość szlabanu

    # Rozmiar klatki
    frame_height, frame_width, _ = frame.shape

    coords_x = int(frame_width * 0.75)

    # Współrzędne szlabanu wjazdowego
    entry_barrier_start = (coords_x, int(frame_height * 0.36))
    entry_barrier_end_down = (coords_x, int(frame_height * 0.5))

    # Współrzędne szlabanu wyjazdowego
    exit_barrier_start = (coords_x, int(frame_height * 0.55))
    exit_barrier_end_down = (coords_x, int(frame_height * 0.68))

    # Rysowanie szlabanu wjazdowego
    if is_entry_barrier_down:
        cv2.line(frame, entry_barrier_start, entry_barrier_end_down, (0, 0, 255), barrier_thickness)
    else:
        cv2.line(frame, entry_barrier_start, entry_barrier_end_down, (0, 255, 0), barrier_thickness)

    # Rysowanie szlabanu wyjazdowego
    if is_exit_barrier_down:
        cv2.line(frame, exit_barrier_start, exit_barrier_end_down, (0, 0, 255), barrier_thickness)
    else:
        cv2.line(frame, exit_barrier_start, exit_barrier_end_down, (0, 255, 0), barrier_thickness)

    return frame

    
def call_car_parked_in_one_slot_api():
    try:
        response = requests.get(f"{API_URL}/takes_one_slot")

        if response.status_code == 200:
            data = response.json()
            improperly_parked_cars = data.get('improperly_parked_cars')  
            is_parked_properly = len(improperly_parked_cars) == 0 if improperly_parked_cars else True
            return is_parked_properly, improperly_parked_cars

        elif response.status_code == 404:
            return True, None

        elif response.status_code == 401:
            return False, None

        else:
            print(f"Unexpected response: {response.status_code} - {response.text}")
            return False, None

    except Exception as e:
        print(f"Error during API call: {e}")
        return False, None
