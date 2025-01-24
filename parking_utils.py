import cv2
import numpy as np


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

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

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

    # Wyjazd
    exit_top_left = (col_starts[2], row_starts[2])
    exit_bottom_right = (col_starts[4], row_starts[3])
    cv2.rectangle(frame, exit_top_left, exit_bottom_right, (0, 0, 255), 2)
    cv2.putText(frame, "Wyjazd",
                (exit_top_left[0] + 10, exit_top_left[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    parking_data.append({
        "id": "X",
        "type": "X",  # Wyjazd
        "top_left": exit_top_left,
        "bottom_right": exit_bottom_right
    })

    for idx, (blue_count, yellow_count, top_left, bottom_right) in enumerate(parking_areas):
        color = (0, 255, 255)  # Żółty
        area_type = "N"

        if label == 1:
            area_type = "K"  # Kierownictwo
            color = (0, 0, 0)  # Czarny
        elif label == 5:
            area_type = "I"  # Inwalidzi
            color = (255, 255, 255)  # Biały

        parking_data.append({
            "id": label,
            "type": area_type,
            "top_left": top_left,
            "bottom_right": bottom_right
        })

        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {area_type}",
                    (top_left[0] + 10, top_left[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        label += 1

    # Droga
    road_top_left = (x_min, row_starts[1])
    road_bottom_right = (x_max - entrance_width, row_starts[3])
    cv2.putText(frame, "Droga",
                (x_min + 10, row_starts[1] + int((row_heights[1] + row_heights[2]) // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    parking_data.append({
        "id": "D",
        "type": "D",  # Droga
        "top_left": road_top_left,
        "bottom_right": road_bottom_right
    })

    # Wjazd
    entrance_top_left = (col_starts[-2], y_min)
    entrance_bottom_right = (col_starts[-1], y_min + (y_max - y_min) // 2)
    cv2.rectangle(frame, entrance_top_left, entrance_bottom_right, (0, 0, 255), 2)
    cv2.putText(frame, "Wjazd",
                (entrance_top_left[0] + 10, entrance_top_left[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    parking_data.append({
        "id": "W",
        "type": "W",
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


def print_parking_data(parking_data, frame_counter, frame_interval):
    pass
    # if frame_counter % frame_interval == 0:
    #     print("Rozpoznane miejsca parkingowe:")
    #     for data in parking_data:
    #         print(data)


def detect_new_car_on_entrance(detections, parking_data):
    entrance_coords = parking_data[-1]
    entrance_top_left = entrance_coords['top_left']

    for _, detection in detections.iterrows():
        if detection['xmin'] > entrance_top_left[0]:
            return True

    return False


def detect_and_scan_license_plate():
    print("Czytam tablice rejestracyjne")
    return True


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

    # Współrzędne szlabanu wjazdowego
    entry_barrier_start = (int(frame_width * 0.85), int(frame_height * 0.1))
    entry_barrier_end_down = (int(frame_width * 0.85), int(frame_height * 0.3))
    entry_barrier_end_up = (int(frame_width * 0.75), int(frame_height * 0.05))

    # Współrzędne szlabanu wyjazdowego
    exit_barrier_start = (int(frame_width * 0.15), int(frame_height * 0.7))
    exit_barrier_end_down = (int(frame_width * 0.15), int(frame_height * 0.9))
    exit_barrier_end_up = (int(frame_width * 0.25), int(frame_height * 0.65))

    # Rysowanie szlabanu wjazdowego
    if is_entry_barrier_down:
        cv2.line(frame, entry_barrier_start, entry_barrier_end_down, (0, 0, 255), barrier_thickness)
    else:
        cv2.line(frame, entry_barrier_start, entry_barrier_end_up, (0, 255, 0), barrier_thickness)

    # Rysowanie szlabanu wyjazdowego
    if is_exit_barrier_down:
        cv2.line(frame, exit_barrier_start, exit_barrier_end_down, (0, 0, 255), barrier_thickness)
    else:
        cv2.line(frame, exit_barrier_start, exit_barrier_end_up, (0, 255, 0), barrier_thickness)

    return frame


def detect_new_car_on_exit(detections, parking_data):
    """
    Wykrywa nowe samochody znajdujące się na wyjeździe z parkingu.

    :param detections: Wykrycia samochodów (np. DataFrame z danymi detekcji, zawierający współrzędne xmin, ymin, xmax, ymax).
    :param parking_data: Dane o parkingu, w tym informacje o obszarze wyjazdu.
    :return: True, jeśli wykryto samochód w obszarze wyjazdu, w przeciwnym razie False.
    """
    # Pobierz współrzędne obszaru wyjazdu
    exit_coords = next((area for area in parking_data if area["type"] == "X"), None)

    if not exit_coords:
        print("Nie znaleziono obszaru wyjazdu w danych parkingu.")
        return False

    exit_top_left = exit_coords['top_left']
    exit_bottom_right = exit_coords['bottom_right']

    # Sprawdź, czy jakakolwiek detekcja znajduje się w granicach obszaru wyjazdu
    for _, detection in detections.iterrows():
        car_top_left = (detection['xmin'], detection['ymin'])
        car_bottom_right = (detection['xmax'], detection['ymax'])

        # Sprawdzenie, czy samochód znajduje się w obszarze wyjazdu
        if (
            car_bottom_right[0] > exit_top_left[0] and  # prawy dolny x samochodu > lewy górny x wyjazdu
            car_top_left[0] < exit_bottom_right[0] and  # lewy górny x samochodu < prawy dolny x wyjazdu
            car_bottom_right[1] > exit_top_left[1] and  # prawy dolny y samochodu > lewy górny y wyjazdu
            car_top_left[1] < exit_bottom_right[1]     # lewy górny y samochodu < prawy dolny y wyjazdu
        ):
            return True  # Wykryto samochód w obszarze wyjazdu

    return False
