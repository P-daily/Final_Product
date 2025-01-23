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
    entrance_bottom_right = (col_starts[-1], y_max)
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











