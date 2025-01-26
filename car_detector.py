import cv2

from parking_utils import  call_all_license_plates_positions_api


def is_car(class_name):
    """Check if the detected class is 'car'."""
    return class_name.lower() == "car"


def detect_objects(frame, model, confidence_threshold=0.6):
    results = model(frame)
    detections = results.pandas().xyxy[0]
    filtered_detections = detections[detections['confidence'] >= confidence_threshold]
    return filtered_detections


def draw_detections(frame, detections):
    car_positions =  call_all_license_plates_positions_api()
    for _, row in detections.iterrows():
        class_name = row['name']
        # Check if the detected object is a car
        if is_car(class_name):
            # Bounding box coordinates
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            license_plate = "UNKNOWN CAR"
            # Check if the car is parked
            if car_positions is not None:

                min_distance = 100000
                for car_position in car_positions:
                    distance = (center_x - car_position['center_x']) ** 2 + (center_y - car_position['center_y']) ** 2
                    if distance < min_distance:
                        min_distance = distance
                        license_plate = car_position['license_plate']

                if min_distance > 1000:
                    license_plate = "UNKNOWN CAR"
                cv2.putText(frame, f"{license_plate}", (center_x - 30, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
            cv2.putText(frame, f"{license_plate}", (center_x - 30, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)

    return frame
