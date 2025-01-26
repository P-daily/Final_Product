import cv2

from parking_utils import call_license_plate_by_position_api, call_last_registered_license_plate_api, \
    assign_license_plate_by_position


def is_car(class_name):
    """Check if the detected class is 'car'."""
    return class_name.lower() == "car"


def detect_objects(frame, model, confidence_threshold=0.6):
    results = model(frame)
    detections = results.pandas().xyxy[0]
    filtered_detections = detections[detections['confidence'] >= confidence_threshold]
    return filtered_detections


def draw_detections(frame, detections):
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

            # zrób enddpoint który zwróci od razu dla wszystkich samochodów

            license_plate = assign_license_plate_by_position(center_x, center_y)
            cv2.putText(frame, f"{license_plate}", (center_x - 30, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)

    return frame
