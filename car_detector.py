import cv2

# Function to check if the detected class is "car"
def is_car(class_name):
    return class_name.lower() == "car"


def detect_car_and_return_frame(frame, model, confidence_threshold=0.6):
    """
    Perform object detection on a single frame and return the processed frame.

    Args:
        frame (numpy.ndarray): Input frame to process.

    Returns:
        numpy.ndarray: Processed frame with bounding boxes and labels drawn.
        :param frame:
        :param model:
        :param confidence_threshold:
    """

    results = model(frame)

    # Get detections in pandas format
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        # Filter detections below the confidence threshold
        if row['confidence'] < confidence_threshold:
            continue

        # Get class name
        class_name = row['name']

        # Check if the detected object is a car
        if is_car(class_name):
            # Bounding box coordinates
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{class_name} ({row['confidence']:.2f})"

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame, detections

