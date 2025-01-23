import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Replace with your IP camera's URL
ip_camera_url = "http://10.128.161.198:8080/video"  # Example URL
cap = cv2.VideoCapture(ip_camera_url)

# Load YOLO model
model = YOLO("best_plate_detector_model.pt")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
if not cap.isOpened():
    print("Error: Unable to connect to the IP camera.")
    exit()

# Process every n-th frame
frame_skip_interval = 10
frame_count = 0

while True:
    ret, frame = cap.read()  # Capture a frame from the video stream
    if not ret:
        print("Failed to grab frame from IP camera.")
        break

    frame_count += 1
    # Skip frames that aren't multiples of the interval
    if frame_count % frame_skip_interval != 0:
        continue

    # Perform inference on the current frame
    results = model(frame)

    # Extract bounding box information
    if results and results[0].boxes:  # Check if any detections exist
        for box in results[0].boxes:  # Iterate over detected boxes
            xyxy = box.xyxy.cpu().numpy()[0]  # Extract coordinates
            x1, y1, x2, y2 = map(int, xyxy)

            # Crop the image using the coordinates
            cropped_img = frame[y1:y2, x1:x2]  # Crop the license plate region

            # Perform OCR on the cropped image
            result = ocr.ocr(cropped_img, cls=True)

            if result and result[0]:
                detected_text = result[0][0][1][0]
                print(f"Detected Text: {detected_text}")
                cv2.putText(frame, detected_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw a bounding box around the detected license plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with detected license plates and OCR text
    cv2.imshow("IP Camera Real-Time License Plate Recognition", frame)

    # Break the loop when the user presses 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
