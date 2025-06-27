import cv2
import torch
import random
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors

# Load YOLO model with segmentation capability (YOLOv8n-seg is a small, fast model)
model = YOLO("yolov8n-seg.pt")  # Download automatically if not present

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set resolution (adjust based on your webcam's capabilities)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366/2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768/2)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_skip = 1  # Adjust this value to analyze every X-th frame
frame_count = 0
alpha = 0.3  # Transparency level for segmentation mask (0 = fully transparent, 1 = opaque)
factor = 2.1  # Upscaling factor

# Assign random colors for each class
colors = Colors()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only analyze every X-th frame
    if frame_count % frame_skip == 0:
        results = model(frame)  # Run YOLO segmentation

        overlay = frame.copy()  # Create an overlay for transparency effect

        # Draw segmentation masks and bounding boxes on the frame
        try:
            for result in results:
                for box, mask in zip(result.boxes, result.masks.xy):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    confidence = box.conf[0]  # Confidence score
                    class_id = int(box.cls[0])  # Class ID
                    label = f"{model.names[class_id]} {confidence:.2f}"  # Label with confidence

                    # Assign a unique color for each class
                    random.seed(class_id)
                    c = random.randint(0, colors.n)
                    color = colors.palette[c]

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw segmentation mask
                    mask_pts = np.array(mask, np.int32)  # Convert mask points to NumPy array
                    cv2.fillPoly(overlay, [mask_pts], color)  # Fill mask on overlay
        except Exception:
            pass

        # Apply transparent overlay
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Upscale frame
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * factor), int(frame.shape[0] * factor)))

        # Display frame
        title = "YOLOv8-Seg Webcam"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(title, frame_resized)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
