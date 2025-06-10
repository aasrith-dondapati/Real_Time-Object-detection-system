
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import torch
import numpy as np

def run_object_detection():
    # 1. Load YOLOv5 Model
    # You can choose different model sizes: 'yolov5s' (small), 'yolov5m' (medium), 'yolov5l' (large), 'yolov5x' (extra large)
    # Smaller models are faster but less accurate, larger models are slower but more accurate.
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        print("Please ensure you have an active internet connection or have downloaded the model weights previously.")
        return

    # Set confidence threshold (adjust as needed)
    # Objects detected with a confidence below this threshold will be ignored.
    model.conf = 0.6
    # Set Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS)
    # NMS removes duplicate bounding boxes for the same object.
    model.iou = 0.5

    # 2. Initialize Webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam. Try 1, 2, etc., if not working.

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 3. Perform Object Detection
        # YOLOv5 expects images in RGB format, but OpenCV reads in BGR.
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)

        # 4. Process Results and Draw Bounding Boxes
        # The 'results' object contains detections.
        # results.pandas().xyxy[0] provides detections in a pandas DataFrame.
        # Columns: xmin, ymin, xmax, ymax, confidence, class, name
        detections = results.pandas().xyxy[0]

        for index, row in detections.iterrows():
            xmin, ymin, xmax, ymax, confidence, class_id, name = row[:7]

            # Convert coordinates to integers
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            # Draw bounding box
            color = (0, 255, 0)  # Green color for bounding box
            thickness = 2
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)

            # Put label and confidence
            label = f"{name} ({confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = xmin
            text_y = ymin - 10 if ymin - 10 > 10 else ymin + text_size[1] + 10 # Adjust text position

            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), color, -1) # Background for text
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA) # Black text

        # 5. Display the frame
        cv2.imshow('YOLOv5 Real-time Object Detection', frame)

        # 6. Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_object_detection()