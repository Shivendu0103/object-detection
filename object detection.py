import cv2
import math
import time
from ultralytics import YOLO

def start_realtime_detection(model_variant="yolov8n.pt"):
    """
    Starts a real-time object detection session using the webcam.
    
    Args:
        model_variant (str): The YOLO model file to use. 
                             'yolov8n.pt' is the nano version, fastest for real-time.
    """
    # 1. Load the YOLO model
    # The 'n' stands for Nano, which is the lightest and fastest for real-time applications
    print(f"Loading {model_variant}...")
    model = YOLO(model_variant)

    # 2. Define Class Names (COCO Dataset standard classes)
    class_names = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "Kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

    # 3. Initialize Webcam
    # 0 is usually the default internal webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # Width
    cap.set(4, 720)  # Height

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting detection loop. Press 'q' to exit.")

    # FPS Calculation setup
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # 4. Run Inference
        # stream=True is more memory-efficient for continuous video
        results = model(img, stream=True)

        # 5. Process Detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box Coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence Score
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Index
                cls = int(box.cls[0])
                current_class = class_names[cls]

                # Draw Visuals on the image
                if conf > 0.5: # Only show detections with > 50% confidence
                    # Draw Rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add Label with Background
                    label = f'{current_class} {conf}'
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(img, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 6. Calculate and Display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 7. Show Output
        cv2.imshow("Real-Time Object Detection", img)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # To use YOLOv10 (NMS-free for even lower latency), 
    # install it via: pip install huggingface_hub ultralytics
    # and pass the appropriate model path here.
    start_realtime_detection(model_variant="yolov8n.pt")