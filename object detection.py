import cv2
import math
import time
import os
from ultralytics import YOLO, YOLOWorld

def start_realtime_detection(model_path="yolov8s-world.pt"):
    """
    Starts a real-time object detection session.
    Automatically handles both YOLO-World (open-vocabulary) 
    and standard YOLO models (custom trained).
    
    Args:
        model_path (str): Path to the .pt model file. 
                          Can be a standard YOLOv8 model or a YOLO-World model.
    """
    # 1. Determine Model Type
    is_world_model = "world" in model_path.lower()
    
    print(f"Loading model: {model_path}...")
    if is_world_model:
        model = YOLOWorld(model_path)
    else:
        model = YOLO(model_path)

    # 2. Setup Classes
    if is_world_model:
        # YOLO-World needs manual class definition
        custom_classes = [
            # People & Personal Items
            "person", "face", "hand", "spectacles", "glasses", "wristwatch", "watch",
            "backpack", "handbag", "wallet", "keys", "id card", "lanyard", "umbrella",
            
            # Electronics (College essentials)
            "mobile phone", "smartphone", "laptop", "tablet", "ipad", "keyboard", 
            "mouse", "monitor", "power adapter", "calculator", "headphones", "earbuds",
            
            # Stationery & Study Materials
            "book", "textbook", "notebook", "pen", "pencil", "marker", "highlighter",
            "ruler", "scissors", "stapler", "folder", "binder", "paper",
            
            # Furniture & Classroom Environment
            "chair", "desk", "table", "whiteboard", "projector", "bench", "lamp",
            "clock", "trash can", "water dispenser",
            
            # Food & Drink
            "bottle", "water bottle", "cup", "coffee cup", "mug", "sandwich", "apple", "banana",
            
            # Miscellaneous Real Life Objects
            "bicycle", "scooter", "helmet", "shoes", "sneakers", "fan", "remote"
        ]
        print(f"YOLO-World detected. Setting detection classes to expanded list.")
        model.set_classes(custom_classes)
        class_names = custom_classes
    else:
        # Standard YOLO models have classes baked in
        class_names = model.names
        print(f"Standard YOLO model detected with classes: {list(class_names.values())[:10]}...")

    # 3. Initialize Webcam
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280) # Width
    cap.set(4, 720)  # Height

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting detection loop with {len(class_names)} classes. Press 'q' to exit.")

    # FPS Calculation setup
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # 4. Run Inference
        results = model.predict(img, stream=True, conf=0.3)

        # 5. Process Detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box Coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence Score
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                current_class = class_names[cls] if isinstance(class_names, list) else class_names[cls]

                # Draw Visuals
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 100), 2)
                label = f'{current_class} {conf}'
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(img, (x1, y1 - 25), (x1 + t_size[0] + 5, y1), (255, 0, 100), -1)
                cv2.putText(img, label, (x1 + 2, y1 - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 6. Display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Object Detection - Multi-Model Support", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # YOLO VERSION OPTIONS:
    # 1. YOLO26n (The LATEST version - Optimized for Edge/Fastest)
    # model_file = "yolo26n.pt" 
    
    # 2. YOLO11s (Extremely reliable and accurate)
    # model_file = "yolo11s.pt" 
    
    # 3. YOLO-World (Use this for custom items: watch, id card, pen, shoes, etc.)
    model_file = "yolov8s-world.pt" 
    
    # 4. Use your OWN custom trained model (.pt file)
    # model_file = "path/to/your/custom_model.pt" 
    
    # Note: YOLO26 is NMS-free and significantly faster on edge devices.
    # It is fully supported by the current Ultralytics library.
    
    start_realtime_detection(model_path=model_file)