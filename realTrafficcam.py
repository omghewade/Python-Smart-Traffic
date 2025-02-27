import cv2
import numpy as np

# Full paths to the YOLO configuration and weights files
cfg_path = "./yolov3.cfg"
weights_path = "./yolov3.weights"
names_path = "./coco.names"

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the video capture object with your stream URL
cap = cv2.VideoCapture("http://83.91.176.170:80/mjpg/video.mjpg")

# Define the ROI coordinates and dimensions (not used initially)
roi_top_left_x = 100
roi_top_left_y = 100
roi_width = 400
roi_height = 400

# Initialize a dictionary to keep track of car positions and speeds
car_data = {}


def detect_vehicles(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class ID for cars in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return len(indexes), boxes, class_ids, confidences, indexes


while True:
    ret, frame = cap.read()

    num_vehicles, boxes, class_ids, confidences, indexes = detect_vehicles(frame)

    current_car_data = {}
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            center_x = x + w // 2
            center_y = y + h // 2
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Calculate speed in pixels per second
            car_id = (x, y)
            # if car_id in car_data:
            #     prev_x, prev_y = car_data[car_id]
            #     # speed_px_per_sec = center_x - prev_x * 1000
            #     # np.sqrt(
            #     #     (center_x - prev_x) ** 2 + (center_y - prev_y) ** 2
            #     # )
            #     # Convert speed to km/h assuming 1 pixel = 1 meter
            #     # speed_km_per_hr = speed_px_per_sec * 3.6
            #     speed_km_per_hr = center_x - prev_x
            # else:
            #     speed_km_per_hr = 0

            current_car_data[car_id] = (center_x, center_y)
            cv2.putText(
                frame,
                f"",
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Collision probability based on distance to other cars
            collision_probability = 0
            for other_id, (other_x, other_y) in current_car_data.items():
                if car_id != other_id:
                    distance = np.sqrt(
                        (center_x - other_x) ** 2 + (center_y - other_y) ** 2
                    )
                    if distance < 40:  # Threshold for collision probability
                        collision_probability += 1
            if collision_probability != 0:
                cv2.putText(
                    frame,
                    f"cp: {collision_probability}",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),  # Red color for collision probability
                    2,
                )

    car_data = current_car_data

    cv2.putText(
        frame,
        f"Number of vehicles: {num_vehicles}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.imshow("Traffic Signal Camera", frame)

    # Here you can add your logic to schedule the traffic signal based on num_vehicles

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
