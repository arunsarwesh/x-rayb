import cv2
import torch
from ultralytics import YOLO

# Load the trained model
model = YOLO("yolo11n.pt")  # Ensure the path is correct

# Load the image
image_path = "images.jpg"  # Change this to your image path
image = cv2.imread(image_path)

# Run inference
results = model(image)[0]  # Get first result

# Process results
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    conf = box.conf[0]  # Confidence score
    cls = int(box.cls[0])  # Class ID

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display label
    label = f"{model.names[cls]} {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output image
cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
