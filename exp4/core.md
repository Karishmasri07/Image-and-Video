
!pip install ultralytics opencv-python matplotlib
# Import libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load pretrained YOLOv8 model (nano version - fast)
model = YOLO("yolov8n.pt")

# Provide your image path here
image_path = "img2.png"   # <-- change to your image file name

# Run detection
results = model(image_path)

# Get annotated image (with bounding boxes)
annotated_image = results[0].plot()

# Convert BGR (OpenCV format) to RGB (Matplotlib format)
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display image inside Jupyter Notebook
plt.figure(figsize=(10, 8))
plt.imshow(annotated_image_rgb)
plt.axis("off")
plt.title("Object Detection Result")
plt.show()

# Print detected objects
print("\nDetected Objects:")
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    label = model.names[class_id]
    print(f"Object: {label}, Confidence: {confidence:.2f}")


<img width="746" height="683" alt="image" src="https://github.com/user-attachments/assets/661a7fd2-2c1b-4ccc-b23e-bde2aa0a9a97" />
