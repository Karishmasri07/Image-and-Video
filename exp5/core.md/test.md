import cv2
import numpy as np
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# Load video file
cap = cv2.VideoCapture("853889-hd_1920_1080_25fps.mp4")

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Edge detection for first frame
prev_edges = cv2.Canny(prev_gray, 50, 150)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Find motion by comparing edges
    motion = cv2.absdiff(prev_edges, edges)

    # Threshold motion
    _, motion_mask = cv2.threshold(motion, 30, 255, cv2.THRESH_BINARY)

    # Show result in Jupyter
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Edges")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Moving Edges (Motion)")
    plt.imshow(motion_mask, cmap="gray")
    plt.axis("off")

    display(plt.gcf())
    clear_output(wait=True)

    prev_edges = edges.copy()

cap.release()

<img width="637" height="595" alt="image" src="https://github.com/user-attachments/assets/2e333211-7a66-44fa-8bd7-7663dd64f60d" />
<img width="641" height="589" alt="image" src="https://github.com/user-attachments/assets/fc24daed-a5e3-4928-afc1-2844361c0d4e" />
<img width="546" height="599" alt="image" src="https://github.com/user-attachments/assets/d39e2df0-3414-4594-ac11-d1967d24723a" />

