import cv2
import numpy as np

def build_t_pyramid(image, levels):
    pyramid = [image]

    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)

    return pyramid


def main():
    image_path = "img.jpg"
    levels = 3
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Error: could not load the image")
        return

    t_pyramid = build_t_pyramid(original_image, levels)

    for i, level_image in enumerate(t_pyramid):
        cv2.imshow(f"Levels {i}", level_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

    
<img width="318" height="209" alt="Screenshot 2026-02-18 094403" src="https://github.com/user-attachments/assets/387d0aed-a5c6-4593-9074-7a0223508dc1" />

<img width="251" height="165" alt="Screenshot 2026-02-18 095359" src="https://github.com/user-attachments/assets/0057020d-e20d-43df-8632-f07868cba6f9" />

<img width="224" height="142" alt="Screenshot 2026-02-18 095427" src="https://github.com/user-attachments/assets/5cc76d47-5be3-4dfc-8bdb-24c7dbd5fb92" />

