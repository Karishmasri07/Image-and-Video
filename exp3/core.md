import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img.jpg", 0)  
h, w = img.shape


angle = 45
M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
rotated = cv2.warpAffine(img, M_rot, (w, h))

scaled = cv2.resize(img, None, fx=1.5, fy=1.5)

M_shear = np.float32([[1, 0.5, 0],
                      [0, 1,   0]])
sheared = cv2.warpAffine(img, M_shear, (int(w*1.5), h))

pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])
pts2 = np.float32([[50,50], [w-50,50], [50,h-50]])
M_affine = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M_affine, (w, h))
pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
pts2 = np.float32([[50,50], [w-50,30], [70,h-50], [w-30,h-30]])
M_bilinear = cv2.getPerspectiveTransform(pts1, pts2)
bilinear = cv2.warpPerspective(img, M_bilinear, (w, h))

titles = ["Original", "Rotated", "Scaled", "Sheared",
          "Affine Transform", "Bilinear Transform"]
images = [img, rotated, scaled, sheared, affine, bilinear]

plt.figure(figsize=(12,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()

<img width="669" height="358" alt="Screenshot 2026-02-18 092252" src="https://github.com/user-attachments/assets/8cca4425-df39-4d6a-a07b-7a3f2b437f5f" />

<img width="1599" height="762" alt="image" src="https://github.com/user-attachments/assets/ce59bad6-3d86-4b38-a311-fe4d5a9cff5a" />

<img width="1594" height="769" alt="image" src="https://github.com/user-attachments/assets/99a53f3b-3c11-4136-8eb8-9a465d3c1ca7" />

