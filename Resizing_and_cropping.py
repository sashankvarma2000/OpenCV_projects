import cv2
import numpy as np

img = cv2.imread("Resources/nn.png")
print(img.shape)
imgResize = cv2.resize(img,(300,300))
print(imgResize.shape)

imgCropped = img[0:800,400:800]

cv2.imshow("Image",img)
cv2.imshow("Resized Image",imgResize)
cv2.imshow("Cropped Image",imgCropped)

cv2.waitKey(0)