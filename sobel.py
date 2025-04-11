import cv2
import numpy as np

def sobel(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

    return image, np.abs(sobel_x).astype(np.uint8), np.abs(sobel_y).astype(np.uint8), sobel_magnitude