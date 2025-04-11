import cv2
import numpy as np

def prewitt(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    img_x = cv2.filter2D(img, -1, kernel_x)
    img_y = cv2.filter2D(img, -1, kernel_y)
    combined = cv2.addWeighted(img_x, 0.5, img_y, 0.5, 0)

    return img, img_x, img_y, combined