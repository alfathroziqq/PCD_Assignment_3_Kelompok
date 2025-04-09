import cv2
import numpy as np

def LoGFilter(img_path, sigma=2.0):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    size = int(6 * sigma + 1)
    if size % 2 == 0: size += 1

    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1),
                       np.arange(-size//2+1, size//2+1))

    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2)/(2*sigma**2))) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel /= np.sum(np.abs(kernel))

    result = cv2.filter2D(img, -1, kernel)
    result = cv2.convertScaleAbs(result)

    return img, result