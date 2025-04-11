import cv2
import numpy as np

def roberts_edge_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)

    edge_x = cv2.filter2D(gray, -1, kernel_x)
    edge_y = cv2.filter2D(gray, -1, kernel_y)

    edge_roberts = np.sqrt(np.square(edge_x) + np.square(edge_y))
    edge_roberts = np.clip(edge_roberts, 0, 255).astype(np.uint8)
    edge_roberts = cv2.normalize(edge_roberts, None, 0, 255, cv2.NORM_MINMAX)

    return gray, edge_roberts