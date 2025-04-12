import numpy as np
from PIL import Image
from scipy import ndimage

def compass_operator(image, direction='all'):
    img_gray = np.array(image.convert('L')).astype(float)

    compass_filters = {
        'Utara': np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]]),
        'Timur Laut': np.array([[1, 1, 1], [-1, -2, 1], [-1, -1, 1]]),
        'Timur': np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]]),
        'Tenggara': np.array([[-1, -1, 1], [-1, -2, 1], [1, 1, 1]]),
        'Selatan': np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]]),
        'Barat Daya': np.array([[1, -1, -1], [1, -2, -1], [1, 1, 1]]),
        'Barat': np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]]),
        'Barat Laut': np.array([[1, 1, 1], [1, -2, -1], [1, -1, -1]])
    }

    if direction != 'all':
        gradient = ndimage.convolve(img_gray, compass_filters[direction])
        gradient = np.abs(gradient)
    else:
        responses = []
        for kernel in compass_filters.values():
            response = ndimage.convolve(img_gray, kernel)
            responses.append(response)
        gradient = np.sqrt(np.sum(np.square(np.array(responses)), axis=0))

        threshold = 20
        gradient[gradient < threshold] = 0

    gradient = (gradient / gradient.max() * 255).astype(np.uint8)
    return Image.fromarray(gradient)