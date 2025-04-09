import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def compass_operator(img_path):
    img = Image.open(img_path).convert('L')
    img_np = np.array(img)

    kompas_kernels = {
        'Utara':      np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]]),
        'Timur Laut': np.array([[1, 1, 1], [-1, -2, 1], [-1, -1, 1]]),
        'Timur':      np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]]),
        'Tenggara':   np.array([[-1, -1, 1], [-1, -2, 1], [1, 1, 1]]),
        'Selatan':    np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]]),
        'Barat Daya': np.array([[1, -1, -1], [1, -2, -1], [1, 1, 1]]),
        'Barat':      np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]]),
        'Barat Laut': np.array([[1, 1, 1], [1, -2, -1], [1, -1, -1]])
    }

    edge_maps = {k: convolve(img_np, kernel, mode='reflect') for k, kernel in kompas_kernels.items()}
    edge_combined = np.max(np.stack(list(edge_maps.values())), axis=0)

    return img_np, edge_maps, edge_combined