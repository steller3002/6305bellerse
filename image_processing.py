import numpy as np
import math
from utils import measure_time

def create_gauss_matrix(n):
    sigma = (n - 1) / 6
    matrix = np.zeros((n, n))
    center = n // 2
    for y in range(n):
        for x in range(n):
            offset_x = abs(x - center)
            offset_y = abs(y - center)
            matrix[y][x] = math.e ** -(((offset_x ** 2) + (offset_y ** 2)) / (2*sigma**2))
    return matrix / np.sum(matrix)

@measure_time
def to_halftone_f_sh(array):
    # Алгоритм дизеринга Флойда-Штайнберга
    gray_array = (np.array(0.299*array[:,:,0] + 0.587*array[:,:,1] + 0.114*array[:,:,2]))
    h, w = len(gray_array), len(gray_array[0])

    for y in range(h-1):
        for x in range(1, w-1):
            old_pixel = gray_array[y][x]
            new_pixel = 255 if old_pixel > 128 else 0
            gray_array[y][x] = new_pixel

            error = old_pixel - new_pixel
            gray_array[y, x+1] += error * 7 / 16
            gray_array[y+1, x-1] += error * 3 / 16
            gray_array[y+1, x] += error * 5 / 16
            gray_array[y+1, x+1] += error * 1 / 16

    return gray_array.astype(np.uint8)

@measure_time
def convolution(array, mask):
    result = np.zeros_like(array).astype(np.float32)
    mask = np.array(mask).astype(np.float32)
    mask_sum = np.sum(mask)

    indent = len(mask[0]) // 2
    h, w = len(array), len(array[0])

    for y in range(indent, h-indent):
        for x in range(indent, w-indent):
            field = array[y-indent:y+indent+1,x-indent:x+indent+1]
            field_on_mask = field * mask[:, :, np.newaxis]
            result[y][x] = np.sum(field_on_mask, axis=(0, 1)) / mask_sum

    return np.clip(result, 0, 255).astype(np.uint8)

@measure_time
def sobel(array):
    if len(array[0][0]) == 3:
        array = to_halftone(array)

    h, w = len(array), len(array[0])
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    result = np.zeros_like(array).astype(np.float32)

    for y in range(1, h-1):
        for x in range(1, w-1):
            field = array[y - 1:y + 2, x - 1:x + 2]
            field_on_g_x = np.sum(field * g_x)
            field_on_g_y = np.sum(field * g_y)
            result[y][x] = np.sqrt(field_on_g_x ** 2 + field_on_g_y ** 2)

    return np.clip(result, 0, 255).astype(np.uint8)

@measure_time
def to_halftone(array):
    gray_array = (np.array(0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]))
    return gray_array.astype(np.uint8)