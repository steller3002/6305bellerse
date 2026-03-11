import math
from typing import Dict
import numpy as np
from numpy.typing import NDArray

class Artwork:
    def __init__(self, nparray: NDArray[np.uint8], metadata: Dict[str, str]):
        self.nparray = nparray
        self.metadata = metadata

    def to_halftone(self) -> NDArray[np.uint8]:
        gray_array = np.array(
            0.299 * self.nparray[:, :, 0]
           + 0.587 * self.nparray[:, :, 1]
           + 0.114 * self.nparray[:, :, 2])
        return gray_array.astype(np.uint8)

    def to_halftone_f_sh(self) -> NDArray[np.uint8]:
        # Алгоритм дизеринга Флойда-Штайнберга
        gray_array = np.array(
            0.299 * self.nparray[:, :, 0]
           + 0.587 * self.nparray[:, :, 1]
           + 0.114 * self.nparray[:, :, 2]).astype(np.float32)

        w = len(gray_array[0])
        h = len(gray_array)

        for y in range(h - 1):
            for x in range(1, w - 1):
                old_pixel = gray_array[y][x]
                new_pixel = 255 if old_pixel > 128 else 0
                gray_array[y][x] = new_pixel

                error = old_pixel - new_pixel
                gray_array[y, x + 1] += error * 7 / 16
                gray_array[y + 1, x - 1] += error * 3 / 16
                gray_array[y + 1, x] += error * 5 / 16
                gray_array[y + 1, x + 1] += error * 1 / 16

        return gray_array.astype(np.uint8)

    def sobel(self) -> NDArray[np.uint8]:
        array = self.to_halftone()

        h, w = len(array), len(array[0])
        g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        result = np.zeros_like(array).astype(np.float32)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                field = array[y - 1:y + 2, x - 1:x + 2]
                field_on_g_x = np.sum(field * g_x)
                field_on_g_y = np.sum(field * g_y)
                result[y][x] = np.sqrt(field_on_g_x**2 + field_on_g_y**2)

        return np.clip(result, 0, 255).astype(np.uint8)

    def convolution(self, mask: NDArray[np.float32]) -> NDArray[np.uint8]:
        result = np.zeros_like(self.nparray).astype(np.float32)
        mask_sum = np.sum(mask)

        indent = len(mask[0]) // 2
        h, w = len(self.nparray), len(self.nparray[0])
        if mask.ndim == 3:
            mask = mask[:, :, np.newaxis]

        for y in range(indent, h - indent):
            for x in range(indent, w - indent):
                field = self.nparray[y - indent:y + indent + 1, x - indent:x + indent + 1]
                field_on_mask = field * mask
                result[y][x] = np.sum(field_on_mask, axis=(0, 1))

        return np.clip(result, 0, 255).astype(np.uint8)

    def __add__(self, other: 'Artwork') -> 'Artwork':
        h_self, w_self = len(self.nparray), len(self.nparray[0])
        h_other, w_other = len(other.nparray), len(other.nparray[0])
        nparray_other = other.nparray

        if h_self != h_other or w_self != w_other:
            temp = np.zeros_like(self.nparray).astype(np.float32)
            h_min, w_min = min(h_self, h_other), min(w_self, w_other)
            temp[:h_min, :w_min] = nparray_other[:h_min, :w_min]
            nparray_other = temp

        result = (self.nparray.astype(np.uint16) + nparray_other.astype(np.uint16)) // 2
        return Artwork(result, self.metadata.copy())

    def __str__(self) -> str:
        return str(self.metadata.copy())

    @staticmethod
    def create_gauss_matrix(n) -> NDArray[np.float32]:
        sigma: float = (n - 1) / 6
        if sigma <= 0: sigma = 1
        matrix = np.zeros((n, n))
        center = n // 2
        for y in range(n):
            for x in range(n):
                offset_x = abs(x - center)
                offset_y = abs(y - center)
                matrix[y][x] = math.e ** -(((offset_x ** 2) + (offset_y ** 2)) / (2 * sigma ** 2))
        return matrix / np.sum(matrix)