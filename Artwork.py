import math
from typing import Dict

import cv2
import numpy as np
from numpy.typing import NDArray

class Artwork:
    __slots__ = ('__nparray', '__metadata')

    def __init__(self, nparray: NDArray[np.uint8], metadata: Dict[str, str]):
        self.__nparray = nparray
        self.__metadata = metadata

    @property
    def image(self) -> NDArray[np.uint8]: return self.__nparray
    @property
    def metadata(self) -> Dict[str, str]: return self.__metadata

    def to_halftone(self) -> NDArray[np.uint8]:
        gray_array = np.array(
            0.299 * self.__nparray[:, :, 0]
           + 0.587 * self.__nparray[:, :, 1]
           + 0.114 * self.__nparray[:, :, 2])
        return gray_array.astype(np.uint8)

    def to_halftone_f_sh(self) -> NDArray[np.uint8]:
        # Алгоритм дизеринга Флойда-Штайнберга
        gray_array = np.array(
            0.299 * self.__nparray[:, :, 0]
           + 0.587 * self.__nparray[:, :, 1]
           + 0.114 * self.__nparray[:, :, 2]).astype(np.float32)

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
        h, w, c = self.__nparray.shape
        result = np.zeros_like(self.__nparray, dtype=np.float32)

        indent = len(mask) // 2
        refined_mask = mask[:, :, np.newaxis]

        for y in range(indent, h - indent):
            for x in range(indent, w - indent):
                field = self.__nparray[y - indent: y + indent + 1, x - indent: x + indent + 1]
                pixel_sum = np.sum(field * refined_mask, axis=(0, 1))
                result[y, x] = pixel_sum

        return np.clip(result, 0, 255).astype(np.uint8)

    def __add__(self, other: 'Artwork') -> 'Artwork':
        target_h, target_w = self.__nparray.shape[:2]
        resized_other = cv2.resize(
            other.image,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

        result = (
                (self.__nparray.astype(np.uint16) + resized_other.astype(np.uint16)) // 2
        ).astype(np.uint8)

        return Artwork(result, self.__metadata.copy())

    def __str__(self) -> str:
        return str(self.__metadata.copy())

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