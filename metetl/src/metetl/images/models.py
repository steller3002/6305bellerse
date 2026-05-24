import math
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ArtworkPath:
    image_path: str
    metadata_path: str


class Artwork:
    __slots__ = ("__nparray", "__metadata")

    def __init__(self, nparray: NDArray[np.uint8], metadata: Dict[str, str]):
        self.__nparray = nparray
        self.__metadata = metadata

    @property
    def image(self) -> NDArray[np.uint8]:
        return self.__nparray

    @property
    def metadata(self) -> Dict[str, str]:
        return self.__metadata

    def to_halftone(self) -> NDArray[np.uint8]:
        gray = (
            0.299 * self.__nparray[:, :, 0]
            + 0.587 * self.__nparray[:, :, 1]
            + 0.114 * self.__nparray[:, :, 2]
        )
        return gray.astype(np.uint8)

    def to_halftone_f_sh(self) -> NDArray[np.uint8]:
        gray = (
            0.299 * self.__nparray[:, :, 0]
            + 0.587 * self.__nparray[:, :, 1]
            + 0.114 * self.__nparray[:, :, 2]
        ).astype(np.float32)

        h, w = gray.shape
        for y in range(h - 1):
            for x in range(1, w - 1):
                old = gray[y, x]
                new = 255.0 if old > 128 else 0.0
                gray[y, x] = new
                err = old - new
                gray[y, x + 1] += err * 7 / 16
                gray[y + 1, x - 1] += err * 3 / 16
                gray[y + 1, x] += err * 5 / 16
                gray[y + 1, x + 1] += err * 1 / 16

        return gray.astype(np.uint8)

    def sobel(self) -> NDArray[np.uint8]:
        array = self.to_halftone()
        h, w = array.shape
        g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        result = np.zeros_like(array, dtype=np.float32)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                field = array[y - 1 : y + 2, x - 1 : x + 2]
                gx = np.sum(field * g_x)
                gy = np.sum(field * g_y)
                result[y, x] = np.sqrt(gx**2 + gy**2)

        return np.clip(result, 0, 255).astype(np.uint8)

    def convolution(self, mask: NDArray[np.float32]) -> NDArray[np.uint8]:
        h, w, _ = self.__nparray.shape
        result = np.zeros_like(self.__nparray, dtype=np.float32)
        indent = len(mask) // 2
        refined_mask = mask[:, :, np.newaxis]

        for y in range(indent, h - indent):
            for x in range(indent, w - indent):
                field = self.__nparray[y - indent : y + indent + 1, x - indent : x + indent + 1]
                result[y, x] = np.sum(field * refined_mask, axis=(0, 1))

        return np.clip(result, 0, 255).astype(np.uint8)


    @staticmethod
    def create_gauss_matrix(n: int) -> NDArray[np.float32]:
        sigma = (n - 1) / 6 or 1.0
        center = n // 2
        matrix = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                ox, oy = abs(x - center), abs(y - center)
                matrix[y, x] = math.e ** (-((ox**2 + oy**2) / (2 * sigma**2)))
        return (matrix / np.sum(matrix)).astype(np.float32)

    def __add__(self, other: "Artwork") -> "Artwork":
        th, tw = self.__nparray.shape[:2]
        resized = cv2.resize(other.image, (tw, th), interpolation=cv2.INTER_AREA)
        blended = ((self.__nparray.astype(np.uint16) + resized.astype(np.uint16)) // 2).astype(np.uint8)
        return Artwork(blended, self.__metadata.copy())

    def __str__(self) -> str:
        return str(self.__metadata.copy())