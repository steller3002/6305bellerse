from typing import Optional

import numpy as np
from numpy._typing import NDArray

from Artwork import Artwork
from DataProvider import DataProvider
from ApiProvider import ApiProvider
from utils import measure_time


class ImageProcessor:
    def __init__(self, api_provider: ApiProvider,
                 data_provider: DataProvider):
        self.__api_provider = api_provider
        self.__data_provider = data_provider
        self.__current_artwork = None

    @property
    def artwork(self) -> Artwork | None:
        return self.__current_artwork

    @measure_time
    def download_new_artwork(self) -> Artwork:
        print('Начало скачивания...')
        self.__current_artwork = self.__api_provider.download_random_artwork()
        print(f'Скачивание завершено')
        return self.__current_artwork

    def __check_image_existing(self):
        if not self.__current_artwork:
            raise RuntimeError('Изображение не найдено')

    @measure_time
    def halftone(self, save_name: str = "halftone") -> NDArray[np.uint8]:
        self.__check_image_existing()

        print('Обработка изображения')
        result = self.__current_artwork.to_halftone()
        print('Сохранение изображения')
        self.__data_provider.save_numpy_image(result, save_name)
        return result

    @measure_time
    def dithering(self, save_name: str = "dithering") -> NDArray[np.uint8]:
        self.__check_image_existing()

        print('Обработка изображения')
        result = self.__current_artwork.to_halftone_f_sh()
        print('Сохранение изображения')
        self.__data_provider.save_numpy_image(result, save_name)
        return result

    @measure_time
    def sobel(self, save_name: str = "sobel") -> NDArray[np.uint8]:
        self.__check_image_existing()

        print('Обработка изображения')
        result = self.__current_artwork.sobel()
        print('Сохранение изображения')
        self.__data_provider.save_numpy_image(result, save_name)
        return result

    @measure_time
    def gauss_blur(self, size: int = 5, save_name: str = "blurred") -> NDArray[np.uint8]:
        self.__check_image_existing()

        print('Генерация матрицы Гаусса')
        mask = self.__current_artwork.create_gauss_matrix(size)
        print('Обработка изображения')
        result = self.__current_artwork.convolution(mask)
        print('Сохранение изображения')
        self.__data_provider.save_numpy_image(result, save_name)
        return result

    def save_original(self) -> None:
        if self.__current_artwork:
            print('Сохранение изображения')
            self.__data_provider.save_artwork_info(self.__current_artwork)