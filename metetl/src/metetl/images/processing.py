import asyncio
import csv
import json
import os
import time
from concurrent.futures.process import ProcessPoolExecutor
from io import BytesIO
from random import choice

import aiofiles
import aiohttp
import numpy as np
from PIL import Image

from metetl.images.models import Artwork
from logging import getLogger

logger = getLogger(__name__)

class DataProvider:
    def __init__(self, csv_path: str, save_directory: str):
        self.__csv_path = csv_path
        self.__save_directory = save_directory
        if not os.path.exists(self.__save_directory):
            os.makedirs(self.__save_directory)
        if not os.path.exists(self.__csv_path):
            raise FileNotFoundError
        logger.debug("DataProvider инициализирован: csv=%s, dir=%s", csv_path, save_directory)

    @property
    def csv_path(self): return self.__csv_path

    @property
    def save_directory(self): return self.__save_directory

    def get_painting_ids(self) -> list[str]:
        try:
            if self.__csv_path.endswith('.json'):
                with open(self.__csv_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [str(item['object_id']) for item in data]

            painting_ids = []
            with open(self.__csv_path, encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get("Classification") == "Paintings":
                        painting_ids.append(row["Object ID"])
            return painting_ids
        except Exception as e:
            logger.error("Ошибка при получении ID: %s", e)
            return []

    def clear_save_directory(self) -> None:
        for file_name in os.listdir(self.__save_directory):
            file_path = os.path.join(self.__save_directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        logger.debug("Директория %s очищена", self.__save_directory)

    async def save_artwork_info_async(self, artwork: Artwork, filename_prefix: str) -> None:
        image_path = os.path.join(self.__save_directory, f"{filename_prefix}.png")
        metadata_path = os.path.join(self.__save_directory, f"{filename_prefix}.json")

        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(artwork.metadata))

        img = Image.fromarray(artwork.image)
        buf = BytesIO()
        img.save(buf, format='PNG')
        async with aiofiles.open(image_path, 'wb') as f:
            await f.write(buf.getvalue())

        logger.debug("Сохранено: %s и %s", image_path, metadata_path)

    async def save_numpy_image_async(self, ndarray, filename: str) -> None:
        path = os.path.join(self.__save_directory, f"{filename}.png")
        img = Image.fromarray(ndarray)
        buf = BytesIO()
        img.save(buf, format='PNG')
        async with aiofiles.open(path, 'wb') as f:
            await f.write(buf.getvalue())
        logger.debug("Сохранено обработанное изображение: %s", path)


def run_cpu_tasks_in_pool(artwork: Artwork, index: int, painting_id: str):
    pid = os.getpid()
    logger.debug("Свёртка для изображения %d началась (PID %d)", index, pid)

    t = time.perf_counter()
    halftone_res = artwork.to_halftone()
    logger.debug("Полутонирование изображения %d выполнено за: %.3fс", index, time.perf_counter() - t)

    t = time.perf_counter()
    dithering_res = artwork.to_halftone_f_sh()
    logger.debug("Дизеринг изображения %d выполнен за: %.3fс", index, time.perf_counter() - t)

    t = time.perf_counter()
    sobel_res = artwork.sobel()
    logger.debug("Собель изображения %d выполнен за: %.3fс", index, time.perf_counter() - t)

    t = time.perf_counter()
    mask = artwork.create_gauss_matrix(3)
    blurred_res = artwork.convolution(mask)
    logger.debug("Размытие изображения %d выполнено за: %.3fс", index, time.perf_counter() - t)

    logger.debug("Свёртка для изображения %d окончена (PID %d)", index, pid)

    return halftone_res, dithering_res, sobel_res, blurred_res



class ApiProvider:
    def __init__(self, data_provider: DataProvider):
        self.__data_provider = data_provider
        logger.debug("ApiProvider инициализирован")

    async def download_random_artwork_async(self, session: aiohttp.ClientSession) -> Artwork:
        ids = self.__data_provider.get_painting_ids()
        if not ids:
            raise FileNotFoundError("ID картин не найдены")

        while True:
            art_id = choice(ids)
            url = f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{art_id}'
            logger.debug("Запрос метаданных: %s", url)

            async with session.get(url) as response:
                if response.status != 200:
                    logger.debug("Пропуск ID %s (статус %d)", art_id, response.status)
                    continue
                metadata = await response.json()

                if metadata.get('primaryImage'):
                    image_url = metadata['primaryImage']
                    async with session.get(image_url) as img_response:
                        if img_response.status == 200:
                            image_bytes = await img_response.read()
                            image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
                            image_array = np.array(image_pil)
                            logger.debug("Изображение %s загружено, размер %s", art_id, image_array.shape)
                            return Artwork(image_array, metadata)


class ImageProcessor:
    def __init__(self, api_provider: ApiProvider, data_provider: DataProvider,
                 index: int, session: aiohttp.ClientSession, pool: ProcessPoolExecutor):
        self.__api_provider = api_provider
        self.__data_provider = data_provider
        self.index = index
        self.session = session
        self.pool = pool
        self.__current_artwork = None

    async def process_pipeline(self):
        logger.info("Началось скачивание изображения номер %d", self.index)
        self.__current_artwork = await self.__api_provider.download_random_artwork_async(self.session)
        painting_id = self.__current_artwork.metadata.get("objectID", "unknown")
        logger.info("Скачивание изображения номер %d завершилось (ID: %s)", self.index, painting_id)

        prefix_original = f"{self.index}_{painting_id}_original"
        await self.__data_provider.save_artwork_info_async(self.__current_artwork, prefix_original)

        loop = asyncio.get_running_loop()
        h_res, d_res, s_res, b_res = await loop.run_in_executor(
            self.pool,
            run_cpu_tasks_in_pool,
            self.__current_artwork,
            self.index,
            painting_id
        )

        await self.__data_provider.save_numpy_image_async(h_res, f"{self.index}_{painting_id}_processed_halftone")
        await self.__data_provider.save_numpy_image_async(d_res, f"{self.index}_{painting_id}_processed_dithering")
        await self.__data_provider.save_numpy_image_async(s_res, f"{self.index}_{painting_id}_processed_sobel")
        await self.__data_provider.save_numpy_image_async(b_res, f"{self.index}_{painting_id}_processed_blurred")
        logger.debug("Все файлы изображения №%d сохранены", self.index)