from io import BytesIO
from random import choice

from PIL import Image
import numpy as np
import requests
from Artwork import Artwork
from DataProvider import DataProvider


class ApiProvider:
    def __init__(self, data_provider: DataProvider):
        self.__data_provider = data_provider

    def download_random_artwork(self) -> Artwork:
        ids = self.__data_provider.get_painting_ids()
        if ids is None or len(ids) == 0:
            raise FileNotFoundError

        metadata = {}
        is_primary_image_exists = False
        while not is_primary_image_exists:
            id = choice(ids)
            response = requests.get(f'https://collectionapi.metmuseum.org'
                                    f'/public/collection/v1/objects/{id}')

            metadata = response.json()
            if metadata['primaryImage'] != '':
                is_primary_image_exists = True

        image_bytes = self.__download_image(metadata['primaryImage'])
        image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image_pil)
        return Artwork(image_array, metadata)

    @staticmethod
    def __download_image(url) -> bytes:
        r = requests.get(url)
        r.raise_for_status()
        return r.content