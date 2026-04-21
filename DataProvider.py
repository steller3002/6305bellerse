import csv
from dataclasses import dataclass
import json
import os
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from Artwork import Artwork
from PIL import Image


@dataclass(frozen=True)
class ArtworkPath:
    image_path: str
    metadata_path: str

class DataProvider:
    def __init__(self, csv_path: str, save_directory: str):
        self.__csv_path = csv_path
        self.__save_directory = save_directory
        if not os.path.exists(self.__save_directory):
            os.makedirs(self.__save_directory)
        if not os.path.exists(self.__csv_path):
            raise FileNotFoundError

    @property
    def csv_path(self): return self.__csv_path
    @property
    def save_directory(self): return self.__save_directory

    def get_painting_ids(self) -> list[str]:
        try:
            data = pd.read_csv(self.csv_path, low_memory=False)
            painting_ids = []
            with open(self.__csv_path, encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row["Classification"] == "Paintings":
                        painting_ids.append(row["Object ID"])

            return painting_ids
        except FileNotFoundError:
            return []

    def clear_save_directory(self) -> None:
        for file_name in os.listdir(self.__save_directory):
            file_path = os.path.join(self.__save_directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def save_artwork_info(self, artwork: Artwork) -> ArtworkPath:
        image_id = artwork.metadata["objectID"]
        image_path = os.path.join(self.__save_directory, f"{image_id}.jpg")
        metadata_path = os.path.join(self.__save_directory, f"{image_id}.json")

        with open(metadata_path, 'w') as f:
            json.dump(artwork.metadata, f)
        img = Image.fromarray(artwork.image)
        img.save(image_path)

        return ArtworkPath(image_path, metadata_path)

    def save_numpy_image(self, ndarray: NDArray[np.uint8], name: str) -> str:
        path = os.path.join(self.__save_directory, f"{name}.jpg")
        img = Image.fromarray(ndarray)
        img.save(path)
        return path