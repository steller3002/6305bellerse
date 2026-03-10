import csv
from dataclasses import dataclass
import json
import os

import numpy as np
from numpy._typing import NDArray

from Artwork import Artwork
from PIL import Image


@dataclass
class ArtworkPath:
    image_path: str
    metadata_path: str

class DataProvider:
    def __init__(self, csv_path: str, save_directory: str):
        self.csv_path = csv_path
        self.save_directory = save_directory
        if not os.path.exists(self.save_directory):
            raise NotADirectoryError
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError

    def get_painting_ids(self) -> list[str]:
        try:
            painting_ids = []
            with open(self.csv_path, encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row["Classification"] == "Paintings":
                        painting_ids.append(row["Object ID"])

            return painting_ids
        except FileNotFoundError:
            return []

    def clear_save_directory(self) -> None:
        for file_name in os.listdir(self.save_directory):
            file_path = os.path.join(self.save_directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def save_artwork_info(self, artwork: Artwork) -> ArtworkPath:
        image_id = artwork.metadata["objectID"]
        image_path = os.path.join(self.save_directory, f"{image_id}.jpg")
        metadata_path = os.path.join(self.save_directory, f"{image_id}.json")

        with open(metadata_path, 'w') as f:
            json.dump(artwork.metadata, f)
        img = Image.fromarray(artwork.nparray)
        img.save(image_path)

        return ArtworkPath(image_path, metadata_path)

    def save_numpy_image(self, ndarray: NDArray[np.uint8], name: str) -> str:
        path = os.path.join(self.save_directory, f"{name}.jpg")
        img = Image.fromarray(ndarray)
        img.save(path)
        return path