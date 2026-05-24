import json
from typing import List

import pandas as pd

from logging import getLogger

logger = getLogger(__name__)


def filter_paintings(csv_path: str) -> pd.DataFrame:
    logger.info("Чтение CSV: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    logger.debug("Загружено %d строк из CSV", len(df))

    paintings = df[df["Classification"] == "Paintings"].copy()
    logger.info("Найдено картин: %d", len(paintings))
    return paintings


def load_to_download_json(json_path: str) -> List[dict]:
    logger.debug("Чтение to_download.json: %s", json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Загружено %d записей из to_download.json", len(data))
    return data


def prepare_to_download(csv_path: str, output_path: str) -> None:
    """
    Читает CSV, отбирает картины и сохраняет метаданные в JSON.

    Parameters
    ----------
    csv_path : str
        Путь к MetObjects.csv.
    output_path : str
        Путь к выходному JSON-файлу.
    """
    paintings = filter_paintings(csv_path)

    records = paintings[["Object ID", "Title", "Artist Display Name", "Object Date"]].rename(
        columns={
            "Object ID": "object_id",
            "Title": "title",
            "Artist Display Name": "artist",
            "Object Date": "date",
        }
    )

    data = records.fillna("").to_dict(orient="records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Метаданные %d картин сохранены в %s", len(data), output_path)