import json
import logging
import logging.config
import os

def setup_logging():
    if not os.path.exists("logs"):
        os.makedirs("logs")

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_config.json")

    if os.path.exists(config_path):
        with open(config_path, "rt", encoding="utf-8") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning("Файл конфигурации %s не найден. Использован базовый конфиг.", config_path)