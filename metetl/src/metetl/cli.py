import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metetl",
        description="Сбор и обработка данных о картинах из Metropolitan Museum of Art.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<команда>")

    p_prepare = subparsers.add_parser(
        "prepare",
        help="Подготовить to_download.json из CSV",
        description="Читает MetObjects.csv, отбирает картины и сохраняет метаданные в JSON.",
    )
    p_prepare.add_argument("--csv", required=True, metavar="PATH", help="Путь к MetObjects.csv")
    p_prepare.add_argument("--output", default="data/to_download.json", metavar="PATH", help="Путь к выходному JSON")

    p_process = subparsers.add_parser(
        "process",
        help="Скачать и обработать изображения",
        description="Асинхронно скачивает случайные картины из API МЕТ и применяет фильтры.",
    )
    p_process.add_argument("--input", required=True, metavar="PATH", help="Путь к CSV или JSON с данными")
    p_process.add_argument("--output", default="images", metavar="DIR", help="Директория для сохранения изображений")
    p_process.add_argument("--num", type=int, required=True, metavar="N", help="Количество изображений для обработки")

    p_analyze = subparsers.add_parser(
        "analyze",
        help="Анализ датасета из CSV",
        description="Строит графики по датасету и сохраняет их.",
    )
    p_analyze.add_argument("--csv", required=True, metavar="PATH", help="Путь к MetObjects.csv")
    p_analyze.add_argument("--output-dir", default="data/plots", metavar="DIR",
                           help="Директория для сохранения графиков")

    return parser