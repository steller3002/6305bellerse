import asyncio
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
import aiohttp

from metetl.logging_config import setup_logging
from metetl.cli import build_parser

# Импорты бизнес-логики напрямую (чтобы не пробрасывать через cli)
from metetl.analysis.data_to_download import prepare_to_download
from metetl.analysis.aggregations import run_analysis
from metetl.images.processing import ApiProvider, DataProvider, ImageProcessor
from metetl.decorators import measure_time

setup_logging()

logger = logging.getLogger(__name__)


@measure_time
async def _run_process_pipeline(input_path: str, output_dir: str, num: int):
    storage = DataProvider(input_path, output_dir)
    api = ApiProvider(storage)

    with ProcessPoolExecutor() as pool:
        async with aiohttp.ClientSession() as session:
            tasks = [
                ImageProcessor(api, storage, i, session, pool).process_pipeline()
                for i in range(1, num + 1)
            ]
            await asyncio.gather(*tasks)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger.info("=== metetl запущен: команда '%s' ===", args.command)

    try:
        if args.command == "prepare":
            prepare_to_download(args.csv, args.output)
            logger.info("Готово: %s", args.output)

        elif args.command == "process":
            asyncio.run(_run_process_pipeline(args.input, args.output, args.num))
            logger.info("Пайплайн скачивания завершён. Результаты в: %s", args.output)

        elif args.command == "analyze":
            run_analysis(args.csv, args.output_dir)
            logger.info("Анализ завершён. Графики сохранены в: %s", args.output_dir)

    except Exception as exc:
        logger.error("Критическая ошибка при выполнении команды '%s': %s", args.command, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()