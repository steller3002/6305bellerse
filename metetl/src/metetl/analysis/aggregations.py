import os
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logging import getLogger

logger = getLogger(__name__)


def read_chunks(file_path: str, chunk_size: int = 5000) -> Generator[pd.DataFrame, None, None]:
    cols = ['Medium', 'Object Begin Date', 'Object End Date']
    logger.debug("Чтение CSV чанками по %d строк: %s", chunk_size, file_path)
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=cols, low_memory=False):
        yield chunk


def clean_and_calc_chunks(chunks: Generator[pd.DataFrame, None, None]) -> Generator[pd.DataFrame, None, None]:
    for chunk in chunks:
        chunk['Object Begin Date'] = pd.to_numeric(chunk['Object Begin Date'], errors='coerce')
        chunk['Object End Date']   = pd.to_numeric(chunk['Object End Date'],   errors='coerce')
        chunk = chunk.dropna(subset=['Medium', 'Object Begin Date', 'Object End Date'])

        chunk['Duration'] = chunk['Object End Date'] - chunk['Object Begin Date']
        chunk = chunk[chunk['Duration'] >= 0].copy()

        yield chunk


def local_aggregate_chunks(
    chunks: Generator[pd.DataFrame, None, None]
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    for chunk in chunks:
        chunk['Duration_sq'] = chunk['Duration'] ** 2

        stats_chunk = chunk.groupby('Medium').agg(
            count=('Duration', 'count'),
            sum=('Duration', 'sum'),
            sum_sq=('Duration_sq', 'sum')
        )

        timeline_chunk = chunk.groupby(['Medium', 'Object Begin Date']).agg(
            count=('Duration', 'count'),
            sum=('Duration', 'sum')
        )

        yield stats_chunk, timeline_chunk


def collect_global_stats(
    local_agg_gen: Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_stats = None
    df_timeline = None

    for stats_chunk, timeline_chunk in local_agg_gen:
        if df_stats is None:
            df_stats = stats_chunk
            df_timeline = timeline_chunk
        else:
            df_stats    = df_stats.add(stats_chunk, fill_value=0)
            df_timeline = df_timeline.add(timeline_chunk, fill_value=0)

    logger.debug("Глобальная агрегация завершена: %d уникальных материалов", len(df_stats))
    return df_stats, df_timeline


def plot_bar_with_intervals(df_top: pd.DataFrame, output_dir: str) -> None:
    x = np.arange(len(df_top))

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.errorbar(
        x, df_top['mean'], yerr=df_top['scatter_95'],
        fmt='none', ecolor='lightblue', capsize=0,
        elinewidth=8, label='95% интервал рассеяния',
    )

    ax.bar(
        x, df_top['mean'], yerr=df_top['ci_95'],
        capsize=5, ecolor='black', color='steelblue',
        label='Среднее и 95% доверительный интервал',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_top.index, rotation=45, ha='right', fontsize=9)
    ax.set_title("Топ-10 материалов: средняя длительность создания объекта")
    ax.set_ylabel("Лет")
    ax.legend()
    ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(output_dir, "task1_bar_intervals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("График задание 1 сохранён: %s", path)


def plot_timeline_for_best_medium(
    df_timeline: pd.DataFrame, medium: str, output_dir: str
) -> None:
    timeline = df_timeline.xs(medium, level='Medium').copy()
    timeline['mean_duration'] = timeline['sum'] / timeline['count']
    timeline = timeline.sort_index()
    timeline['rolling_mean'] = timeline['mean_duration'].rolling(window=5, min_periods=1).mean()

    logger.debug("Timeline для '%s': %d точек", medium, len(timeline))

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        timeline.index, timeline['mean_duration'],
        color='steelblue', alpha=0.5, linewidth=1,
        label='Средняя длительность по году',
    )
    ax.plot(
        timeline.index, timeline['rolling_mean'],
        color='crimson', linewidth=2,
        label='Скользящее среднее (окно 5)',
    )

    ax.set_xlabel("Год начала создания объекта")
    ax.set_ylabel("Средняя длительность (лет)")
    ax.set_title(f"Динамика длительности создания объектов\nМатериал: {medium[:70]}")
    ax.legend()
    ax.grid(alpha=0.4)

    plt.tight_layout()
    safe_name = "".join(c if c.isalnum() else "_" for c in medium)[:40]
    path = os.path.join(output_dir, f"task2_timeline_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("График задание 2 сохранён: %s", path)


def run_analysis(csv_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Запуск анализа: %s → %s", csv_path, output_dir)

    raw_chunks       = read_chunks(csv_path, chunk_size=5000)
    processed_chunks = clean_and_calc_chunks(raw_chunks)
    local_gen        = local_aggregate_chunks(processed_chunks)
    df_stats, df_timeline = collect_global_stats(local_gen)

    df_top = df_stats.nlargest(10, 'count').copy()
    df_top['mean']       = df_top['sum'] / df_top['count']
    variance             = (df_top['sum_sq'] / df_top['count']) - (df_top['mean'] ** 2)
    df_top['std']        = np.sqrt(variance.clip(lower=0))
    df_top['ci_95']      = 1.96 * (df_top['std'] / np.sqrt(df_top['count']))
    df_top['scatter_95'] = 1.96 * df_top['std']

    logger.info("Топ-10 материалов по числу объектов:\n%s", df_top[['count', 'mean', 'ci_95']].to_string())

    valid = df_stats[df_stats['count'] > 50].copy()
    valid['mean'] = valid['sum'] / valid['count']
    best_medium = valid['mean'].idxmax()
    logger.info("Материал с наибольшим средним сроком: %s (%.1f лет)", best_medium, valid.loc[best_medium, 'mean'])

    plot_bar_with_intervals(df_top, output_dir)
    plot_timeline_for_best_medium(df_timeline, best_medium, output_dir)

    logger.info("Анализ завершён. Результаты в: %s", output_dir)