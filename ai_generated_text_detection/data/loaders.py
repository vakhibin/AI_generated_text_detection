"""Модуль для загрузки препроцессированных данных."""

import pickle
from pathlib import Path
import torch
from ai_generated_text_detection.logger import logger


def load_preprocessed_data(
        dataloaders_dir: str = "preprocessed_data",
        dataloaders_filename: str = "dataloaders.pkl"
) -> tuple:
    """
    Загружает препроцессированные данные из файла pickle.

    Args:
        dataloaders_dir: Директория с файлом даталоадеров
        dataloaders_filename: Имя файла с даталоадерами

    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab, inverse_vocab)

    Raises:
        FileNotFoundError: Если файл с данными не найден
        Exception: Если произошла ошибка при загрузке
    """
    data_dir = Path(dataloaders_dir)
    data_path = data_dir / dataloaders_filename

    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл с препроцессированными данными не найден: {data_path}\n"
            f"Сначала выполните препроцессинг: python data/preprocessing.py"
        )

    try:
        logger.info(f"Загрузка препроцессированных данных из {data_path}")

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # Извлекаем компоненты
        train_loader = data["train_loader"]
        val_loader = data["val_loader"]
        test_loader = data["test_loader"]
        vocab = data["vocab"]
        inverse_vocab = data["inverse_vocab"]

        logger.info(f"Данные успешно загружены:")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
        logger.info(f"   Размер словаря: {len(vocab)}")

        return train_loader, val_loader, test_loader, vocab, inverse_vocab

    except Exception as e:
        raise Exception(f"Ошибка при загрузке препроцессированных данных из {data_path}: {e}")


def get_vocab() -> dict:
    """
    Возвращает словарь из сохраненных данных.

    Returns:
        Словарь {слово: индекс}

    Raises:
        FileNotFoundError: Если файл с данными не найден
    """
    train_loader, val_loader, test_loader, vocab, inverse_vocab = load_preprocessed_data()
    return vocab


def get_vocab_size() -> int:
    """
    Возвращает размер словаря.

    Returns:
        Размер словаря

    Raises:
        FileNotFoundError: Если файл с данными не найден
    """
    vocab = get_vocab()
    return len(vocab)