import re
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import hydra
from omegaconf import DictConfig
import fire
from pathlib import Path
import pickle

from ai_generated_text_detection.logger import logger

# Глобальные переменные для словаря
vocab = None
inverse_vocab = None
tokenizer = WordPunctTokenizer()
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(texts: list, vocab_size: int = 5000) -> dict:
    """Создает словарь из текстов"""
    global vocab, inverse_vocab

    word_counts = Counter()

    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())
        word_counts.update(words)

    common_words = word_counts.most_common(vocab_size - 2)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for idx, (word, _) in enumerate(common_words, start=2):
        vocab[word] = idx

    inverse_vocab = {v: k for k, v in vocab.items()}

    logger.info(f"Создан словарь размером {len(vocab)} токенов")
    return vocab


def get_words(text: str) -> list:
    """Токенизация текста"""
    text = text.lower()
    words = tokenizer.tokenize(text)
    return words


def text_to_numbers(text: str, max_len: int = 200) -> list:
    """Преобразует текст в последовательность чисел"""
    global vocab

    words = get_words(text)
    numbers = []

    for word in words:
        numbers.append(vocab.get(word, vocab[UNK_TOKEN]))

    # Обрезка/дополнение до max_len
    if len(numbers) > max_len:
        numbers = numbers[:max_len]
    else:
        numbers = numbers + [vocab[PAD_TOKEN]] * (max_len - len(numbers))

    return numbers


def prepare_data(data: pd.DataFrame, max_len: int = 200) -> tuple:
    """Подготавливает данные для обучения"""
    global vocab

    if vocab is None:
        build_vocab(data["text"].tolist())

    texts = data["text"].tolist()
    labels = data["generated"].tolist()

    sequences = []
    for text in texts:
        seq = text_to_numbers(text, max_len)
        sequences.append(seq)

    X = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    return X, y


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def preprocess_data(cfg: DictConfig) -> tuple:
    """
    Основная функция препроцессинга

    Returns:
        train_loader, val_loader, test_loader, vocab, inverse_vocab
    """
    logger.info("============ ПРЕПРОЦЕССИНГ ДАННЫХ ============")

    # 1. Загружаем данные
    logger.info("Загрузка данных...")

    train_path = Path(cfg.data.train_path)
    val_path = Path(cfg.data.val_path)
    test_path = Path(cfg.data.infer_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Train файл не найден: {train_path}")

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    logger.info(f"  Train: {len(train_data)} записей")
    logger.info(f"  Val: {len(val_data)} записей")
    logger.info(f"  Test: {len(test_data)} записей")

    # 2. Строим словарь на train данных
    logger.info(f"\n Построение словаря (размер: {cfg.data.vocab_size})...")
    build_vocab(train_data["text"].tolist(), cfg.data.vocab_size)

    # 3. Подготавливаем данные
    logger.info(f"Подготовка данных (max_len: {cfg.in_features})...")

    X_train, y_train = prepare_data(train_data, cfg.in_features)
    X_val, y_val = prepare_data(val_data, cfg.in_features)
    X_test, y_test = prepare_data(test_data, cfg.in_features)

    logger.info("  Размерности:")
    logger.info(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"    X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"    X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 4. Создаем даталоадеры
    logger.info("Создание DataLoader...")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    use_pin_memory = cfg.device in ["cuda", "mps"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory,
    )

    logger.info("DataLoader созданы:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")

    preprocessed_dir = cfg.data.dataloaders_dir
    preprocessed_filename = cfg.data.dataloaders_filename

    preprocessed_dir = Path(preprocessed_dir)
    preprocessed_filename = Path(preprocessed_filename)
    preprocessed_path = preprocessed_dir / preprocessed_filename
    preprocessed_dir.mkdir(exist_ok=True)

    # Сохраняем загрузчики
    with open(preprocessed_path, "wb") as f:
        pickle.dump(
            {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
                "vocab": vocab,
                "inverse_vocab": inverse_vocab,
                "config": cfg,  # сохраняем конфиг для воспроизводимости
            },
            f,
        )

    logger.info(f"\n Препроцессированные данные сохранены: {preprocessed_path}")


def preprocess_data_with_config():
    """
    Функция для вызова из других модулей
    """
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
        preprocess_data(cfg)


if __name__ == "__main__":
    fire.Fire(preprocess_data_with_config)
