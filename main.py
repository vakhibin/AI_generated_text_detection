import fire
from ai_generated_text_detection.logger import logger


def download_data() -> None:
    """
    Загружает и разделяет данные на train/val/test.

    Скачивает датасет с Kaggle, разделяет на обучающую, валидационную
    и тестовую выборки, сохраняет в директорию data/.

    Пример:
        python main.py download_data
    """
    from ai_generated_text_detection.data.load_data import load_data_with_config

    load_data_with_config()


def preprocess() -> None:
    """
    Выполняет препроцессинг текстовых данных.

    Создает словарь, токенизирует тексты, преобразует в тензоры,
    создает DataLoader'ы и сохраняет их для последующего использования.

    Пример:
        python main.py preprocess
    """
    import ai_generated_text_detection.data.processing as proc_module

    proc_module.preprocess_data_with_config()


def train() -> None:
    """
    Обучает модель на подготовленных данных.

    Загружает препроцессированные данные, создает модель,
    настраивает обучение с логированием и сохраняет веса.

    Пример:
        python main.py train
        python main.py train --model=lstm_classifier
    """
    from ai_generated_text_detection.train import train as train_func

    train_func()


def test() -> dict:
    """
    Тестирует обученную модель на тестовых данных.

    Загружает модель и тестовые данные, вычисляет метрики,
    сохраняет результаты и создает submission файл.

    Returns:
        dict: Словарь с метриками тестирования

    Пример:
        python main.py test
    """
    from ai_generated_text_detection.test import test as test_func

    return test_func()


def all() -> None:
    """
    Запускает полный пайплайн от загрузки данных до тестирования.

    Последовательно выполняет:
    1. download_data - загрузка и разделение данных
    2. preprocess - препроцессинг текстов
    3. train - обучение модели
    4. test - тестирование модели

    Пример:
        python main.py all
    """
    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА")
    logger.info("=" * 60)

    # 1. Загрузка данных
    logger.info("\n1️⃣  ЗАГРУЗКА И РАЗДЕЛЕНИЕ ДАННЫХ")
    logger.info("-" * 40)
    download_data()

    # 2. Препроцессинг
    logger.info("\n2️⃣  ПРЕПРОЦЕССИНГ ТЕКСТОВ")
    logger.info("-" * 40)
    preprocess()

    # 3. Обучение
    logger.info("\n3️⃣  ОБУЧЕНИЕ МОДЕЛИ")
    logger.info("-" * 40)
    train()

    # 4. Тестирование
    logger.info("\n4️⃣  ТЕСТИРОВАНИЕ МОДЕЛИ")
    logger.info("-" * 40)
    test_metrics = test()

    logger.info("\n" + "=" * 60)
    logger.info("✅ ПОЛНЫЙ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН")
    logger.info("=" * 60)

    if test_metrics:
        logger.info("\n📊 ИТОГОВЫЕ МЕТРИКИ:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric:12}: {value:.4f}")


def status() -> None:
    """
    Показывает статус проекта: какие файлы созданы, что можно запустить.

    Проверяет наличие необходимых файлов и директорий,
    дает рекомендации по дальнейшим действиям.

    Пример:
        python main.py status
    """
    from pathlib import Path

    logger.info("=" * 60)
    logger.info("📊 СТАТУС ПРОЕКТА")
    logger.info("=" * 60)

    # Проверяем существование ключевых файлов и директорий
    checks = [
        ("data/train.csv", "Файл с тренировочными данными"),
        ("data/val.csv", "Файл с валидационными данными"),
        ("data/test.csv", "Файл с тестовыми данными"),
        ("preprocessed_data/dataloaders.pkl", "Препроцессированные данные"),
        ("outputs/", "Директория с результатами"),
        ("configs/", "Директория с конфигурациями"),
    ]

    all_ok = True
    for path_str, description in checks:
        path = Path(path_str)
        exists = path.exists()
        status = "✅" if exists else "❌"

        if exists and path.is_dir():
            # Для директорий показываем количество файлов
            file_count = len(list(path.glob("*")))
            logger.info(f"{status} {description:40} {path_str} ({file_count} файлов)")
        else:
            logger.info(f"{status} {description:40} {path_str}")

        if not exists:
            all_ok = False

    logger.info("\n" + "=" * 60)
    logger.info("🎯 РЕКОМЕНДАЦИИ:")
    logger.info("=" * 60)

    if not Path("data/train.csv").exists():
        logger.info("1. Загрузите данные:    python main.py download_data")

    if not Path("preprocessed_data/dataloaders.pkl").exists():
        logger.info("2. Выполните препроцессинг: python main.py preprocess")

    if not Path("outputs/").exists() or len(list(Path("outputs/").glob("*.pth"))) == 0:
        logger.info("3. Обучите модель:      python main.py train")

    if all_ok:
        logger.info("✅ Все этапы выполнены. Можно запустить тестирование:")
        logger.info("   python main.py test")
        logger.info("\n🔥 Или запустить полный пайплайн заново:")
        logger.info("   python main.py all")


def clean() -> None:
    """
    Очищает временные файлы и директории.

    Удаляет:
    - Препроцессированные данные (preprocessed_data/)
    - Результаты обучения (outputs/)
    Сохраняет исходные данные (data/).

    Пример:
        python main.py clean
    """
    import shutil
    from pathlib import Path

    logger.info("🧹 ОЧИСТКА ВРЕМЕННЫХ ФАЙЛОВ")
    logger.info("-" * 40)

    dirs_to_clean = ["preprocessed_data", "outputs"]

    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info(f"✅ Удалено: {dir_name}/")
        else:
            logger.info(f"ℹ️  Не найдено: {dir_name}/")

    logger.info("\n✅ Очистка завершена!")
    logger.info("📁 Сохранены: data/, configs/")


if __name__ == "__main__":
    fire.Fire(
        {
            "download_data": download_data,
            "preprocess": preprocess,
            "train": train,
            "test": test,
            "all": all,
            "status": status,
            "clean": clean,
        }
    )
