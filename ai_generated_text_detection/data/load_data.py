import fire
import pandas as pd
import shutil
import kagglehub
from sklearn.model_selection import train_test_split
from pathlib import Path
from omegaconf import DictConfig
import hydra

from ai_generated_text_detection.logger import logger


class LoadingDataException(Exception):
    pass


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def load_and_split_data(cfg: DictConfig) -> None:
    """
    Скачивает данные, разделяет на train/val/test и сохраняет в директории

    Args:
        cfg: Конфигурация Hydra с параметрами из data.yaml
    """
    try:
        # 1. Скачивание данных
        # sunilthite/llm-detect-ai-generated-text-dataset
        logger.info("Скачивание датасета...")
        download_path_str = kagglehub.dataset_download(
            handle="sunilthite/llm-detect-ai-generated-text-dataset"
        )
        download_path = Path(download_path_str)

        # 2. Поиск CSV файлов
        csv_files = list(download_path.glob("*.csv"))

        if not csv_files:
            csv_files = list(download_path.rglob("*.csv"))

        if not csv_files:
            raise LoadingDataException("Не найдены CSV файлы в датасете")

        first_csv = csv_files[0]
        logger.info(f"Чтение файла: {first_csv.name}")

        # 3. Чтение данных
        full_data = pd.read_csv(first_csv)

        # 4. Разделение данных
        logger.info(
            f"Разделение данных: train={cfg.data.train_size}, "
            f"val={cfg.data.val_size}, test={cfg.data.test_size}"
        )

        stratify_col = None
        if "generated" in full_data.columns:
            stratify_col = full_data["generated"]
            logger.info("Стратификация по столбцу 'generated'")
        else:
            logger.info("Столбец 'generated' не найден, разделение без стратификации")

        train_val_data, test_data = train_test_split(
            full_data,
            test_size=cfg.data.test_size,
            random_state=cfg.random_state,
            stratify=stratify_col,
        )

        train_size_adjusted = cfg.data.train_size / (
            cfg.data.train_size + cfg.data.val_size
        )

        stratify_train_val = None
        if stratify_col is not None:
            # Получаем соответствующие метки для train_val_data
            stratify_train_val = train_val_data["generated"]

        train_data, val_data = train_test_split(
            train_val_data,
            train_size=train_size_adjusted,
            random_state=cfg.random_state,
            stratify=stratify_train_val,
        )

        # 5. Создание директорий и сохранение данных
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Преобразуем пути из конфигурации в Path объекты
        train_path = Path(cfg.data.train_path)
        val_path = Path(cfg.data.val_path)
        infer_path = Path(cfg.data.infer_path)

        # Сохраняем данные
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(infer_path, index=False)

        logger.info("Данные сохранены:")
        logger.info(f"  Train: {train_path.absolute()} ({len(train_data)} записей)")
        logger.info(f"  Val: {val_path.absolute()} ({len(val_data)} записей)")
        logger.info(f"  Test: {infer_path.absolute()} ({len(test_data)} записей)")

        # 6. Проверка сохранения файлов
        logger.info("\nПроверка сохраненных файлов:")
        for path in [train_path, val_path, infer_path]:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✓ {path.name}: {size_mb:.2f} MB")
            else:
                logger.info(f"  ✗ {path.name}: файл не найден")

        # 7. ДОБАВЛЕННЫЙ КОД: Версионирование в DVC
        logger.info("Добавление данных в DVC для версионирования...")
        import subprocess
        for csv_file in [train_path, val_path, infer_path]:
            subprocess.run(["dvc", "add", str(csv_file)], capture_output=True)

        # 8. Очистка временных файлов
        logger.info(f"\nОчистка временных файлов из {download_path}")
        shutil.rmtree(download_path)

    except Exception as e:
        raise LoadingDataException(f"Ошибка при загрузке данных: {e}")

def load_data_with_config():
    """
    Функция для вызова из других модулей
    """
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
        load_and_split_data(cfg)


if __name__ == "__main__":
    fire.Fire(component=load_data_with_config)
