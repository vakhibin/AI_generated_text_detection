"""Модуль для запуска процесса обучения модели."""

from pathlib import Path

import fire
import lightning as L
import torch
from hydra import compose, initialize
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

from ai_generated_text_detection.data.loaders import load_preprocessed_data
from ai_generated_text_detection.models.load_models import create_model_and_module
from ai_generated_text_detection.logger import logger


def train() -> None:
    """
    Запускает процесс обучения модели.
    """
    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name="config")

        logger.info("========== ЗАПУСК ОБУЧЕНИЯ ==========")

        # 1. Загрузка данных (препроцессинг уже должен быть выполнен)
        logger.info("Загрузка данных...")
        try:
            # Загружаем данные через универсальную функцию
            train_loader, val_loader, _, vocab, _ = load_preprocessed_data(
                dataloaders_dir=cfg.data.dataloaders_dir,
                dataloaders_filename=cfg.data.dataloaders_filename,
            )

            logger.info(f"  Train batches: {len(train_loader)}")
            logger.info(f"  Val batches: {len(val_loader)}")
            logger.info(f"  Размер словаря: {len(vocab)}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            logger.error("  Убедитесь, что сначала выполнен препроцессинг:")
            logger.error("  python data/preprocessing.py")
            return

        # 2. Создание модели и Lightning модуля
        logger.info(f"Создание модели ({cfg.model.type})...")

        try:
            # Используем фабричную функцию для создания модели и модуля
            model, model_module = create_model_and_module(cfg)
            logger.info(f"Модель создана: {cfg.model.type}")
            logger.info("Lightning модуль создан")

        except Exception as e:
            logger.error(f"шибка при создании модели: {e}")
            logger.error("Проверьте конфигурацию модели в configs/model/")
            return

        # 3. Настройка логирования
        logger.info("Настройка логирования...")

        loggers = []

        if cfg.logging.tensorboard:
            tb_logger = TensorBoardLogger(
                save_dir=Path(cfg.outputs_dir) / "tensorboard_logs",
                name=cfg.model.type,
                log_graph=True,
            )
            loggers.append(tb_logger)
            logger.info(f"TensorBoard: {tb_logger.log_dir}")

        if cfg.logging.mlflow and cfg.logging.tracking_uri:
            mlflow_logger = MLFlowLogger(
                tracking_uri=cfg.logging.tracking_uri,
                experiment_name=cfg.logging.experiment_name,
            )
            loggers.append(mlflow_logger)
            logger.info(
                f"MLflow: {cfg.logging.tracking_uri}/{cfg.logging.experiment_name}"
            )

        # 4. Callbacks
        logger.info("Настройка callbacks...")

        callbacks = []

        # Checkpoint callback
        checkpoint_dir = Path(cfg.outputs_dir) / "checkpoints"
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename=f"{cfg.model.type}-best-checkpoint",
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=cfg.training_params.get("early_stopping_patience", 5),
            mode="min",
        )
        callbacks.append(early_stopping)

        logger.info(f"Checkpoints: {checkpoint_dir}")
        logger.info(
            f"Early stopping (patience: {cfg.training_params.get('early_stopping_patience', 5)})"
        )

        # 5. Создание тренера
        logger.info("Настройка тренера...")

        # Определяем accelerator для Lightning
        if cfg.device == "cuda":
            accelerator = "gpu"
        elif cfg.device == "mps":
            accelerator = "mps"
        else:
            accelerator = "cpu"

        trainer = L.Trainer(
            max_epochs=cfg.training_params.num_epochs,
            accelerator=accelerator,
            devices=1,
            callbacks=callbacks,
            logger=loggers if loggers else None,
            log_every_n_steps=10,
            enable_progress_bar=True,
            default_root_dir=cfg.outputs_dir,
        )

        logger.info(f"  Эпохи: {cfg.training_params.num_epochs}")
        logger.info(f"  Устройство: {cfg.device} (accelerator: {accelerator})")

        # 6. Обучение
        logger.info(" Запуск обучения...")

        trainer.fit(
            model=model_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # 7. Сохранение модели
        logger.info("Сохранение модели...")

        outputs_path = Path(cfg.state_dict_file)
        outputs_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем лучшую модель
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            logger.info(f"Лучшая модель сохранена: {best_model_path}")

        # Сохраняем веса модели
        torch.save(obj=model_module.model.state_dict(), f=cfg.state_dict_file)
        logger.info(f"Веса модели сохранены: {cfg.state_dict_file}")

        # 8. Логирование гиперпараметров в MLflow
        if cfg.logging.mlflow and cfg.logging.tracking_uri:
            # Логируем все гиперпараметры модели
            model_params = {
                "model_type": cfg.model.type,
                "num_epochs": cfg.training_params.num_epochs,
                "learning_rate": cfg.training_params.lr,
                "vocab_size": len(vocab),  # Используем реальный размер словаря
                "batch_size": 32,
                "in_features": cfg.in_features,
                "num_classes": cfg.num_classes,
                "optimizer": cfg.training_params.get("optimizer", "adam"),
                "weight_decay": cfg.training_params.get("weight_decay", 0.0),
            }

            # Добавляем специфичные параметры модели
            if hasattr(cfg.model, "params"):
                for key, value in cfg.model.params.items():
                    model_params[f"model_{key}"] = value

            mlflow_logger.log_hyperparams(model_params)

            # Логируем метрики
            if trainer.logged_metrics:
                mlflow_logger.log_metrics(trainer.logged_metrics)

        logger.info("Обучение завершено!")
        logger.info(f"   Модель: {cfg.model.type}")
        logger.info(f"   Файл весов: {cfg.state_dict_file}")
        logger.info(f"   Выходная директория: {cfg.outputs_dir}")

        # Выводим путь к лучшей модели
        if best_model_path:
            logger.info(f"   Лучшая модель: {best_model_path}")

        # 9. Версионирование модели в DVC
        logger.info("Добавление обученной модели в DVC...")
        import subprocess

        subprocess.run(["dvc", "add", cfg.state_dict_file], capture_output=True)
        if best_model_path:
            subprocess.run(["dvc", "add", best_model_path], capture_output=True)


if __name__ == "__main__":
    fire.Fire(component=train)
