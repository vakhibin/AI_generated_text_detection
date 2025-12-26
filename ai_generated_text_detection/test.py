"""Модуль для тестирования модели."""

from pathlib import Path

import fire
import torch
import torchmetrics
from hydra import compose, initialize

from data.loaders import load_preprocessed_data
from models.load_models import create_model

from ai_generated_text_detection.logger import logger


def test() -> dict:
    """
    Запускает тестирование модели на тестовых данных.

    Returns
    -------
    dict
        Словарь с метриками тестирования.
    """
    with initialize(
        config_path="./ai_generated_text_detection/configs", version_base=None
    ):
        cfg = compose(config_name="config")

        logger.info("=" * 50)
        logger.info("ТЕСТИРОВАНИЕ МОДЕЛИ")
        logger.info("=" * 50)

        # 1. Загрузка данных
        logger.info("Загрузка тестовых данных...")

        try:
            _, _, test_loader, vocab, _ = load_preprocessed_data(
                dataloaders_dir=cfg.data.dataloaders_dir,
                dataloaders_filename=cfg.data.dataloaders_filename,
            )
            logger.info(f"  Test batches: {len(test_loader)}")
            logger.info(f"  Размер словаря: {len(vocab)}")

        except Exception as e:
            logger.info(f"Ошибка при загрузке данных: {e}")
            logger.info("  Убедитесь, что сначала выполнен препроцессинг:")
            logger.info("  python data/preprocessing.py")
            return {}

        # 2. Загрузка модели
        logger.info(f"Загрузка модели ({cfg.model.type})...")

        try:
            # Проверяем существование файла весов
            model_path = Path(cfg.state_dict_file)
            if not model_path.exists():
                logger.info(f"Файл весов не найден: {model_path}")
                logger.info("Убедитесь, что модель была обучена:")
                logger.info("python train.py")
                return {}

            # Создаем модель через фабричную функцию
            model_kwargs = {
                "vocab_size": len(vocab),  # Используем реальный размер словаря
            }

            # Добавляем специфичные параметры модели
            if hasattr(cfg.model, "params"):
                model_kwargs.update(cfg.model.params)

            # Для LSTM модели
            if cfg.model.type == "lstm":
                model_kwargs.update(
                    {
                        "embedding_dim": cfg.model.get("embedding_dim", 256),
                        "hidden_dim": cfg.model.get("hidden_dim", 128),
                        "num_layers": cfg.model.get("num_layers", 1),
                    }
                )
            # Для Transformer модели
            elif cfg.model.type == "transformer":
                model_kwargs.update(
                    {
                        "max_len": cfg.in_features,
                        "d_model": cfg.model.get("d_model", 128),
                        "nhead": cfg.model.get("nhead", 4),
                        "num_layers": cfg.model.get("num_layers", 3),
                        "dim_feedforward": cfg.model.get("dim_feedforward", 256),
                        "dropout": cfg.model.get("dropout", 0.1),
                    }
                )
            # Для Baseline модели
            elif cfg.model.type == "baseline":
                model_kwargs.update(
                    {
                        "in_features": cfg.in_features,
                        "num_classes": cfg.num_classes,
                    }
                )

            # Создаем модель
            model = create_model(cfg.model.type, **model_kwargs)

            # Загружаем веса
            model.load_state_dict(torch.load(model_path, map_location=cfg.device))
            model.to(cfg.device)
            model.eval()

            logger.info(f"  Модель загружена: {model_path}")
            logger.info(f"  Устройство: {cfg.device}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            return {}

        # 3. Инициализация метрик
        logger.info("Настройка метрик...")

        accuracy = torchmetrics.Accuracy(task="binary")
        precision = torchmetrics.Precision(task="binary")
        recall = torchmetrics.Recall(task="binary")
        f1 = torchmetrics.F1Score(task="binary")
        auc = torchmetrics.AUROC(task="binary")

        # 4. Тестирование
        logger.info("Выполнение тестирования...")

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(cfg.device)
                batch_y = batch_y.to(cfg.device)

                outputs = model(batch_X)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(batch_y.cpu())

        # Объединяем результаты
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)

        # Вычисляем метрики
        test_metrics = {}

        test_metrics["accuracy"] = accuracy(all_preds, all_labels).item()
        test_metrics["precision"] = precision(all_preds, all_labels).item()
        test_metrics["recall"] = recall(all_preds, all_labels).item()
        test_metrics["f1_score"] = f1(all_preds, all_labels).item()
        test_metrics["roc_auc"] = auc(all_probs, all_labels).item()

        # 5. Вывод результатов
        logger.info("=============== Результаты тестирования: ===============")
        for metric_name, value in test_metrics.items():
            logger.info(f"  {metric_name:12}: {value:.4f}")

        # 6. Сохранение результатов
        logger.info("Сохранение результатов...")

        results_dir = Path(cfg.outputs_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем метрики в файл
        results_file = results_dir / f"test_results_{cfg.model.type}.txt"
        with open(results_file, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Модель:           {cfg.model.type}\n")
            f.write(f"Датасет:          {Path(cfg.data.infer_path).name}\n")
            f.write(f"Входные признаки: {cfg.in_features}\n")
            f.write(f"Классы:           {cfg.num_classes}\n")
            f.write(f"Размер словаря:   {len(vocab)}\n")
            f.write(f"Файл весов:       {cfg.state_dict_file}\n")
            f.write(f"Время тестирования: {torch.tensor([]).device}\n\n")

            f.write("Метрики:\n")
            f.write("-" * 30 + "\n")
            for metric_name, value in test_metrics.items():
                f.write(f"{metric_name:12}: {value:.4f}\n")
            f.write("=" * 50 + "\n")

        logger.info(f"Результаты сохранены: {results_file}")

        # 7. Сохранение submission файла
        if hasattr(cfg, "submission_file") and cfg.submission_file:
            logger.info("\nСоздание submission файла...")
            try:
                import pandas as pd

                # Загружаем тестовые данные для submission
                test_data_path = Path(cfg.data.infer_path)
                if test_data_path.exists():
                    test_df = pd.read_csv(test_data_path)

                    # Создаем submission DataFrame
                    submission_df = pd.DataFrame(
                        {
                            "text_id": range(len(all_preds)),
                            "text": test_df["text"]
                            if "text" in test_df.columns
                            else test_df.iloc[:, 0],
                            "generated_pred": all_preds.numpy().flatten().astype(int),
                            "generated_prob": all_probs.numpy().flatten(),
                            "is_correct": (
                                all_preds.numpy().flatten()
                                == all_labels.numpy().flatten()
                            ).astype(int),
                        }
                    )

                    # Сохраняем submission
                    submission_path = Path(cfg.submission_file)
                    submission_path.parent.mkdir(parents=True, exist_ok=True)
                    submission_df.to_csv(submission_path, index=False)

                    logger.info(f"Submission файл сохранен: {submission_path}")
                    logger.info(f"Размер submission: {len(submission_df)} записей")

                    # Статистика предсказаний
                    pred_stats = submission_df["generated_pred"].value_counts()
                    logger.info("Распределение предсказаний:")
                    for label, count in pred_stats.items():
                        logger.info(
                            f"      Класс {int(label)}: {count} ({count / len(submission_df) * 100:.1f}%)"
                        )

            except Exception as e:
                logger.error(f"Ошибка при создании submission: {e}")

        logger.info("Тестирование завершено!")
        logger.info(f"Модель: {cfg.model.type}")
        logger.info(f"Метрики сохранены: {results_file}")

        if hasattr(cfg, "submission_file") and cfg.submission_file:
            logger.info(f"   Submission файл: {cfg.submission_file}")

        return test_metrics


if __name__ == "__main__":
    fire.Fire(component=test)
