from torch import nn
from ai_generated_text_detection.models.model_module import UniversalModelModule


def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Фабричная функция для создания моделей по имени.

    Parameters
    ----------
    model_name : str
        Название модели ("lstm", "transformer").
    **kwargs : dict
        Параметры модели.

    Returns
    -------
    nn.Module
        Экземпляр модели.

    Raises
    ------
    ValueError
        Если модель с таким именем не найдена.
    """
    model_name = model_name.lower()

    # Убираем _classifier если есть
    if model_name.endswith('_classifier'):
        model_name = model_name.replace('_classifier', '')

    if model_name == "lstm":
        from ai_generated_text_detection.models.lstm_classifier import EssayLSTMClassifier
        return EssayLSTMClassifier(**kwargs)

    elif model_name == "transformer":
        from ai_generated_text_detection.models.transformer_classifier import SimpleTransformerClassifier
        return SimpleTransformerClassifier(**kwargs)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}. "
                         f"Доступные модели: 'lstm', 'transformer'")


def create_model_module(model: nn.Module, cfg) -> UniversalModelModule:
    """
    Создает UniversalModelModule из модели и конфигурации Hydra.

    Parameters
    ----------
    model : nn.Module
        PyTorch модель.
    cfg : DictConfig
        Конфигурация Hydra.

    Returns
    -------
    UniversalModelModule
        Lightning модуль для обучения.
    """
    # Получаем чистый тип модели
    model_type = cfg.model.type
    if model_type.endswith('_classifier'):
        clean_model_type = model_type.replace('_classifier', '')
    else:
        clean_model_type = model_type

    return UniversalModelModule(
        model=model,
        model_type=clean_model_type,
        learning_rate=cfg.training_params.lr,
        weight_decay=cfg.training_params.get("weight_decay", 0.0),
        optimizer=cfg.training_params.get("optimizer", "adam"),
        scheduler=cfg.training_params.get("scheduler", None),
        scheduler_patience=cfg.training_params.get("scheduler_patience", 5),
        scheduler_factor=cfg.training_params.get("scheduler_factor", 0.1),
        log_interval=cfg.training_params.get("log_interval", 10)
    )


def create_model_and_module(cfg) -> tuple:
    """
    Создает модель и Lightning модуль из конфигурации Hydra.

    Parameters
    ----------
    cfg : DictConfig
        Конфигурация Hydra.

    Returns
    -------
    tuple
        (model, model_module) где model - PyTorch модель,
        model_module - UniversalModelModule.
    """
    # Получаем тип модели
    model_type = cfg.model.type

    # Получаем параметры из cfg.model.params
    params = cfg.model.params

    # Для LSTM
    if model_type == "lstm_classifier":
        from ai_generated_text_detection.models.lstm_classifier import LSTMClassifier
        model = LSTMClassifier(
            vocab_size=params.vocab_size,
            embedding_dim=params.embedding_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
        )
        clean_model_type = "lstm"

    # Для Transformer
    elif model_type == "transformer_classifier":
        from ai_generated_text_detection.models.transformer_classifier import SimpleTransformerClassifier
        model = SimpleTransformerClassifier(
            vocab_size=params.vocab_size,
            max_len=cfg.in_features,
            d_model=params.embedding_dim,
            nhead=params.num_heads,
            num_layers=params.num_layers,
            dim_feedforward=params.hidden_dim,
            dropout=params.dropout,
        )
        clean_model_type = "transformer"

    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    # Создаем Lightning модуль
    model_module = UniversalModelModule(
        model=model,
        model_type=clean_model_type,  # Используем чистый тип без _classifier
        learning_rate=cfg.training_params.lr,
        weight_decay=cfg.training_params.get("weight_decay", 0.0),
        optimizer=cfg.training_params.get("optimizer", "adam"),
        scheduler=cfg.training_params.get("scheduler", None),
        scheduler_patience=cfg.training_params.get("scheduler_patience", 5),
        scheduler_factor=cfg.training_params.get("scheduler_factor", 0.1),
        log_interval=cfg.training_params.get("log_interval", 10)
    )

    return model, model_module