# Detection of AI generated text

## Описание проекта

### Детекция сгенерированного LLM текста.

В настоящее время множество студентов и работников используют Large Language Models (LLM) для генерации различных текстов. Возникает необходимость в эффективном различении текстов, сгенерированных искусственным интеллектом, и написанных вручную. Это особенно важно в образовательном контексте для проверки самостоятельности выполнения студенческих работ.

Данный проект направлен на разработку системы детекции AI-сгенерированного текста с использованием методов машинного обучения.

## Формат входных и выходных данных

Данные взяты отсюда: https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset

**Структура данных:**

- `text` - тексты эссе
- `generated` - бинарная переменная, где $0$ - написанное человеком эссе, $1$ - эссе, сгенерировнные AI.
  **Размерность входа:** `(2, n)`, где `n` - количество строк в датасете

Всего строк: $27340$.

## Метрики качества

Задача представляет собой бинарную классификацию. Используемые метрики:

- **ROC-AUC** (основная метрика)
- **Precision, Recall, F1-score**
- **Accuracy**

**Целевые значения:** Ожидается достижение ROC-AUC в диапазоне 0.85-0.9, аналогичные значения для F1-score и Accuracy.

Для оценки модели используется разделение данных на обучающую, валидационную и тестовую выборки с помощью `train_test_split`. Тестовые данные также могут быть взяты из отдельного тестового датасета, доступного в соревновании Kaggle.

## Setup

### Requirements

- Python 3.11+ (3.11.13)
- UV

### Шаги установки (Linux)

1. **Установить UV**:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
pip install uv
```

2. Клонирование репозитория

```bash
git clone <your-repo-url>
cd AI_generated_text_detection
```

3. Создаем виртуальное окружение и устанавливаем зависимости

```bash
uv venv
uv sync
```

4. Актвивируем окружение:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

5. Установка git hooks

```bash
pre-commit install
```

6. Запускать команды лучше через окружение uv

```bash
uv run python3 main.py <команда>
```

### Дополнительно

- Для запуска логирования с MLflow нужно поменять соответствующие настройки в Hydra, а также запустить mlflow сервер:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
uv run tensorboard --logdir <папка с логами>
```

## Train

Чтобы запустить обучение модели, нужно предварительно подготовить данные с помощью команд

1. `uv run python3 main.py download_data`
2. `uv run python3 main.py preprocess`

После этоо нужно запустить команду:

```bash
uv run python3 main.py train
```

Для последующей валидацидации используется: `uv run python3 main.py test`

## Основные команды

### Load data

```bash
uv run python3 main.py download_data
```

### Preprocessing

```bash
uv run python3 main.py preprocess
```

### Train

```bash
uv run python3 main.py train
```

### Test

```bash
uv run python3 main.py test
```

### Clean

Удаляет временные файлы вроде логов, обученных моделей и т.п. Оставляет скаченные данные.

```bash
uv run python3 main.py clean
```

### Status

Показывает status проекта.

```bash
uv run python3 main.py status
```

### All

Запускает весь pipeline

```bash
uv run python3 main.py all
```
