# Detection of AI generated text

## Setup
### Requirements
- Python 3.11+ (3.11.13)
- UV 

### Шаги установки
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
cd HSE_MLOps_Project
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
uv run python main.py <команда>
```

## Основные команды
### Load data
```bash
uv run main.py load_data
```

### Preprocessing
```bash
uv run main.py preprocess
```

### Train
```bash
uv run main.py train
```


### Test
```
uv run main.py test
```

