# MLflow Models: Стандартизація та деплоймент моделей

Цей проект демонструє роботу з MLflow Models - стандартним форматом для пакування та деплойменту ML-моделей.

## 📚 Зміст

### Базові приклади
1. **01_basic_model_logging.ipynb** - Базове логування та завантаження моделей
2. **02_model_signatures.ipynb** - Сигнатури моделей та валідація входів
3. **03_model_flavors.ipynb** - Різні model flavors (sklearn, pyfunc)

### Розширені приклади
4. **04_custom_pyfunc_models.ipynb** - Створення кастомних PyFunc моделей
5. **05_model_deployment.ipynb** - Локальний деплоймент та serving

##  Навчальні цілі

Після проходження цих прикладів студенти зможуть:

-  Розуміти структуру MLflow Model
-  Логувати моделі з метаданими та залежностями
-  Використовувати model signatures для валідації
-  Працювати з різними model flavors
-  Створювати кастомні моделі
-  Деплоїти моделі локально та на сервери

##  Швидкий старт

### 1. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 2. Запуск Jupyter

```bash
cd notebooks
jupyter notebook
```

### 3. Відкрийте перший ноутбук

Почніть з `01_basic_model_logging.ipynb`

## Що таке MLflow Model?

**MLflow Model** - це стандартний формат для пакування ML-моделей, який:

- 📦 **Незалежний від фреймворку** - підтримує sklearn, pytorch, tensorflow, та інші
- 🔧 **Включає залежності** - зберігає conda.yaml та requirements.txt
-  **Містить метадані** - схема входів/виходів, версії бібліотек
-  **Спрощує деплоймент** - на різні платформи (локальні сервери, cloud)

### Структура MLflow Model

```
model/
├── MLmodel              # Метадані моделі (YAML)
├── conda.yaml           # Conda залежності
├── python_env.yaml      # Python залежності
├── requirements.txt     # Pip залежності
├── input_example.json   # Приклад вхідних даних
└── model.pkl            # Сама модель
```

### Приклад MLmodel файлу

```yaml
time_created: 2025-10-05T14:30:10.24
flavors:
  sklearn:
    sklearn_version: 1.3.0
    pickled_model: model.pkl
  python_function:
    loader_module: mlflow.sklearn
    python_version: 3.9.0
    data: model.pkl
signature:
  inputs: '[{"name": "feature1", "type": "double"}, ...]'
  outputs: '[{"type": "long"}]'
```

## 🔑 Ключові концепції

### 1. Model Flavors

**Flavor** - спосіб, яким MLflow зберігає та завантажує модель.

Підтримувані flavors:
- `python_function` (pyfunc) - універсальний
- `sklearn` - для scikit-learn
- `pytorch` - для PyTorch
- `tensorflow` - для TensorFlow/Keras
- `xgboost`, `lightgbm` - для gradient boosting
- `onnx` - для ONNX моделей

### 2. Model Signature

Визначає схему вхідних та вихідних даних:

```python
from mlflow.models import infer_signature

# Автоматичне визначення
signature = infer_signature(X_train, y_pred)

# Логування з сигнатурою
mlflow.sklearn.log_model(
    model, 
    "model",
    signature=signature
)
```

### 3. Input Example

Приклад вхідних даних для документації:

```python
input_example = X_train[:5]

mlflow.sklearn.log_model(
    model,
    "model",
    input_example=input_example
)
```

## 💡 Приклади використання

### Базове логування

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Тренування моделі
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Логування
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
```

### Логування з сигнатурою та прикладом

```python
from mlflow.models import infer_signature

# Визначаємо сигнатуру
signature = infer_signature(X_train, y_pred)
input_example = X_train[:5]

mlflow.sklearn.log_model(
    model, 
    "model",
    signature=signature,
    input_example=input_example
)
```

### Завантаження моделі

```python
# Як sklearn модель
model = mlflow.sklearn.load_model("runs:/<run_id>/model")

# Як Python Function (universal)
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")

# З локальної директорії
model = mlflow.sklearn.load_model("./models/my_model")
```

### Кастомна PyFunc модель

```python
class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        # Кастомна логіка
        return self.model.predict(model_input)

mlflow.pyfunc.log_model("model", python_model=CustomModel(model))
```

### Експорт моделі

```python
from src.model_inspector import export_model

# Експорт з MLflow tracking
export_model(
    model_uri="runs:/<run_id>/model",
    output_dir="./models",
    model_name="production_model_v1"
)
```

### Інспекція моделі

```python
from src.model_inspector import inspect_model_structure

# Детальний огляд структури
inspect_model_structure("runs:/<run_id>/model", verbose=True)
```

### REST API Client

```python
from src.model_client import MLflowModelClient

# Створюємо client
client = MLflowModelClient(
    url="http://localhost:5001/invocations",
    feature_names=feature_names
)

# Передбачення
predictions = client.predict(X_test)
single_pred = client.predict_single(X_test[0])
```

## 📊 Датасети

Використовуються вбудовані датасети sklearn:
- **Iris** - класифікація квітів (3 класи)
- **Wine** - класифікація вина (3 класи)  
- **Breast Cancer** - діагностика раку (2 класи)

## 🛠️ Технічний стек

- **MLflow** - model tracking і management
- **Scikit-learn** - ML моделі
- **Pandas** - обробка даних
- **NumPy** - числові операції
- **Jupyter** - інтерактивні ноутбуки

## 📁 Структура проекту

```
03_mlflow_models/
├── README.md                          # Цей файл
├── requirements.txt                   # Python залежності
├── manage_models.py                   # CLI для управління моделями
├── notebooks/                         # Jupyter ноутбуки
│   ├── 01_basic_model_logging.ipynb       # Основи логування
│   ├── 02_model_signatures.ipynb          # Сигнатури та валідація
│   ├── 03_model_flavors.ipynb             # Різні flavors
│   ├── 04_custom_pyfunc_models.ipynb      # Кастомні моделі
│   ├── 05_model_deployment.ipynb          # Деплоймент
│   └── mlruns/                            # MLflow tracking (автогенерується)
├── src/                               # Допоміжний код
│   ├── __init__.py
│   ├── data_loader.py                     # Утиліти для даних
│   ├── model_utils.py                     # Утиліти для моделей
│   ├── model_inspector.py                 # Інспекція моделей
│   └── model_client.py                    # REST API client
├── data/                              # Датасети (якщо потрібні)
└── models/                            # Експортовані моделі
    └── (моделі з'являться після виконання notebooks)
```

### Пояснення директорій:

#### `notebooks/mlruns/` 
**Призначення**: MLflow автоматично зберігає тут всі експерименти, runs та моделі.

**Структура**:
```
mlruns/
├── 0/                          # Default experiment
├── <experiment_id>/            # Ваш експеримент
│   ├── <run_id>/              # Конкретний run
│   │   ├── artifacts/
│   │   │   └── model/         #  ТУТ ЗБЕРІГАЄТЬСЯ МОДЕЛЬ
│   │   │       ├── MLmodel
│   │   │       ├── model.pkl
│   │   │       ├── conda.yaml
│   │   │       ├── requirements.txt
│   │   │       └── input_example.json
│   │   ├── metrics/
│   │   ├── params/
│   │   └── tags/
```

**Коли використовувати**: Автоматично під час виконання `mlflow.log_model()`

#### `models/` 
**Призначення**: Експортовані моделі для production або спільного використання.

**Структура**:
```
models/
├── iris_basic_rf/              # Експортована модель
│   ├── MLmodel
│   ├── model.pkl
│   ├── conda.yaml
│   └── requirements.txt
└── wine_classifier_serving/    # Модель для serving
```

**Коли використовувати**: 
- Коли потрібно зберегти "final" версію моделі
- Для деплойменту на production
- Для спільного використання між проектами

**Як експортувати**:
```python
from model_inspector import export_model
export_model("runs:/<run_id>/model", "../models", "my_model")
```

#### `src/`
**Призначення**: Reusable код для роботи з моделями.

**Модулі**:
- `data_loader.py` - завантаження тестових датасетів
- `model_utils.py` - створення моделей
- `model_inspector.py` - інспекція та експорт моделей
- `model_client.py` - REST API client

## 🔍 Корисні команди

### Перегляд MLflow UI

```bash
cd notebooks
mlflow ui --backend-store-uri file:./mlruns
```

Відкрийте http://localhost:5000 в браузері.

### Model Management CLI

Використовуйте скрипт `manage_models.py` для управління моделями:

```bash
# Список експериментів
python manage_models.py list-experiments

# Список runs в експерименті
python manage_models.py list-runs --experiment-id 0

# Інспекція моделі
python manage_models.py inspect --run-id <run_id>

# Експорт моделі
python manage_models.py export --run-id <run_id> --output ./models --name my_model

# Порівняння моделей
python manage_models.py compare --run-id1 <id1> --run-id2 <id2>

# Розмір моделі
python manage_models.py size --run-id <run_id>
```

### Model Serving

```bash
# Локальний serving
mlflow models serve -m runs:/<run_id>/model -p 5001 --no-conda

# Serving з директорії
mlflow models serve -m ./models/my_model -p 5001 --no-conda

# З conda environment
mlflow models serve -m runs:/<run_id>/model -p 5001 --env-manager conda
```

### Список експериментів (Python)

```python
import mlflow
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")
```

### Пошук runs

```python
runs = mlflow.search_runs(experiment_ids=["0"])
print(runs[['run_id', 'metrics.accuracy', 'params.n_estimators']])
```

## 📚 Додаткові ресурси

- [MLflow Models Documentation](https://mlflow.org/docs/latest/models.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [Model Signatures](https://mlflow.org/docs/latest/models.html#model-signature)
- [Custom Python Models](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)

## 🎓 Для викладачів

### Рекомендований порядок викладання:

1. **Лекція**: Концепція MLflow Models, flavors, signatures
2. **Практика**: Notebook 01 - базове логування
3. **Практика**: Notebook 02 - signatures та валідація
4. **Практика**: Notebook 03 - різні flavors
5. **Домашнє завдання**: Створити власну модель з кастомним PyFunc

### Ключові моменти для пояснення:

- Чому важлива стандартизація формату моделей
- Різниця між різними flavors
- Коли використовувати signatures
- Переваги MLflow Models для production

## ❓ FAQ

**Q: В чому різниця між sklearn та pyfunc flavor?**
A: Sklearn - специфічний для sklearn моделей з доступом до всіх методів. PyFunc - універсальний інтерфейс для будь-яких моделей.

**Q: Чи обов'язково вказувати signature?**
A: Ні, але рекомендується для валідації та документації.

**Q: Чи можна логувати кілька моделей в одному run?**
A: Так, використовуйте різні artifact_path для кожної моделі.

**Q: Як видалити старі runs?**
A: Через MLflow UI або програмно через `mlflow.delete_run(run_id)`.

**Q: Що зберігається в models/ vs mlruns/?**
A: `mlruns/` - автоматичне зберігання MLflow під час експериментів. `models/` - експортовані моделі для production або спільного використання.

**Q: Як перенести модель з mlruns/ в models/?**
A: Використайте утиліту `export_model()` або CLI команду `python manage_models.py export`.

**Q: Чи потрібен MLflow tracking для використання експортованої моделі?**
A: Ні! Експортовані моделі в `models/` можна завантажувати без tracking:
```python
model = mlflow.sklearn.load_model('./models/my_model')
```

**Q: Як працює model serving?**
A: MLflow запускає Flask REST API сервер, який приймає JSON запити та повертає передбачення.

##  Ліцензія

Цей проект створено для навчальних цілей.
