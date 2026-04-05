# Quick Start Guide - MLflow Models

Швидкий старт для роботи з MLflow Models демонстраційним проектом.

##  Швидкий старт (5 хвилин)

### 1. Встановіть залежності

```bash
cd 03_mlflow_models
pip install -r requirements.txt
```

### 2. Запустіть Jupyter

```bash
cd notebooks
jupyter notebook
```

### 3. Відкрийте перший ноутбук

Відкрийте `01_basic_model_logging.ipynb` та виконайте всі комірки.

### 4. Переглядайте результати в MLflow UI

```bash
# В директорії notebooks
mlflow ui --backend-store-uri file:./mlruns
```

Відкрийте http://localhost:5000

## 📚 Рекомендований порядок вивчення

### День 1: Основи
1. `01_basic_model_logging.ipynb` - Логування та структура моделей
2. `02_model_signatures.ipynb` - Сигнатури та валідація

### День 2: Flavors та управління
3. `03_model_flavors.ipynb` - Різні flavors (sklearn, pyfunc)
4. Практика з CLI: `python manage_models.py --help`

### День 3: Advanced
5. `04_custom_pyfunc_models.ipynb` - Кастомні моделі
6. `05_model_deployment.ipynb` - Деплоймент та serving

##  Що ви навчитесь

 Логувати ML моделі з MLflow
 Розуміти структуру MLflow Model (MLmodel, conda.yaml, etc.)
 Працювати з model signatures
 Створювати кастомні PyFunc моделі  
 Експортувати та управляти моделями
 Запускати model serving
 Працювати з REST API

## 💡 Корисні поради

### Для студентів:

1. **Виконуйте notebooks послідовно** - кожен базується на попередньому
2. **Експериментуйте** - змінюйте параметри, пробуйте інші моделі
3. **Дивіться в MLflow UI** - відстежуйте свої експерименти
4. **Читайте коментарі** - вони пояснюють кожен крок

### Для викладачів:

1. **Лекція перед практикою** - спочатку концепція, потім код
2. **Live demo** - покажіть MLflow UI в реальному часі
3. **Групові завдання** - створити та порівняти різні моделі
4. **Homework** - створити власну кастомну PyFunc модель

## 🛠️ Troubleshooting

### Проблема: MLflow UI не запускається

```bash
# Переконайтесь, що ви в правильній директорії
cd notebooks
mlflow ui --backend-store-uri file:./mlruns
```

### Проблема: Model serving не працює

```bash
# Використовуйте --no-conda для швидшого запуску
mlflow models serve -m runs:/<run_id>/model -p 5001 --no-conda
```

### Проблема: Модуль src не знайдено

```python
# Додайте в початок notebook
import sys
sys.path.append('../src')
```

## Додаткові ресурси

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Models Guide](https://mlflow.org/docs/latest/models.html)
- [Model Signatures](https://mlflow.org/docs/latest/models.html#model-signature)
- [Custom Python Models](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)

## 🎓 Завдання для самостійної роботи

1. **Базове**: Створіть модель з кастомним preprocessing
2. **Середнє**: Створіть ensemble з 3 різних моделей
3. **Складне**: Створіть PyFunc модель з зовнішнім API lookup
4. **Expert**: Додайте A/B testing між двома моделями

## 📞 Потрібна допомога?

- Перевірте README.md для детальної інформації
- Подивіться FAQ секцію
- Використовуйте `manage_models.py --help` для CLI команд
