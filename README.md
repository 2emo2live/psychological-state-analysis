# Анализ психического состояния по сообщениям пользователей

## Содержание

1. [Краткое описание проекта](#1-краткое-описание-проекта)
2. [Постановка задачи](#3-постановка-задачи)
3. [Данные](#4-данные)
4. [Метрики и валидация](#5-метрики-и-валидация)
5. [Структура репозитория](#6-структура-репозитория)
6. [Setup](#7-setup)
7. [Конфигурации Hydra](#8-конфигурации-hydra)
8. [DVC: данные и модельные артефакты](#9-dvc-данные-и-модельные-артефакты)
9. [Data workflow](#10-data-workflow)
10. [MLflow](#11-mlflow)
11. [Train](#12-train)
12. [Evaluation](#13-evaluation)
13. [Model selection](#14-model-selection)
14. [Production preparation](#15-production-preparation)
15. [Infer](#16-infer)
16. [Google Colab workflow](#17-google-colab-workflow)
17. [Краткая карта основных команд](#18-краткая-карта-основных-команд)

---

## 1. Краткое описание проекта

Целью данной работы является разработка системы автоматической классификации психического состояния пользователя на основе его сообщений.
В настоящее время задача оценки психического состояния пользователей является особенно актуальной в связи с активным использованием социальных сетей и чат-ботов. Отдельно стоит отметить рост числа сообщений о негативных психологических эффектах взаимодействия пользователей с чат-ботами (так называемый ‘ИИ-психоз’). Разработка алгоритмов автоматического анализа сообщений пользователей может помочь повысить безопасность взаимодействия человека с системами на основе генеративного ИИ.

Пайплайн включает:

- предобработку текста;
- DVC‑версионирование данных и модельных артефактов;
- обучение бейзлайна (TF‑IDF + логистическая регрессия);
- обучение основной модели (DistilBERT с дополнительным линейным слоем, дообучение последних 2 слоёв);
- логирование экспериментов в MLflow;
- standalone‑оценку с отчётами и графиками;
- выбор лучшей модели по валидационному макро‑F1;
- экспорт модели в ONNX;
- валидацию ONNX через ONNX Runtime;
- экспорт модели в TensorRT;
- подготовку репозитория для Triton Inference Server;
- инференс через HTTP‑клиент Triton.

---

## 3. Постановка задачи

### 3.1. Тип задачи

Многоклассовая классификация текстовых сообщений.

**Вход:**
Текстовое сообщение (строка произвольной длины, от 2 до 32759 символов).
Пример: `"trouble sleeping, confused mind, restless heart. All out of tune"`.

**Выход:**
Метка одного из семи классов:

- Anxiety: 0
- Bi‑Polar: 1
- Depression: 2
- Normal: 3
- Personality Disorder: 4
- Stress: 5
- Suicidal: 6

### 3.2. Входные данные в обучении

Датасет содержит ~53 000 записей. Каждая запись содержит поле `ыtatement` (текст) и `status` (метка).
Максимальная длина текста после токенизации для DistilBERT — **512 токенов**, в проекте используется порог обрезки`max_len = 512` (покрывает >95% сообщений).
Для бейзлайна текст предобрабатывается (стемминг, удаление пунктуации и ссылок). Для основной модели текст не изменяется (сохраняется пунктуация и регистр, т.к. DistilBERT использует токенизатор с субсловными единицами).

### 3.3. Выходные данные инференса

Система возвращает структурированный ответ:

```json
{
  "predicted_class": "Anxiety",
  "predicted_class_index": 3,
  "confidence": 0.8234,
  "top_k": [
    {"class_index": 3, "class_name": "Anxiety", "confidence": 0.8234},
    {"class_index": 1, "class_name": "Depression", "confidence": 0.1123},
    ...
  ]
}
```

### 3.4. Практическая мотивация

Сервис может использоваться как компонент для мониторинга психологического состояния пользователей в чат‑ботах и социальных сетях, помогая своевременно выявлять тревожные сигналы и снижать риски негативного воздействия ИИ.

---

## 4. Данные

### 4.1. Источник данных

Kaggle‑датасет:
[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)

### 4.2. Характеристики датасета

| Свойство           | Значение         |
| ------------------ | ---------------- |
| Тип данных         | тексты           |
| Количество записей | ~53 000          |
| Размер             | 31.5 МБ          |
| Количество классов | 7                |
| Пропуски           | есть (удаляются) |
| Целевая колонка    | `status`         |

### 4.3. Распределение классов

| Класс                | Доля |
| -------------------- | ---- |
| Normal               | 31 % |
| Depression           | 29 % |
| Suicidal             | 20 % |
| Anxiety              | 7 %  |
| Stress               | 5 %  |
| Bi‑Polar             | 5 %  |
| Personality Disorder | 2 %  |

Датасет несбалансирован (самый частый класс встречается в ~15 раз чаще самого редкого).

### 4.4. Стратифицированное разделение

Для сохранения распределения классов используется стратифицированный сплит:
`train / val / test = 70 / 15 / 15`.

---

## 5. Метрики и валидация

### 5.1. Основные метрики

В проекте используются следующие метрики:

- **Accuracy** – доля правильных ответов.
- **Macro Precision** – средняя точность по классам (без учёта дисбаланса).
- **Macro Recall** – средняя полнота по классам.
- **Macro F1** – среднее гармоническое Precision и Recall.
- **Weighted F1** – F1, взвешенный по поддержке классов (учитывает дисбаланс).

Дополнительно логируются `train_loss` и `val_loss`.

### 5.2. Почему выбраны эти метрики

Поскольку датасет несбалансирован, макро‑усреднённые метрики дают более объективную оценку, чем обычная accuracy. Weighted F1 помогает контролировать качество на частых классах.

### 5.3. Ожидаемые значения

Ожидаемые значения основаны на анализе решений с Kaggle и публикациях по аналогичным задачам:

- Бейзлайн (TF‑IDF + LR): **Balanced Accuracy ≈ 0.60–0.70**, **Macro F1 ≈ 0.60–0.70**.
- Основная модель (DistilBERT): **Balanced Accuracy ≥ 0.80**, **Macro F1 ≥ 0.80**.

Референсные значения подтверждены несколькими открытыми ноутбуками на Kaggle.

### 5.4. Валидация и тестирование

Валидация используется для выбора лучшего чекпоинта (по Macro F1).
Тестовая выборка используется только для финальной оценки (standalone‑отчёт).

---

## 6. Структура репозитория

```
psychology_state_analyzer/
├── .dvc/
│   └── config
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── README.md
├── pyproject.toml
├── uv.lock
├── psychology_state_analyzer/
│   ├── __init__.py
│   ├── commands.py                    # единая точка входа
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── baseline_preprocessing.py  # очистка текста (nltk / spaCy)
│   │   ├── dataset.py                 # MentalHealthDataset
│   │   ├── load_data.py               # загрузка данных
│   │   └── datamodule.py              # MentalHealthDataModule
│   ├── model_class/
│   │   ├── __init__.py
│   │   ├── baseline.py                # BaselineModel (TF‑IDF + LR)
│   │   └── main_model.py              # MentalHealthLightningModule (DistilBERT)
│   ├── train_models/
│   │   ├── __init__.py
│   │   ├── train_baseline.py
│   │   └── train_main_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── export_to_onnx.py
│   │   └── export_to_tensorrt.py
├── configs/
│   ├── config.yaml                    # главный
│   ├── data/
│   │   └── default.yaml
│   ├── baseline/
│   │   └── default.yaml
│   ├── model/
│   │   └── distilbert.yaml
│   ├── train/
│   │   └── default.yaml
│   ├── logger/
│   │   └── default.yaml
│   └── export/
│       └── default.yaml
├── data/
│   └── Data.csv.dvc
├── models/                              # артефакты (не в Git)
│   ├── baseline/
│   │   ├── model.pkl
│   │   ├── vectorizer.pkl
│   │   ├── params.pkl
│   │   └── label_encoder.pkl
│   ├── model.onnx
│   └── model.plan
├── triton_server/
│   └── model_repository/
│       ├── psychological_state_analyzer/
│       │   └── 1/
│       │       └── model.onnx
│       └───── config.pbtxt
└── plots/
    ├── train_loss.png
    ├── val_loss.png
    └── val_macro_f1.png
```

---

## 7. Setup

### 7.1. Требования

- Python ≥3.10
- `uv`
- Git
- DVC
- Docker (для Triton)
- NVIDIA GPU + TensorRT (для экспорта в TensorRT)

### 7.2. Клонирование

```bash
git clone https://github.com/yourusername/psychology_state_analyzer.git
cd psychological_state_analysis
```

### 7.3. Установка uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 7.4. Установка зависимостей

```bash
uv sync
```

Если не нужна группа `inference` (Triton клиент) на Mac:

```bash
uv sync --no-group inference
```

Если планируется обучать бейзлайн:

```bash
uv sync --extra baseline
```

### 7.5. Pre‑commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

---

## 8. Конфигурации Hydra

Главная точка входа — `configs/config.yaml`.

Группы конфигов:

- `data/default.yaml` – путь к данным, `max_len`, `batch_size`
- `baseline/default.yaml` – TF‑IDF параметры, логистическая регрессия
- `model/distilbert.yaml` – имя модели, `num_layers_to_train`, `learning_rate`
- `train/default.yaml` – `max_epochs`, `accelerator`, `devices`, `seed`
- `logging/mlflow.yaml` – `mlflow_uri`, `experiment_name`
- `export/onnx.yaml` – `opset_version`, `input_names`, `output_names`
- `export/tensorrt.yaml` – `precision`, `min_batch_size`, `opt_batch_size`, `max_batch_size`

Пример переопределения:

```bash
uv run python -m psychology_state_analyzer.training.train_main \
  model=distilbert \
  model.learning_rate=1e-5 \
  data.batch_size=16 \
  trainer.max_epochs=10
```

---

## 9. DVC: данные и модельные артефакты

### 9.1. DVC remotes

В проекте два хранилища:

- `data-remote` – для датасета и обработанных CSV
- `models-remote` – для чекпоинтов, ONNX и TensorRT артефактов

Настройка (локальное хранилище):

Хранилища расположены в Google Drive, поэтому для их использования необходимо настроить dvc-gdrive (см. гайд на фоциальном сайте: https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended)

### 9.2. Получение данных

```bash
dvc pull -r data-remote
```

### 9.3. Добавление данных в DVC

```bash
dvc add data/raw/Combined\ Data.csv
dvc push -r data-remote
```

Аналогично для моделей:

```bash
dvc add models/onnx/model.onnx
dvc push -r models-remote
```

---

## 11. MLflow

### 11.1. Запуск локального сервера

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --default-artifact-root ./mlruns
```

### 11.2. Что логируется

- Гиперпараметры (из Hydra)
- train/val loss, метрики
- графики (через `self.log`)
- лучший чекпоинт

---

## 12. Train

### 12.1. Бейзлайн

Обучение TF‑IDF + LogisticRegression:

```bash
uv run python -m psychology_state_analyzer.commands train_baseline
```

Конфиг: `configs/model/baseline.yaml`.
Гиперпараметры TF‑IDF: `max_features=10000`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`.
LogisticRegression: `C=1.0`, `solver='lbfgs'`, `max_iter=1000`.

Ожидаемые метрики:
`val_macro_f1 ≈ 0.65–0.70`, `test_macro_f1 ≈ 0.65–0.70`.

### 12.2. Основная модель (DistilBERT)

```bash
uv run python -m psychology_state_analyzer.commands train_main
```

Особенности:

- Загружается `AutoModel` (без головы).
- Добавляется линейный слой `nn.Linear(hidden_size, num_classes)`.
- Все слои DistilBERT замораживаются, затем размораживаются последние `num_layers_to_train` (по умолчанию 2).
- Оптимизатор: AdamW с `lr=2e-5`, `weight_decay=0.01`.
- Планировщик: линейный warmup (10% шагов).

Ожидаемые метрики:
`val_macro_f1 ≈ 0.82–0.85`, `test_macro_f1 ≥ 0.80`.

### 12.3. Ресурсы

- Бейзлайн: CPU (любой), память ~2 ГБ.
- DistilBERT: рекомендуется GPU с 8+ ГБ VRAM (например, NVIDIA T4). Обучение 3 эпох на T4 занимает ~20 минут.

---

## 13. Evaluation

Standalone‑оценка на тестовой выборке:

Пример `summary_metrics.json`:

```json
{
  "accuracy": 0.85,
  "macro_precision": 0.83,
  "macro_recall": 0.82,
  "macro_f1": 0.825,
  "weighted_f1": 0.85
}
```

## 15. Production preparation

### 15.1. Экспорт в ONNX

```bash
uv run python -m psychology_state_analyzer.commands export_onnx
```

Параметры: `opset_version=17`, динамический батч, динамическая длина последовательности.

### 15.2. Экспорт в TensorRT

Требуется NVIDIA GPU и `trtexec`. Выполняется:

```bash
uv run python -m psychology_state_analyzer.commands export_tensorrt
```

Создаётся движок `.plan`.

### 15.3. Triton model repository

Сборка репозитория:

```bash
uv run python -m psychology_state_analyzer.serving.triton_repository \
  model=distilbert \
  onnx_path=models/onnx/model.onnx \
  tensorrt_path=models/tensorrt/model.plan
```

Структура:

```
deployment/triton_model_repository/
├── mental_health_classifier_onnx/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
└── mental_health_classifier_tensorrt/
    ├── 1/
    │   └── model.plan
    └── config.pbtxt
```

`config.pbtxt` для ONNX (CPU):

```protobuf
name: "mental_health_classifier_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  { name: "input_ids", data_type: TYPE_INT64, dims: [128] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [128] }
]
output [
  { name: "logits", data_type: TYPE_FP32, dims: [7] }
]
instance_group [ { kind: KIND_CPU } ]
```

Для TensorRT (GPU) – аналогично, но `platform: "tensorrt_plan"` и `kind: KIND_GPU`.

---

## 16. Infer

### 16.1. Запуск Triton сервера

```bash
TRITON_ENABLE_GPU=false TRITON_LOAD_MODEL=mental_health_classifier_onnx \
  bash scripts/run_triton_server.sh
```

Проверка готовности:

```bash
curl -s localhost:8000/v2/health/ready
```

### 16.2. Клиентский скрипт (infer.py)

Пример использования:

```bash
uv run python infer.py --text "I feel very anxious"
```

Ответ:

```json
{
  "predicted_class": "Anxiety",
  "predicted_class_index": 3,
  "confidence": 0.87,
  "top_k": [...]
}
```

Клиент выполняет предобработку (токенизация через `AutoTokenizer`), отправляет запрос в Triton, применяет softmax к логитам и возвращает классы.
