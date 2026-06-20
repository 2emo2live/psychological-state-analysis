# Анализ психического состояния по сообщениям пользователей

## Содержание

1. [Краткое описание проекта](#1-краткое-описание-проекта)
2. [Постановка задачи](#2-постановка-задачи)
3. [Данные](#3-данные)
4. [Метрики и валидация](#4-метрики-и-валидация)
5. [Структура репозитория](#5-структура-репозитория)
6. [Setup](#6-setup)
7. [Конфигурации Hydra](#7-конфигурации-hydra)
8. [DVC: данные и модельные артефакты](#8-dvc-данные-и-модельные-артефакты)
9. [MLflow](#9-mlflow)
10. [Train](#10-train)
11. [Production preparation](#11-production-preparation)
12. [Infer](#12-infer)

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

## 2. Постановка задачи

### 2.1. Тип задачи

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

### 2.2. Входные данные в обучении

Датасет содержит ~53 000 записей. Каждая запись содержит поле `ыtatement` (текст) и `status` (метка).
Максимальная длина текста после токенизации для DistilBERT — **512 токенов**, в проекте используется порог обрезки`max_len = 512` (покрывает >95% сообщений).
Для бейзлайна текст предобрабатывается (стемминг, удаление пунктуации и ссылок). Для основной модели текст не изменяется (сохраняется пунктуация и регистр, т.к. DistilBERT использует токенизатор с субсловными единицами).

### 2.3. Практическая мотивация

Сервис может использоваться как компонент для мониторинга психологического состояния пользователей в чат‑ботах и социальных сетях, помогая своевременно выявлять тревожные сигналы и снижать риски негативного воздействия ИИ.

---

## 3. Данные

### 3.1. Источник данных

Kaggle‑датасет:
[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)

### 3.2. Характеристики датасета

| Свойство           | Значение         |
| ------------------ | ---------------- |
| Тип данных         | тексты           |
| Количество записей | ~53 000          |
| Размер             | 31.5 МБ          |
| Количество классов | 7                |
| Пропуски           | есть (удаляются) |
| Целевая колонка    | `status`         |

### 3.3. Распределение классов

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

### 3.4. Стратифицированное разделение

Для сохранения распределения классов используется стратифицированный сплит:
`train / val / test = 70 / 15 / 15`.

---

## 4. Метрики и валидация

### 4.1. Основные метрики

В проекте используются следующие метрики:

- **Accuracy** – доля правильных ответов.
- **Macro Precision** – средняя точность по классам (без учёта дисбаланса).
- **Macro Recall** – средняя полнота по классам.
- **Macro F1** – среднее гармоническое Precision и Recall.
- **Weighted F1** – F1, взвешенный по поддержке классов (учитывает дисбаланс).

Дополнительно логируются `train_loss` и `val_loss`.

### 4.2. Почему выбраны эти метрики

Поскольку датасет несбалансирован, макро‑усреднённые метрики дают более объективную оценку, чем обычная accuracy. Weighted F1 помогает контролировать качество на частых классах.

### 4.3. Ожидаемые значения

Ожидаемые значения основаны на анализе решений с Kaggle, метрик DistilBERT (https://huggingface.co/distilbert/distilbert-base-uncased) и публикациях по аналогичным задачам:

- Бейзлайн (TF‑IDF + LR): **Balanced Accuracy ≈ 0.60–0.70**, **Macro F1 ≈ 0.60–0.70**.
- Основная модель (DistilBERT): **Balanced Accuracy ≥ 0.80**, **Macro F1 ≥ 0.80**.

Референсные значения подтверждены несколькими открытыми ноутбуками на Kaggle.

### 4.4. Валидация и тестирование

Валидация используется для выбора лучшего чекпоинта (по Macro F1).
Тестовая выборка используется только для финальной оценки (standalone‑отчёт).

---

## 5. Структура репозитория

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
    ├── train_accuracy_epoch.png
    ├── train_accuracy_step.png
    ├── train_loss_epoch.png
    ├── train_loss_step.png
    ├── val_loss.png
    ├── val_accuracy.png
    ├── val_precision.png
    ├── val_recall.png
    └── val_weighted_f1.png
```

---

## 6. Setup

### 6.1. Требования

- Python ≥3.10
- `uv`
- Git
- DVC
- Docker (для Triton)
- NVIDIA GPU + TensorRT (для экспорта в TensorRT)

### 6.2. Клонирование

```bash
git clone https://github.com/2emo2live/psychological-state-analysis
cd psychological_state_analysis
```

### 6.3. Установка uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 6.4. Установка зависимостей

```bash
uv sync
```

Если планируется обучать бейзлайн:

```bash
uv sync --extra baseline
```

### 6.5. Pre‑commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

---

## 7. Конфигурации Hydra

Главная точка входа — `configs/config.yaml`.

Группы конфигов:

```
- `data/default.yaml` – путь к данным, `max_len`, `batch_size`
- `baseline/default.yaml` – TF‑IDF параметры, логистическая регрессия
- `model/distilbert.yaml` – имя модели, `num_layers_to_train`, `learning_rate`
- `train/default.yaml` – `max_epochs`, `accelerator`, `devices`, `seed`
- `logging/mlflow.yaml` – `mlflow_uri`, `experiment_name`
- `export/onnx.yaml` – `opset_version`, `input_names`, `output_names`
- `export/tensorrt.yaml` – `precision`, `min_batch_size`, `opt_batch_size`, `max_batch_size`

```

---

## 8. DVC: данные и модельные артефакты

### 8.1. DVC remotes

В проекте два хранилища:

- `data-remote` – для датасета и обработанных CSV
- `models-remote` – для чекпоинтов, ONNX и TensorRT артефактов

Настройка (локальное хранилище):

Хранилища расположены в Google Drive, поэтому для их использования необходимо настроить dvc-gdrive (см. гайд на офциальном сайте: https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended)

### 8.2. Получение данных

```bash
dvc pull -r data-remote
```

### 8.3. Добавление данных в DVC

```bash
dvc add data/Data.csv
dvc push -r data-remote
```

Аналогично для моделей:

```bash
dvc add models/best_model.onnx
dvc push -r models-remote
```

---

## 9. MLflow

### 9.1. Запуск локального сервера

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

### 9.2. Что логируется

- Гиперпараметры (из Hydra)
- train/val loss, метрики
- графики (через `self.log`)
- лучший чекпоинт

---

## 10. Train

### 10.1. Бейзлайн

Обучение TF‑IDF + LogisticRegression:

```bash
uv run python -m psychology_state_analyzer.commands train_baseline
```

Конфиг: `configs/model/baseline.yaml`.
Гиперпараметры TF‑IDF: `max_features=10000`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`.
LogisticRegression: `C=1.0`, `solver='lbfgs'`, `max_iter=1000`.

Полученные метрики:
`val_macro_f1 ≈ 0.73`, `test_macro_f1 ≈ 0.69`.

### 10.2. Основная модель (DistilBERT)

```bash
uv run python -m psychology_state_analyzer.commands train_main
```

Особенности:

- Загружается `AutoModel` (без головы).
- Добавляется линейный слой `nn.Linear(hidden_size, num_classes)`.
- Все слои DistilBERT замораживаются, затем размораживаются последние `num_layers_to_train` (по умолчанию 2).
- Оптимизатор: AdamW с `lr=2e-5`, `weight_decay=0.01`.
- Планировщик: линейный warmup (10% шагов).

Полученные метрики:
`val_macro_f1 ≈ 0.83`, `test_macro_f1 ≈ 0.82`.

---

## 11. Production preparation

### 11.1. Экспорт в ONNX

```bash
uv run python -m psychology_state_analyzer.commands export_onnx
```

Параметры: `opset_version=17`, динамический батч, динамическая длина последовательности.

### 11.2. Экспорт в TensorRT

Требуется NVIDIA GPU и `trtexec`. Выполняется:

```bash
uv run python -m psychology_state_analyzer.commands export_tensorrt
```

Создаётся движок `.plan`.

### 11.3. Triton model repository

`config.pbtxt` для ONNX (CPU):

```protobuf
name: "mental_health_classifier_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  { name: "input_ids", data_type: TYPE_INT64, dims: [512] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [512] }
]
output [
  { name: "logits", data_type: TYPE_FP32, dims: [7] }
]
instance_group [ { kind: KIND_CPU } ]
```

Для TensorRT (GPU) – аналогично, но `platform: "tensorrt_plan"` и `kind: KIND_GPU`.

---

## 12. Infer

### 12.1. Запуск Triton сервера

```
bash
./triton_server/run_triton_server.sh
```

Проверка готовности:

```bash
curl -s localhost:8000/v2/health/ready
```

### 12.2. Клиентский скрипт (infer.py)

Пример использования:

```bash
uv run python infer.py --text "I feel very anxious"
```

Клиент выполняет предобработку (токенизация через `AutoTokenizer`), отправляет запрос в Triton, получает логиты и возвращает классы.
