# Подготовка датасета YMAD для оценки TrafficCamNet

**Дата:** 14 января 2026  
**Статус:** ✅ Подготовка завершена  
**Автор:** CarCV Team

---

## Обзор

Выполнена подготовка датасета Yandex MAD (Moscow Automotive Dataset) для оценки детектора транспортных средств TrafficCamNet. Датасет содержит изображения российских автомобилей, что делает его отличным выбором для тестирования моделей в условиях российских дорог.

---

## Датасет YMAD

### Характеристики

| Параметр | Значение |
|----------|----------|
| **Всего изображений** | 10,432 |
| **Уникальных автомобилей** | 2,000 |
| **Брендов** | 79 |
| **Моделей** | 484 |
| **Цветов** | 15 |

### Топ-10 брендов

| Бренд | Количество изображений | % от общего |
|-------|------------------------|-------------|
| VAZ (Lada) | 1,298 | 12.4% |
| Kia | 781 | 7.5% |
| Toyota | 709 | 6.8% |
| BMW | 623 | 6.0% |
| Hyundai | 619 | 5.9% |
| Volkswagen | 542 | 5.2% |
| Nissan | 542 | 5.2% |
| Mercedes | 525 | 5.0% |
| Ford | 524 | 5.0% |
| Audi | 396 | 3.8% |

### Топ-10 моделей

| Модель | Количество | Бренд |
|--------|------------|-------|
| Focus | 291 | Ford |
| Rio | 237 | Kia |
| Solaris | 198 | Hyundai |
| Granta | 190 | VAZ |
| Camry | 161 | Toyota |
| 3er | 148 | BMW |
| Octavia | 138 | Skoda |
| 2170 (Priora) | 138 | VAZ |
| Polo | 137 | Volkswagen |
| Ceed | 136 | Kia |

### Распределение по цветам

- Белый (ffffff): 2,281 (21.9%)
- Черный (000000): 2,229 (21.4%)
- Серый (9c9999): 1,640 (15.7%)
- Серебристый (cacecb): 1,229 (11.8%)
- Синий (0000ff): 974 (9.3%)
- Остальные 10 цветов: 2,079 (19.9%)

---

## Выполненные работы

### 1. Создание COCO аннотаций

✅ **Задача:** Конвертация датасета YMAD в формат COCO для оценки детекторов

**Файл:** `data/processed/ymad_detection/annotations.json`

**Детали:**
- Обработано: 10,432 изображений
- Создано аннотаций: 10,432 (по одной на изображение)
- Формат bounding box: [x, y, width, height] (COCO format)
- Категория: `car` (ID: 1)

**Подход:**
Поскольку YMAD содержит обрезанные изображения автомобилей (не сцены с дорогами), для каждого изображения создан bbox, покрывающий 90% площади изображения (с отступами 5% от краев).

### 2. Генерация статистики

✅ **Задача:** Анализ распределения брендов, моделей и цветов

**Файл:** `data/processed/ymad_detection/dataset_stats.json`

**Результаты:**
- Топ-20 брендов с количеством
- Топ-20 моделей с количеством
- Полная статистика по 15 цветам

### 3. Создание test split

✅ **Задача:** Разделение датасета на test/train

**Файл:** `data/processed/ymad_detection/test_split.json`

**Параметры:**
- Test размер: 1,564 изображений (15%)
- Train размер: 8,868 изображений (85%)
- Метод: случайное разделение с seed=42

---

## Созданные инструменты

### 1. prepare_ymad_for_detection.py

Скрипт для подготовки датасета YMAD к оценке детекторов.

**Функционал:**
- Чтение индекса изображений (JSONL)
- Создание COCO аннотаций с bounding boxes
- Генерация статистики датасета
- Создание train/test split

**Использование:**
```bash
uv run python scripts/prepare_ymad_for_detection.py \
  --images-dir data/external/ymad_cars/images \
  --images-index data/external/ymad_cars/images_index.jsonl \
  --output-dir data/processed/ymad_detection \
  --test-ratio 0.15
```

### 2. evaluate_detector_yolo.py

Скрипт для baseline оценки с использованием YOLOv8.

**Функционал:**
- Загрузка предобученной модели YOLOv8
- Детекция на датасете YMAD
- Вычисление метрик: Precision, Recall, F1, TP/FP/FN
- Измерение производительности (FPS, latency)

**Использование:**
```bash
uv run python scripts/evaluate_detector_yolo.py \
  --images-dir data/external/ymad_cars/images \
  --annotations data/processed/ymad_detection/annotations.json \
  --output-dir results/baseline/yolov8n_ymad \
  --model yolov8n.pt
```

### 3. evaluate_trafficcamnet_jetson.py

Скрипт для оценки TrafficCamNet на Jetson устройстве.

**Функционал:**
- Загрузка TensorRT engine TrafficCamNet
- Детекция на датасете с inference через TensorRT
- Вычисление метрик качества и производительности
- Сохранение результатов

**Примечание:** Требует запуск на Jetson с установленным DeepStream SDK.

### 4. README_EVALUATION.md

Полное руководство по процессу оценки, содержащее:
- Пошаговые инструкции для каждого этапа
- Команды для запуска скриптов
- Инструкции по копированию данных на Jetson
- Примеры ожидаемых результатов
- Раздел troubleshooting

---

## Следующие шаги

### Шаг 1: Baseline оценка с YOLOv8 (опционально)

Получить reference метрики для сравнения:

```bash
# Установка Ultralytics
uv pip install ultralytics

# Запуск оценки
uv run python scripts/evaluate_detector_yolo.py \
  --model yolov8s.pt \
  --output-dir results/baseline/yolov8s_ymad
```

**Ожидаемое время:** 15-30 минут (на GPU)

### Шаг 2: Подготовка для Jetson

Упаковать данные для копирования на Jetson:

```bash
cd /home/user/CarCV
tar -czf ymad_detection_data.tar.gz \
  data/external/ymad_cars/images \
  data/processed/ymad_detection

# Копирование на Jetson (замените IP)
scp ymad_detection_data.tar.gz jetson@<JETSON_IP>:~/trafficcamnet_eval/
scp scripts/evaluate_trafficcamnet_jetson.py jetson@<JETSON_IP>:~/trafficcamnet_eval/
```

### Шаг 3: Оценка на Jetson

**На Jetson устройстве:**

```bash
cd ~/trafficcamnet_eval
tar -xzf ymad_detection_data.tar.gz

python3 evaluate_trafficcamnet_jetson.py \
  --engine /opt/nvidia/deepstream/.../trafficcamnet/resnet18_trafficcamnet.engine \
  --images-dir data/external/ymad_cars/images \
  --annotations data/processed/ymad_detection/annotations.json \
  --output-dir results/trafficcamnet_ymad
```

**Ожидаемое время:** 30-60 минут

### Шаг 4: Анализ результатов

После получения результатов с Jetson:

```bash
# Копирование результатов обратно
scp jetson@<JETSON_IP>:~/trafficcamnet_eval/results/trafficcamnet_ymad/*.json \
  /home/user/CarCV/results/baseline/trafficcamnet_ymad/

# Сравнительный анализ
# (будет добавлен скрипт сравнения)
```

---

## Особенности датасета YMAD для оценки детекторов

### Преимущества

✅ **Российские дороги и автомобили**
- Высокое представительство российских брендов (VAZ, UAZ, GAZ)
- Российские номерные знаки
- Условия российских дорог

✅ **Разнообразие**
- 79 брендов, 484 модели
- Различные ракурсы (0°, 25°, 50°, ...)
- 15 различных цветов

✅ **Качество**
- Высокое разрешение изображений
- Чистые изображения без окклюзий
- Профессиональная съемка

### Ограничения

⚠️ **Формат датасета**
- Обрезанные изображения автомобилей (не сцены)
- Отсутствие natural context (дороги, фон)
- Один автомобиль на изображение

⚠️ **Для детекторов**
- Не оценивает способность находить автомобили в сложных сценах
- Не тестирует multiple object detection
- Не содержит edge cases (окклюзии, перекрытия)

⚠️ **Рекомендация**
Использовать YMAD как дополнение к оценке на реальных дорожных сценах.

---

## Структура файлов

```
CarCV/
├── data/
│   ├── external/
│   │   └── ymad_cars/
│   │       ├── images/             # 10,432 изображений
│   │       ├── images_index.jsonl  # Индекс с метаданными
│   │       └── metadata.json       # Общая информация
│   │
│   └── processed/
│       └── ymad_detection/
│           ├── annotations.json    # COCO аннотации
│           ├── dataset_stats.json  # Статистика
│           └── test_split.json     # Train/test split
│
├── scripts/
│   ├── prepare_ymad_for_detection.py         # Подготовка данных
│   ├── evaluate_detector_yolo.py             # YOLOv8 baseline
│   ├── evaluate_trafficcamnet_jetson.py      # TrafficCamNet на Jetson
│   └── README_EVALUATION.md                  # Полное руководство
│
└── results/
    └── baseline/
        ├── yolov8n_ymad/          # Результаты YOLOv8 (будут созданы)
        └── trafficcamnet_ymad/    # Результаты TrafficCamNet (будут созданы)
```

---

## Метрики для оценки

### Метрики качества

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **Precision** | Доля правильных детекций | >0.90 |
| **Recall** | Доля найденных автомобилей | >0.85 |
| **F1-Score** | Гармоническое среднее P и R | >0.87 |
| **True Positives** | Правильные детекции | Максимизировать |
| **False Positives** | Ложные срабатывания | Минимизировать |
| **False Negatives** | Пропущенные автомобили | Минимизировать |

### Метрики производительности

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **Avg Inference Time** | Среднее время на кадр | <15 ms |
| **FPS** | Кадров в секунду | >30 |
| **Min/Max Inference** | Диапазон времени | Стабильность |
| **GPU Memory** | Используемая память | <2 GB |

---

## Технические детали

### COCO Аннотации

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "0_0.jpg",
      "width": 1024,
      "height": 768,
      "car_id": "0",
      "brand": "vaz",
      "model": "granta",
      "color": "000000"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [51, 38, 922, 692],
      "area": 638024,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "car",
      "supercategory": "vehicle"
    }
  ]
}
```

### Метрики вычисления

**IoU (Intersection over Union):**
```
IoU = Area(Prediction ∩ Ground Truth) / Area(Prediction ∪ Ground Truth)
```

**Соответствие детекции:**
- IoU ≥ 0.5 → True Positive
- IoU < 0.5 → False Positive
- Нет соответствия → False Negative

---

## Заключение

✅ Датасет YMAD успешно подготовлен для оценки TrafficCamNet  
✅ Созданы все необходимые инструменты и документация  
✅ Готова инфраструктура для запуска оценки на Jetson  

**Следующий шаг:** Запуск оценки TrafficCamNet на Jetson устройстве или baseline оценка с YOLOv8.

---

**Версия документа:** 1.0  
**Дата создания:** 14 января 2026  
**Статус:** ✅ Завершено
