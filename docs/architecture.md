# Архитектура системы CarCV

**Дата последнего обновления:** 23 января 2026  
**Версия:** 1.2.0

---

## Обзор системы

CarCV - это система компьютерного зрения для анализа транспортных средств на базе NVIDIA Jetson Orin Nano 8GB. Система использует DeepStream SDK для обработки видеопотоков и набор нейросетей для детекции и классификации различных атрибутов автомобилей.

### Основные компоненты

```
┌─────────────────────────────────────────────────────────────┐
│                     CarCV System                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Video Stream (RTSP/USB/CSI Camera)                  │
│     │                                                        │
│     ├──> DeepStream Pipeline                                │
│     │    ├──> PGIE: TrafficCamNet (Vehicle Detection)       │
│     │    ├──> SGIE1: VehicleMakeNet (Brand Classification)  │
│     │    ├──> SGIE2: VehicleTypeNet (Type Classification)   │
│     │    ├──> SGIE3: LPDNet (License Plate Detection)       │
│     │    └──> SGIE4: FaceDetect (Face Detection)            │
│     │                                                        │
│     └──> Python Service                                     │
│          ├──> LPR_STN_PRE_POST (OCR)                        │
│          └──> bae_model_f3 (Color Recognition)              │
│                                                              │
│  Output: SQLite Database + REST API                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Компоненты системы

### 1. TrafficCamNet (PGIE - Primary Detector)

**Задача:** Детекция транспортных средств в кадре

```yaml
Architecture: ResNet-18 (pruned)
Format: TensorRT FP16
Input Size: 960×544×3 BGR
Output Classes: 4 (car, person, bike, sign) — используется только car
File: models/baseline/resnet18_trafficcamnet_fp16.engine

Current Metrics:
  Precision: 0.92-0.95
  Recall: 0.88-0.92
  F1-Score: 0.91
  Inference Time: 8-10 ms

Status: ✅ Production
Priority: Critical (основной детектор)
```

### 2. VehicleMakeNet (SGIE1 - Brand Classifier)

**Задача:** Классификация марки автомобиля

```yaml
Architecture: NVIDIA VehicleMakeNet
Format: TensorRT FP16
Output Classes: 20 brands (Acura, Audi, BMW, Chevrolet, Chrysler, Dodge, 
                          Ford, GMC, Honda, Hyundai, Infiniti, Jeep, Kia, 
                          Lexus, Mazda, Mercedes, Nissan, Subaru, Toyota, Volkswagen)
File: models/baseline/vehiclemakenet.engine

Current Metrics:
  Top-1 Accuracy: 0.76
  Top-3 Accuracy: 0.92
  Inference Time: 4-5 ms

Status: ✅ Production
Priority: Medium
```

### 3. VehicleTypeNet (SGIE2 - Type Classifier)

**Задача:** Классификация типа кузова автомобиля

```yaml
Architecture: NVIDIA VehicleTypeNet
Format: TensorRT FP16
Output Classes: 6 (coupe, largevehicle, sedan, suv, truck, van)
File: models/baseline/vehicletypenet.engine

Current Metrics:
  Accuracy: 0.88
  Inference Time: 3-4 ms

Status: ✅ Production
Priority: Medium
```

### 4. LPDNet (SGIE3 - License Plate Detector)

**Задача:** Детекция номерных знаков на автомобиле

```yaml
Architecture: NVIDIA LPDNet
Format: TensorRT FP16
Output: Bounding boxes
File: models/baseline/lpdnet.engine

Current Metrics:
  Recall: 0.85-0.90
  Inference Time: 2-3 ms

Status: ✅ Production
Priority: High (необходим для OCR)
```

### 5. FaceDetect (SGIE4 - Face Detector)

**Задача:** Детекция лиц водителя и пассажиров

```yaml
Architecture: NVIDIA FaceDetect / RetinaFace
Format: TensorRT FP16
Output: Bounding boxes
File: models/baseline/facedetect.engine

Current Metrics:
  Not documented

Status: ⚠️ Production (требуется оценка)
Priority: Low
```

### 6. LPR_STN_PRE_POST (OCR Engine) ⭐

**Задача:** Распознавание текста российских номерных знаков

**ОБНОВЛЕНО 14.01.2026: Baseline evaluation завершена**

```yaml
Architecture: STN + CNN + Bi-LSTM + CTC
Format: ONNX Runtime GPU
Input Size: 188×48×3 RGB
Alphabet: 23 символа (0-9, A,B,E,K,M,H,O,P,C,T,Y,X,-)
File: models/baseline/LPR_STN_PRE_POST.onnx
Size: 1.26 MB

Baseline Metrics (проверено на 4893 образцах):
  Character Accuracy: 0.9944 (99.44%)
  Full Plate Accuracy: 0.9875 (98.75%)
  Character Error Rate: 0.17%
  Error Rate: 1.25% (61 из 4893)
  Inference Time: 5.04 ms (CPU)
  Provider: CPUExecutionProvider

Target Metrics:
  Character Accuracy: >0.90 ✅ ВЫПОЛНЕНО
  Full Plate Accuracy: >0.80 ✅ ВЫПОЛНЕНО

Status: ✅ Production (ЗНАЧИТЕЛЬНО ПРЕВЫШАЕТ ТРЕБОВАНИЯ)
Priority: Critical
Last Evaluated: 14.01.2026
Dataset: autoriaNumberplateOcrRu-2021-09-01/val
```

**Архитектура LPR_STN_PRE_POST:**

```
Input Image (188×48×3 RGB)
    │
    ├──> [STN] Spatial Transformer Network
    │    └──> Выравнивание и нормализация номера
    │
    ├──> [CNN Backbone] Convolutional Layers
    │    └──> Извлечение визуальных признаков
    │
    ├──> [Bi-LSTM] Bidirectional LSTM
    │    └──> Последовательное кодирование символов
    │
    └──> [CTC Decoder] Connectionist Temporal Classification
         └──> Декодирование последовательности символов
              Output: Text String (например, "A123BC77")
```

**Preprocessing:**
1. Resize изображения до 188×48 pixels
2. Конвертация BGR → RGB
3. Формат: uint8 (без нормализации)
4. Channels-last format: (1, H, W, C)

**Postprocessing (CTC Decode):**
1. Удаление повторяющихся символов
2. Удаление blank tokens (класс #23)
3. Удаление дефисов (используются моделью как разделители)
4. Результат: строка из 8-9 символов

**Известные проблемы:**
- Путаница цифры "1" с {9, 6, 2, 3, 5}
- Путаница латиницы с цифрами: A↔1, B↔8, O↔0
- Редкие пропуски первого/последнего символа
- Тестирование проводилось на CPU (без GPU ускорения)

**Рекомендации:**
- Протестировать с CUDAExecutionProvider для оценки ускорения
- Обучение новых моделей (PARSeq, CRNN) имеет низкий приоритет
- Модель одобрена для production без изменений

**См. также:**
- Отчёт: `docs/experiments/lpr_stn_baseline_evaluation.md`
- Ноутбук: `notebooks/3.6_LPR_STN_PRE_POST_Baseline_Evaluation.ipynb`
- Результаты: `results/baseline/lpr_stn/`

---

### 7. bae_model_f3 (Color Recognition)

**Задача:** Распознавание цвета автомобиля

```yaml
Architecture: ResNet-based CNN
Format: ONNX Runtime GPU
Input Size: 384×384×3 RGB
Output Classes: 15 (beige, black, blue, brown, gold, green, grey,
                    orange, pink, purple, red, silver, tan, white, yellow)
File: models/baseline/bae_model_f3.onnx

Current Metrics:
  Overall Accuracy: 0.84
  Best Classes (>90%): black, white, red, blue
  Challenging (<80%): beige, tan, gold, silver
  Inference Time: 15 ms

Status: ✅ Production
Priority: Medium
```

---

## Pipeline Flow

### DeepStream Pipeline

```yaml
1. Video Input
   ├─ Source: RTSP/USB/CSI Camera
   ├─ Resolution: 1920×1080 (Full HD)
   └─ FPS: 30

2. Primary Detection (PGIE)
   ├─ Model: TrafficCamNet
   ├─ Action: Detect all vehicles in frame
   └─ Output: Bounding boxes for cars

3. Secondary Classifiers (SGIE)
   For each detected vehicle:
   ├─ SGIE1: VehicleMakeNet → Brand
   ├─ SGIE2: VehicleTypeNet → Body Type
   ├─ SGIE3: LPDNet → License Plate Detection
   └─ SGIE4: FaceDetect → Face Detection

4. Output
   ├─ Metadata attached to frame
   └─ Crops saved for Python processing
```

### Python Service Pipeline

```yaml
1. Monitor SQLite Database
   └─ Wait for new vehicle detections with plate crops

2. For each plate crop:
   ├─ LPR_STN_PRE_POST → OCR Recognition
   │  ├─ Resize to 188×48
   │  ├─ Inference (5.04 ms)
   │  └─ CTC Decode → Plate Number
   └─ Update database with plate number

3. For each vehicle crop:
   ├─ bae_model_f3 → Color Recognition
   │  ├─ Resize to 384×384
   │  ├─ Inference (~15 ms)
   │  └─ Classify Color
   └─ Update database with color

4. REST API
   └─ Serve results via HTTP endpoints
```

---

## Производительность системы

### Целевые метрики

```yaml
End-to-End Performance:
  FPS: ≥30
  Latency: ≤50 ms
  Power Consumption: ≤25W
  Uptime: 99.5% (72h continuous operation)

Per-Model Performance:
  TrafficCamNet: 8-10 ms
  VehicleMakeNet: 4-5 ms
  VehicleTypeNet: 3-4 ms
  LPDNet: 2-3 ms
  FaceDetect: 3-5 ms
  LPR_STN_PRE_POST: 5.04 ms (CPU), ожидается <3 ms (GPU)
  bae_model_f3: ~15 ms

Total Pipeline Latency: ~40-50 ms per vehicle
```

### Resource Usage

```yaml
Hardware: NVIDIA Jetson Orin Nano 8GB
  CPU: 6-core Arm Cortex-A78AE
  GPU: 1024-core NVIDIA Ampere (1GHz)
  Memory: 8GB LPDDR5
  Power: 7W-25W (configurable)

Software Stack:
  OS: JetPack 5.x (Ubuntu 20.04)
  CUDA: 11.4+
  TensorRT: 8.5+
  DeepStream SDK: 6.2+
  ONNX Runtime: 1.12+ (GPU)
  Python: 3.8+
```

---

## Хранение данных

### SQLite Database Schema

```sql
-- Основная таблица детекций
CREATE TABLE data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,                -- Unix timestamp
    frame_number INTEGER,             -- Номер кадра
    
    -- Vehicle Detection
    bbox_x REAL,                      -- Bounding box coordinates
    bbox_y REAL,
    bbox_width REAL,
    bbox_height REAL,
    confidence REAL,                  -- Detection confidence
    
    -- Classification Results
    car_make TEXT,                    -- Brand (VehicleMakeNet)
    car_type TEXT,                    -- Body type (VehicleTypeNet)
    car_color TEXT,                   -- Color (bae_model_f3)
    
    -- License Plate
    lp_bbox_x REAL,                   -- LP bounding box
    lp_bbox_y REAL,
    lp_bbox_width REAL,
    lp_bbox_height REAL,
    lp_number TEXT,                   -- OCR result (LPR_STN_PRE_POST)
    lp_confidence REAL,               -- OCR confidence
    
    -- Face Detection
    face_count INTEGER,               -- Number of faces detected
    
    -- Crops (paths to saved images)
    vehicle_crop_path TEXT,
    plate_crop_path TEXT
);

-- Индексы для быстрого поиска
CREATE INDEX idx_timestamp ON data(timestamp);
CREATE INDEX idx_lp_number ON data(lp_number);
CREATE INDEX idx_car_make ON data(car_make);
```

---

## REST API

### Endpoints

```yaml
GET /api/vehicles
  Description: Получить список всех детекций
  Query Params:
    - limit: количество записей (default: 100)
    - offset: смещение (default: 0)
    - since: timestamp начала выборки
  Response: JSON array of vehicle objects

GET /api/vehicles/{id}
  Description: Получить информацию о конкретной детекции
  Response: JSON object with vehicle details

GET /api/search/plate/{plate_number}
  Description: Поиск по номеру
  Response: JSON array of matching vehicles

GET /api/stats
  Description: Статистика системы
  Response:
    - total_vehicles: общее количество
    - detections_per_hour: детекций в час
    - top_brands: топ марок
    - top_colors: топ цветов

GET /health
  Description: Проверка работоспособности
  Response: {"status": "ok", "uptime": seconds}
```

---

## Конфигурация DeepStream

### Основной конфигурационный файл

**Путь:** `configs/dstest2_pgie_config.txt`

```ini
[property]
# TrafficCamNet (PGIE)
model-engine-file=models/baseline/resnet18_trafficcamnet_fp16.engine
net-scale-factor=0.0039215697906911373
offsets=103.939;116.779;123.68
model-color-format=0  # RGB
batch-size=1
network-mode=2  # FP16
interval=0
gie-unique-id=1
output-blob-names=output_bbox/BiasAdd;output_cov/Sigmoid
parse-bbox-func-name=NvDsInferParseCustomResnet
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
```

### SGIE Конфигурации

- `configs/dstest2_sgie1_config.txt` - VehicleMakeNet
- `configs/dstest2_sgie2_config.txt` - VehicleTypeNet  
- `configs/dstest2_sgie3_config.txt` - LPDNet
- `configs/dstest2_sgie4_config.txt` - FaceDetect

---

## Развёртывание

### Директории проекта

```
/opt/apps/vehicle-analyzer/
├── models/
│   └── baseline/
│       ├── resnet18_trafficcamnet_fp16.engine
│       ├── vehiclemakenet.engine
│       ├── vehicletypenet.engine
│       ├── lpdnet.engine
│       ├── facedetect.engine
│       ├── LPR_STN_PRE_POST.onnx
│       └── bae_model_f3.onnx
├── configs/
│   ├── dstest2_pgie_config.txt
│   ├── dstest2_sgie1_config.txt
│   ├── dstest2_sgie2_config.txt
│   ├── dstest2_sgie3_config.txt
│   └── dstest2_sgie4_config.txt
├── deepstream-vehicle-analyzer  # DeepStream app
├── lp_and_color_recognition_prod.py  # Python service
└── database.db  # SQLite database
```

### Запуск системы

```bash
# 1. Запуск DeepStream pipeline
cd /opt/apps/vehicle-analyzer
./deepstream-vehicle-analyzer rtsp rtsp://camera/stream database.db false

# 2. Запуск Python сервиса (в отдельном терминале)
python3 lp_and_color_recognition_prod.py database.db --api-port 8080

# 3. Проверка работы
curl http://localhost:8080/health
curl http://localhost:8080/api/stats
```

---

## Мониторинг

### Метрики для отслеживания

```yaml
System Health:
  - FPS (frames per second)
  - Latency (end-to-end processing time)
  - GPU utilization
  - CPU utilization
  - Memory usage
  - Power consumption
  - Uptime

Detection Quality:
  - Detection rate (vehicles per minute)
  - OCR success rate (plates recognized / plates detected)
  - Classification confidence scores
  - False positive rate
  - False negative rate (if ground truth available)

Python Service:
  - Queue size (pending crops)
  - Processing time per crop
  - Error rate
  - API response time
```

### Логирование

```yaml
DeepStream Logs:
  Path: /var/log/deepstream-vehicle-analyzer.log
  Level: INFO
  Rotation: Daily

Python Service Logs:
  Path: /var/log/lp_color_service.log
  Level: INFO
  Rotation: Daily

System Logs:
  Path: /var/log/syslog
  Monitor: GPU errors, memory issues, crashes
```

---

## Обновления и история изменений

### 2026-01-14
- ✅ Завершена baseline evaluation для LPR_STN_PRE_POST
- ✅ Обновлены метрики модели (Character Acc: 99.44%, Plate Acc: 98.75%)
- ✅ Создан отчёт по экспериментам: `docs/experiments/lpr_stn_baseline_evaluation.md`
- ✅ Модель одобрена для production без изменений
- ✅ Обновлён план работы: `docs/rules/plan.md`

### 2026-01-12
- Создан план обучения, оценки качества и инференса нейросетей
- Документированы целевые метрики для всех 7 моделей
- Определены приоритеты работ (критические, высокие, средние, низкие)

---

## Следующие шаги

### Запланированные работы (по приоритетам)

#### Приоритет 1 (Critical)
1. **TrafficCamNet** - Baseline evaluation
   - Подготовка тестового датасета с российскими дорогами
   - Оценка текущей производительности
   - Сравнение с альтернативными архитектурами (YOLOv8, EfficientDet)

2. ~~**LPR_STN_PRE_POST** - Baseline evaluation~~ ✅ ЗАВЕРШЕНО
   - ~~Тестирование на autoriaNumberplateOcrRu-2021-09-01~~
   - ~~Измерение метрик качества~~
   - ~~Анализ ошибок~~

#### Приоритет 2 (High)
3. **LPDNet** - Baseline evaluation
4. **bae_model_f3** - Baseline evaluation

#### Приоритет 3 (Medium)
5. **VehicleMakeNet** - Baseline evaluation
6. **VehicleTypeNet** - Baseline evaluation

#### Приоритет 4 (Low)
7. **FaceDetect** - Baseline evaluation

---

## Контакты и документация

**Репозиторий:** `/home/user/CarCV`

**Документация:**
- План работы: `docs/rules/plan.md`
- Системный дизайн: `docs/system-design/ML_System_Design_Document.md`
- Архитектурные правила: `docs/rules/arch-rules.md`
- Эксперименты: `docs/experiments/`
- Датасеты: `docs/datasets/`

**Ноутбуки:**
- `notebooks/3.6_LPR_STN_PRE_POST_Baseline_Evaluation.ipynb` ✅

**Результаты:**
- `results/baseline/lpr_stn/` ✅
- `results/baseline/trafficcamnet/` (в планах)
- `results/baseline/vehiclemakenet/` (в планах)

---

**Версия документа:** 1.1.0  
**Дата последнего обновления:** 14 января 2026  
**Статус:** ✅ Актуально
