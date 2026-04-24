# CarCV - Computer Automotive Recognition System

> Интеллектуальная бортовая система компьютерного зрения для распознавания и анализа транспортных средств на платформе NVIDIA Jetson

[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)](https://github.com)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin%20Nano-green)](https://developer.nvidia.com/embedded/jetson-orin)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

---

## 📋 Содержание

- [Обзор](#-обзор)
- [Архитектура](#-архитектура)
- [Модели ML](#-модели-ml)
- [Технические характеристики](#-технические-характеристики)
- [Текущее состояние](#-текущее-состояние)
- [Установка и запуск](#-установка-и-запуск)
- [Документация](#-документация)
- [Roadmap](#-roadmap)

---

## 🎯 Обзор

**CarCV** — это автономная система видеоаналитики реального времени, разработанная для установки на борту автомобиля. Система выполняет комплексный анализ транспортных средств:

- ✅ **Детекция** автомобилей в видеопотоке (30+ FPS)
- ✅ **Распознавание** номерных знаков (OCR с точностью 98.75%)
- ✅ **Классификация** марки, типа кузова и цвета
- ✅ **Трекинг** объектов с присвоением уникальных ID
- ✅ **Детекция лиц** водителей и пассажиров
- ✅ **REST API** для интеграции с внешними системами

### 🎭 Применение

- 🚔 **Правоохранительные органы** — мониторинг розыскных автомобилей
- 🏢 **Контроль доступа** — автоматическая идентификация на КПП
- 🅿️ **Парковочные системы** — учёт въезда/выезда транспорта
- 📊 **Аналитика трафика** — статистика по типам и маркам ТС
- 🚛 **Логистика** — отслеживание грузового транспорта

---

## 🏗️ Архитектура

### Общая схема системы

```
┌─────────────────────────────────────────────────────────────┐
│                        CarCV System                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Video Input (1080p/30fps)                                  │
│       │                                                      │
│       ├─► DeepStream Pipeline (NVIDIA GPU)                  │
│       │   ├─ PGIE: TrafficCamNet (Vehicle Detection)        │
│       │   ├─ SGIE1: VehicleMakeNet (Brand)                  │
│       │   ├─ SGIE2: VehicleTypeNet (Body Type)              │
│       │   ├─ SGIE3: LPDNet (License Plate Detection)        │
│       │   └─ SGIE4: FaceDetect (Face Detection)             │
│       │                                                      │
│       └─► Python Service (ONNX Runtime)                     │
│           ├─ LPR_STN_PRE_POST (OCR Recognition) ⭐          │
│           └─ bae_model_f3 (Color Recognition)               │
│                                                              │
│  Output: SQLite DB + REST API + Web UI                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Компоненты

| Компонент | Технология | Назначение |
|-----------|-----------|------------|
| **DeepStream SDK** | C/GStreamer | Видеопайплайн и inference на GPU |
| **TensorRT** | FP16/INT8 | Ускорение нейросетей |
| **ONNX Runtime** | GPU/CPU | Inference для OCR и цвета |
| **SQLite** | Database | Хранение результатов детекций |
| **REST API** | Python/HTTP | Интеграция с внешними системами |
| **Web UI** | HTML/JS | Мониторинг в реальном времени |

---

## 🤖 Модели ML

Система использует **7 специализированных нейросетей**:

### 1️⃣ TrafficCamNet (Primary Detector)

```yaml
Задача: Детекция транспортных средств
Архитектура: ResNet-18 (pruned)
Формат: TensorRT FP16
Входной размер: 960×544×3 BGR
Метрики:
  - Precision: 0.92-0.95
  - Recall: 0.88-0.92
  - F1-Score: 0.91
  - Inference: 8-10 ms
Статус: ✅ Production
Приоритет: 🔴 Critical
```

### 2️⃣ VehicleMakeNet (Brand Classifier)

```yaml
Задача: Классификация марки автомобиля
Выходные классы: 20 брендов (Acura, Audi, BMW, ...)
Метрики:
  - Top-1 Accuracy: 0.76
  - Top-3 Accuracy: 0.92
  - Inference: 4-5 ms
Статус: ✅ Production
```

### 3️⃣ VehicleTypeNet (Type Classifier)

```yaml
Задача: Классификация типа кузова
Выходные классы: 6 типов (sedan, suv, truck, van, coupe, largevehicle)
Метрики:
  - Accuracy: 0.88
  - Inference: 3-4 ms
Статус: ✅ Production
```

### 4️⃣ LPDNet (License Plate Detector)

```yaml
Задача: Детекция номерных знаков на автомобиле
Метрики:
  - Recall: 0.85-0.90
  - Inference: 2-3 ms
Статус: ✅ Production
Приоритет: 🟡 High
```

### 5️⃣ FaceDetect (Face Detector)

```yaml
Задача: Детекция лиц водителя и пассажиров
Статус: ⚠️ Production (требуется оценка)
Приоритет: 🟢 Low
```

### 6️⃣ LPR_STN_PRE_POST (OCR Engine) ⭐ ЛУЧШИЙ РЕЗУЛЬТАТ

```yaml
Задача: Распознавание текста российских номерных знаков
Архитектура: STN + CNN + Bi-LSTM + CTC
Формат: ONNX Runtime GPU/CPU
Входной размер: 188×48×3 RGB
Алфавит: 23 символа (0-9, A,B,E,K,M,H,O,P,C,T,Y,X,-)

🏆 Baseline Metrics (проверено на 4893 образцах):
  - Character Accuracy: 99.44% ✅ (цель: >90%)
  - Full Plate Accuracy: 98.75% ✅ (цель: >80%)
  - Character Error Rate: 0.17%
  - Ошибок: 61 из 4893 (1.25%)
  - Inference Time: 5.04 ms (CPU)

Статус: ✅ Production - ПРЕВЫШАЕТ ТРЕБОВАНИЯ
Приоритет: 🔴 Critical
Последнее тестирование: 14.01.2026
Dataset: autoriaNumberplateOcrRu-2021-09-01/val
```

**Архитектура LPR_STN:**

```
Input (188×48×3) → STN (выравнивание) → CNN (features) → 
Bi-LSTM (sequence) → CTC Decoder → "A123BC77"
```

### 7️⃣ bae_model_f3 (Color Recognition)

```yaml
Задача: Распознавание цвета автомобиля
Архитектура: ResNet-based CNN
Формат: ONNX Runtime GPU
Входной размер: 384×384×3 RGB
Выходные классы: 15 цветов (black, white, red, blue, silver, ...)
Метрики:
  - Overall Accuracy: 0.84
  - Best Classes (>90%): black, white, red, blue
  - Challenging (<80%): beige, tan, gold, silver
  - Inference: 15 ms
Статус: ✅ Production
```

---

## ⚙️ Технические характеристики

### Аппаратная платформа

```
NVIDIA Jetson Orin Nano 8GB
├─ GPU: 1024 CUDA cores (Ampere)
├─ CPU: 6-core ARM Cortex-A78AE @ 2.0 GHz
├─ RAM: 8GB LPDDR5 (unified memory)
├─ AI Performance: 40 TOPS (INT8)
└─ Power: 7W / 15W / 25W modes
```

### Производительность

| Метрика | Целевое значение | Текущее состояние |
|---------|------------------|-------------------|
| **FPS** (1080p input) | ≥30 | ✅ 30+ |
| **End-to-end latency** | <50 ms | ✅ ~40-50 ms |
| **GPU utilization** | 70-85% | ✅ Оптимально |
| **RAM usage** | <6 GB | ✅ В норме |
| **Power consumption** | <25W | ✅ 18-25W |
| **Uptime** | 99% (72h) | 🔄 Тестируется |

### Программный стек

```yaml
OS: Ubuntu 22.04 ARM64 (JetPack 6.2)
CUDA: 12.6
TensorRT: 10.3.0
DeepStream SDK: 7.1
ONNX Runtime: 1.16.0
GStreamer: 1.0
OpenCV: 4.8.0
Python: 3.8+
```

---

## 📊 Текущее состояние

### ✅ Завершённые задачи (Milestone 1)

- [x] **LPR_STN_PRE_POST Baseline Evaluation** (14.01.2026)
  - Протестировано на 4893 образцах
  - Метрики: 99.44% char accuracy, 98.75% plate accuracy
  - Модель одобрена для production БЕЗ ИЗМЕНЕНИЙ
  - Документация: `docs/experiments/lpr_stn_baseline_evaluation.md`

- [x] **Подготовка инфраструктуры для тестирования** (14.01.2026)
  - Каталог российских датасетов: `docs/datasets/russian_vehicle_datasets.md`
  - Руководство по тестированию: `docs/datasets/README_TESTING_GUIDE.md`
  - Скрипты для быстрого старта: `scripts/quick_test_baseline.sh`

- [x] **Системная документация**
  - Архитектура системы: `docs/architecture.md`
  - ML System Design: `docs/system-design/ML_System_Design_Document.md`
  - План работы: `docs/rules/plan.md`

### 🔄 В работе (Current Sprint)

- [ ] **TrafficCamNet Baseline Evaluation** (Приоритет 1)
  - Подготовка тестового датасета (2000+ кадров)
  - Тестирование на российских дорогах
  - Сравнение с альтернативными архитектурами (YOLOv8, EfficientDet)

### 📅 Запланировано (Next Milestones)

**Q1 2026:**
- [ ] Baseline evaluation остальных моделей (LPDNet, bae_model_f3, VehicleMakeNet, VehicleTypeNet, FaceDetect)
- [ ] Prometheus metrics + Grafana dashboards
- [ ] Structured logging (JSON format)
- [ ] Unit tests (critical paths)

**Q2 2026:**
- [ ] Multi-camera support (2-3 streams)
- [ ] Webhook notifications
- [ ] Cloud backup (S3 compatible)
- [ ] INT8 quantization для всех моделей

**Q3-Q4 2026:**
- [ ] Additional LP alphabets (EU, CN)
- [ ] Vehicle damage detection
- [ ] Speed estimation
- [ ] Fleet management integration

---

## 🚀 Установка и запуск

### Предварительные требования

```bash
# Аппаратура
- NVIDIA Jetson Orin Nano 8GB (или выше)
- Camera (CSI-2 / USB / RTSP)
- NVMe SSD 128+ GB
- Power supply 5V/4A

# Программное обеспечение
- JetPack 6.2 (Ubuntu 22.04)
- DeepStream SDK 7.1
- Python 3.8+
```

### Установка

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd CarCV

# 2. Установить зависимости
sudo apt update
sudo apt install -y deepstream-7.1 python3-pip python3-opencv
pip3 install -r requirements.txt

# 3. Скомпилировать приложение
make CUDA_VER=12.6 clean all

# 4. Создать директории
mkdir -p lp_images car_images face_images results/baseline
```

### Запуск

```bash
# Вариант 1: Из видеофайла
./deepstream-vehicle-analyzer file sample.mp4 my.db false &
python3 lp_and_color_recognition_prod.py my.db --api-port 8080 &

# Вариант 2: Из RTSP потока
./deepstream-vehicle-analyzer rtsp rtsp://camera/stream my.db false &
python3 lp_and_color_recognition_prod.py my.db --api-port 8080 &

# Вариант 3: Из USB камеры
./deepstream-vehicle-analyzer usb /dev/video0 my.db false &
python3 lp_and_color_recognition_prod.py my.db --api-port 8080 &

# Проверка работы
curl http://localhost:8080/api/health
# {"status": "ok", "uptime": 123}
```

### Web Interface

Открыть в браузере: `http://<jetson-ip>:8080`

Функционал:
- ✅ Лента детекций в реальном времени
- ✅ Добавление паттернов номеров для отслеживания
- ✅ Просмотр списка отслеживаемых номеров
- ✅ Автообновление каждые 2-5 секунд

---

## 📚 Документация

### Основные документы

| Документ | Описание |
|----------|----------|
| [`docs/architecture.md`](docs/architecture.md) | Архитектура системы (7 моделей, pipeline, API) |
| [`docs/system-design/ML_System_Design_Document.md`](docs/system-design/ML_System_Design_Document.md) | ML System Design (требования, развёртывание) |
| [`docs/rules/plan.md`](docs/rules/plan.md) | План обучения и оценки моделей |
| [`docs/rules/arch-rules.md`](docs/rules/arch-rules.md) | Архитектурные правила |

### Датасеты и тестирование

| Документ | Описание |
|----------|----------|
| [`docs/datasets_and_models_overview.md`](docs/datasets_and_models_overview.md) | **НОВЫЙ:** Обзор открытых датасетов и SOTA моделей 2025-2026 ⭐ |
| [`docs/datasets/SUMMARY.md`](docs/datasets/SUMMARY.md) | Резюме подготовки датасетов |
| [`docs/datasets/russian_vehicle_datasets.md`](docs/datasets/russian_vehicle_datasets.md) | Каталог российских датасетов |
| [`docs/datasets/README_TESTING_GUIDE.md`](docs/datasets/README_TESTING_GUIDE.md) | Руководство по тестированию |

### Эксперименты и результаты

| Документ | Описание |
|----------|----------|
| [`docs/experiments/SUMMARY_14_01_2026.md`](docs/experiments/SUMMARY_14_01_2026.md) | Сводка обновлений 14.01.2026 |
| [`docs/experiments/lpr_stn_baseline_evaluation.md`](docs/experiments/lpr_stn_baseline_evaluation.md) | Детальный отчёт по LPR_STN ⭐ |

### Notebooks

| Notebook | Описание |
|----------|----------|
| [`notebooks/3.6_LPR_STN_PRE_POST_Baseline_Evaluation.ipynb`](notebooks/3.6_LPR_STN_PRE_POST_Baseline_Evaluation.ipynb) | Baseline evaluation LPR_STN |

---

## 🗺️ Roadmap

### 2026 Q1 - Foundation ✅ (частично завершено)

- [x] LPR_STN_PRE_POST baseline (99.44% accuracy)
- [x] Документация системы
- [x] Инфраструктура для тестирования
- [ ] TrafficCamNet baseline evaluation
- [ ] Остальные модели baseline evaluation

### 2026 Q2 - Optimization

- [ ] Multi-camera support (2-3 streams)
- [ ] INT8 quantization (2× speed boost)
- [ ] Prometheus + Grafana monitoring
- [ ] Cloud backup integration
- [ ] Mobile-friendly Web UI

### 2026 Q3 - Features

- [ ] Additional LP alphabets (EU, China)
- [ ] Vehicle damage detection
- [ ] Speed estimation
- [ ] Direction detection
- [ ] Docker containerization

### 2026 Q4 - Production

- [ ] Fleet management integration
- [ ] Analytics dashboard
- [ ] A/B testing framework
- [ ] Model auto-update mechanism

---

## 📈 Метрики проекта

### Модели ML

| Модель | Precision/Accuracy | Recall | Latency | Статус |
|--------|-------------------|--------|---------|--------|
| TrafficCamNet | 0.92-0.95 | 0.88-0.92 | 8-10 ms | ✅ Production |
| VehicleMakeNet | 0.76 (Top-1) | - | 4-5 ms | ✅ Production |
| VehicleTypeNet | 0.88 | - | 3-4 ms | ✅ Production |
| LPDNet | - | 0.85-0.90 | 2-3 ms | ✅ Production |
| FaceDetect | - | - | 3-5 ms | ⚠️ Needs eval |
| **LPR_STN** | **99.44%** (char) | - | **5.04 ms** | ✅ **Excellent** |
| bae_model_f3 | 0.84 | - | 15 ms | ✅ Production |

### Система (целевые метрики)

- **FPS:** ≥30 (1080p input) ✅
- **Latency:** <50 ms ✅
- **Power:** <25W ✅
- **Uptime:** 99% (72h) 🔄

---

## 🏆 Ключевые достижения

1. ⭐ **LPR_STN_PRE_POST - 99.44% Character Accuracy**
   - Превышает целевые метрики на 9.4%
   - Только 61 ошибка из 4893 образцов (1.25%)
   - Одобрена для production без изменений

2. 🚀 **Real-time Performance - 30 FPS @ 1080p**
   - End-to-end latency ~40-50 ms
   - Работает на edge device без облака

3. 📊 **Комплексный анализ ТС**
   - 7 специализированных нейросетей
   - Детекция, распознавание, классификация, трекинг

4. 🔧 **Production-ready Architecture**
   - DeepStream + TensorRT для GPU
   - ONNX Runtime для Python models
   - SQLite + REST API + Web UI

---

## 👥 Команда и контрибуция

**Разработчики:** CARS Development Team

**Правила контрибуции:**
- Следовать архитектурным правилам: `docs/rules/arch-rules.md`
- Все изменения ML моделей документировать в `docs/experiments/`
- Обновлять `docs/architecture.md` при изменении системы
- Обновлять этот README при добавлении новых feature

---

## 📝 Лицензия

Proprietary © 2026 CARS Team

---

## 📞 Контакты

**Репозиторий:** `/home/user/CarCV`  
**Документация:** `docs/`  
**Эксперименты:** `docs/experiments/`  
**Результаты:** `results/baseline/`

---

## 🔖 Версия

**Версия README:** 1.0.1  
**Дата обновления:** 17 января 2026  
**Последнее изменение:** Добавлен обзор открытых датасетов и моделей для улучшения детекторов

---

<p align="center">
  <strong>CarCV</strong> - Intelligent Vehicle Recognition on the Edge
</p>

<p align="center">
  Powered by NVIDIA Jetson | DeepStream SDK | TensorRT | ONNX Runtime
</p>
