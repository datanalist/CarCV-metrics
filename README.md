# CarCV-metrics — валидация ML-стека системы CarCV

> Репозиторий **оценки и валидации** моделей бортовой системы видеоаналитики **CarCV** (распознавание транспортных средств в реальном времени на NVIDIA Jetson Orin Nano 8GB).

[![Status](https://img.shields.io/badge/Campaign-2%20PASS%20%7C%204%20FAIL%20%7C%201%20UNDEF-orange)](#-результаты-валидации)
[![Platform](https://img.shields.io/badge/Target-NVIDIA%20Jetson%20Orin%20Nano-green)](https://developer.nvidia.com/embedded/jetson-orin)
[![Metrics](https://img.shields.io/badge/Metrics-measured%20%7C%20honest-blue)](#-результаты-валидации)

---

## 📋 Содержание

- [Что это за репозиторий](#-что-это-за-репозиторий)
- [Конвейер CarCV](#-конвейер-carcv)
- [Стек моделей](#-стек-моделей)
- [Результаты валидации](#-результаты-валидации)
- [Датасеты](#-датасеты)
- [Лицензии и ограничения](#-лицензии-и-ограничения)
- [Целевая платформа](#-целевая-платформа)
- [Как запустить валидацию](#-как-запустить-валидацию)
- [Документация](#-документация)
- [Статус кампании и дальнейшие шаги](#-статус-кампании-и-дальнейшие-шаги)

---

## 🎯 Что это за репозиторий

**CarCV** — автономная бортовая система видеоаналитики реального времени: детекция ТС, распознавание номеров (LP Detection + OCR), классификация марки/типа/цвета, детекция лиц, трекинг.

**CarCV-metrics** (этот репозиторий) — **не сам продукт, а его измерительный контур**: воспроизводимая валидация моделей продакшн-стека против открытых датасетов. Цель — получить **честные измеренные метрики** по каждой паре «модель × датасет» с однозначным вердиктом **PASS / FAIL / UNDEF**.

Принципы кампании:

- **Источник истины по моделям** — [`models.md`](models.md); по датасетам — [`datasets.md`](datasets.md). Детальные спецификации — в [`docs/about_models/`](docs/about_models) и [`docs/about_datasets/`](docs/about_datasets).
- **Измеренные метрики** локальной кампании хранятся в `results/<model>/metrics.json` (+ `EXPERIMENT.md`), сводка — в [`results/SUMMARY.md`](results/SUMMARY.md); прежний удалённый архив qudata — в `results_collected/` (ориентир). Пороги pass/fail — в `deploy/evaluation/evaluate.py` (`EVAL_CONFIGS`).
- **FAIL — валидный окончательный результат.** Модель не тюнингуется ради «зелёного» вердикта; провал из-за доменного разрыва фиксируется как факт о применимости.
- **Метрики измерены на x86 GPU через ONNX Runtime** и считаются **верхней границей** точности для TensorRT-движка на Jetson.

> ⚠️ **Legacy-предупреждение.** Прежняя версия этого README и документы `docs/architecture.md` / `docs/system-design/` описывали **старый US-ориентированный стек** (LPDNet, LPR_STN_PRE_POST, US LPRNet) и **аспирационные/непроверенные** числа (TrafficCamNet «P=0.92–0.95», OCR «99.44%», Color «0.84», latency/FPS). Эти значения **не измерялись** в данном репозитории и здесь не воспроизводятся. Актуальный стек — RU/UA-ориентированный (см. ниже).

---

## 🏗️ Конвейер CarCV

```
Video (1080p) ─► PGIE: TrafficCamNet (детекция ТС, DeepStream/TensorRT)
                   │
                   ├─► SGIE1: VehicleMakeNet  (марка, 20 классов)
                   ├─► SGIE2: VehicleTypeNet  (тип кузова, 6 классов)
                   └─► SGIE4: FaceDetect      (лица водителя/пассажиров)

                 Python-сервисы (PyTorch / ONNX Runtime):
                   ├─► nomeroff_lpd  (LP Detection, YOLOv11x + 4 keypoints)  ──► выравнивание
                   ├─► nomeroff_ocr  (OCR RU-номера, CRNN + CTC)
                   └─► bae_model_f3  (цвет кузова, EfficientNet-B3)   [веса вне репозитория]
```

**Ключевая смена стека.** RU/UA-сегмент потребовал замены US-обученных моделей NVIDIA TAO:

- **`nomeroff_lpd`** заменил **US LPDNet** — на RU/UA-пластинах US-детектор консервативен (recall ≈ 0.30 при том же датасете).
- **`nomeroff_ocr`** заменил **US LPRNet** — US-OCR использует латинский алфавит `A–Z`/иной CTC-blank и проваливает RU-номера (char 0.59 / plate 0.06).

`nomeroff_lpd` и `nomeroff_ocr` работают как Python-сервисы (а не узлы DeepStream SGIE), как и сервис цвета `bae_model_f3`.

---

## 🤖 Стек моделей

Источник: [`models.md`](models.md). Полные спецификации — по ссылкам в столбце «Спека».

| Задача | Модель | Источник / архитектура | Роль | Спека |
|--------|--------|------------------------|------|-------|
| Detection | **TrafficCamNet** | NVIDIA TAO · DetectNet_v2, ResNet-18 (pruned) | PGIE — первичный детектор | [📄](docs/about_models/trafficcamnet.md) |
| LP Detection | **nomeroff_lpd** | Nomeroff Net · YOLOv11x + keypoints (4 угла) | Python-сервис (RU/UA) | [📄](docs/about_models/nomeroff_lpd.md) |
| OCR | **nomeroff_ocr** | Nomeroff Net · CRNN ResNet-18 + CTC (RU) | Python-сервис | [📄](docs/about_models/nomeroff_ocr.md) |
| Color | **bae_model_f3** | Кастомная · EfficientNet-B3 (15 цветов) | Python-сервис · *веса вне репо* | [📄](docs/about_models/bae_model_f3.md) |
| Make | **VehicleMakeNet** | NVIDIA TAO · TAO-classifier, ResNet-18 (20 US/EU марок) | SGIE1 | [📄](docs/about_models/vehiclemakenet.md) |
| Type | **VehicleTypeNet** | NVIDIA TAO · TAO-classifier, ResNet-18 (6 типов) | SGIE2 | [📄](docs/about_models/vehicletypenet.md) |
| Face Detection | **FaceDetect** | NVIDIA TAO FaceNet · DetectNet_v2 (1 класс `face`) | SGIE4 | [📄](docs/about_models/facedetect.md) |
| Face embedding | **— (None / Trainable)** | модель не выбрана и не существует | планируемая задача | [📄](docs/about_models/face_embedding.md) |

> **Семейства препроцессинга различаются** и легко путаются (источник «тихой» деградации точности):
> - **DetectNet_v2** (TrafficCamNet, FaceDetect): BGR→RGB, `/255`, без offsets, NCHW.
> - **TAO classifier** (VehicleMakeNet, VehicleTypeNet): **BGR без свопа**, **без `/255`**, offsets `(104,117,124)` по каналам B;G;R, NCHW.
> - **Generic ImageNet** (bae_model_f3): BGR→RGB, `/255`, `(x−mean)/std`, NCHW.
> - **YOLO11 / CRNN** (nomeroff_lpd, nomeroff_ocr): препроцессинг инкапсулирован внутри пайплайна Nomeroff Net.

---

## 📊 Результаты валидации

Каждая модель сопоставлена с целевым датасетом из [`datasets.md`](datasets.md). Метрики измерены локально (`results/<model>/metrics.json` + `EXPERIMENT.md`); пороги — из `eval_*` в `deploy/evaluation/evaluate.py`. Полная сводка — [`results/SUMMARY.md`](results/SUMMARY.md).

**Итог по 7 парам:** **2 PASS / 4 FAIL / 1 UNDEF**. Четыре FAIL и UNDEF — локальная кампания (RTX 3090, onnxruntime 1.24, Python 3.13, `.venv`); **nomeroff_lpd / nomeroff_ocr — PASS на удалённом прогоне qudata** (локальный прогон блокирован упаковкой `modelhub_client`).

| Модель | Задача | Датасет | Метрика (измерено) | Порог | Вердикт |
|--------|--------|---------|---------------------|-------|---------|
| **TrafficCamNet** | Detection | BDD100K val (10 000 img) | car P=0.3909 · R=0.2584 · F1=0.3111 (conf_thr=0.2) | P≥0.90 R≥0.85 F1≥0.87 | ❌ **FAIL** |
| **VehicleTypeNet** | Type | Stanford Cars TRAIN (7577 img; 567 пропущено) | Top-1=0.3549 · Top-3=0.6932 | Top-1≥0.85 | ❌ **FAIL** |
| **VehicleMakeNet** | Make | VMMRdb (243 519 img, 20 марок, `skipped_ood=0`) | Top-1=0.4387 · Top-3=0.6250 | Top-1≥0.70 Top-3≥0.85 | ❌ **FAIL** |
| **bae_model_f3** | Color | MAD-Cars (2000 dedup img, покрытие 13/15) | Top-1=0.6205 · Top-3=0.8395; best_group_min=0.4793; challenging_group_min=0.0 | overall≥0.80, best_group_min≥0.90, challenging_group_min≥0.70 | ❌ **FAIL** (осторожный) |
| **nomeroff_lpd** | LP Detection | AUTO.RIA Detection 2021-05-12 | P=0.9056 · R=0.9221 · F1=0.9138 *(qudata)* | P≥0.70, R≥0.80 | ✅ **PASS** |
| **nomeroff_ocr** | OCR | autoriaNumberplateOcrRu 2021-09-01 / val | char=0.9995 · plate=0.9978 *(qudata)* | char≥0.90, plate≥0.80 | ✅ **PASS** |
| **FaceDetect** | Face Detection | WIDER FACE val | модель недоступна (FaceNet ONNX не получен; код+данные готовы) | AP@0.5≥0.50 (Easy≥0.80/Med≥0.70/Hard≥0.50) | ❓ **UNDEF** |

> Пара **Face embedding** (модель `None / Trainable`) в кампанию из 7 пар не входит — измерять нечего (см. стек моделей выше).

### Что важно понимать о вердиктах

- ✅ **PASS только у RU-моделей** (`nomeroff_lpd`, `nomeroff_ocr`) — они и были введены под RU/UA-домен. На тех же данных US-модели проваливаются: **US LPDNet** R=0.296, **US LPRNet** char=0.59 / plate=0.06. Числа — с **удалённого прогона qudata** (P=0.91 / R=0.92; char 0.9995 / plate 0.9978); локально импорт пока блокирован упаковкой `modelhub_client`.
- ❌ **Четыре FAIL измерены на целевых датасетах и окончательны.** TrafficCamNet × BDD100K (car F1=0.3111), VehicleMakeNet × VMMRdb (Top-1=0.4387), VehicleTypeNet × Stanford Cars (Top-1=0.3549), Color × MAD-Cars (Top-1=0.6205). Все **выше** прежних суррогатов (COCO car F1≈0.064, mad-cars Top-1≈0.083), но порогов не достигают; тюнинг не применялся. Корневая причина — доменный разрыв (US/EU-обучение NGC TAO + суррогатные/шумные метки).
- ❌ **VehicleTypeNet и Color — нюансы меток.** Stanford размечен по make/model (тип кузова выведен по ключевым словам, класс `largevehicle` отсутствует) → шумные суррогатные метки (`truck` 0.6137 / `sedan` 0.2867). У Color маппинг индекс→класс (`COLOR_CLASSES`) непроверяем без оригинального файла меток — вердикт **осторожный** (cautious), не коллапс (`black` 0.897 / `white` 0.794 / `red` 0.730 узнаются уверенно).
- ❓ **FaceDetect — UNDEF (не FAIL).** Эвалуатор `eval_facedetect` + `parse_wider_gt` **реализованы и протестированы** (13 passed), данные WIDER FACE на месте. Не измерено: deployable FaceNet ONNX недоступен (`.etlt` зашифрован, `tao_converter` не установлен, NGC ONNX → 404). Прогон стартует без изменений кода, как только появится ONNX.

> Подробности локальной кампании — [`results/SUMMARY.md`](results/SUMMARY.md), `results/<model>/EXPERIMENT.md` и ноутбук [`notebooks/local_validation_campaign.ipynb`](notebooks/local_validation_campaign.ipynb). Прежний удалённый отчёт — `results_collected/FINAL_REPORT.md` (архив qudata).

---

## 🗂️ Датасеты

Источник пар «задача × датасет»: [`datasets.md`](datasets.md).

| Задача | Датасет | Назначение | Лицензия | Спека |
|--------|---------|------------|----------|-------|
| Detection | **BDD100K** | детекция ТС (дорожная съёмка), val ~10K | UC Berkeley custom (комм. → OTL) | [📄](docs/about_datasets/bdd100k.md) |
| LP Detection | **AUTO.RIA Numberplate 2021-05-12** | 8042 кадра, VIA-полигоны пластин | **CC BY 4.0** ✅ | [📄](docs/about_datasets/autoria_numberplate_detection_ru.md) |
| OCR | **autoriaNumberplateOcrRu 2021-09-01** | 57 120 кропов RU-номеров (ГОСТ-Р 50577) | **CC BY 4.0** ✅ | [📄](docs/about_datasets/autoria_numberplate_ocr_ru.md) |
| Color | **MAD-Cars (Yandex)** | 360° съёмки; локальный sample покрыл 13/15 цветов | **CC BY-NC-SA 4.0** (NonCommercial) | [📄](docs/about_datasets/mad_cars.md) |
| Make | **VMMRdb** | 9170 классов make/model/year (US) | репо MIT; права на фото — TBD | [📄](docs/about_datasets/vmmrdb.md) |
| Type | **Stanford Cars** | 16 185 изобр., 196 классов | ImageNet-like (non-commercial) | [📄](docs/about_datasets/stanford_cars.md) |
| Face Detection | **WIDER FACE** | val 3226 изобр., AP Easy/Med/Hard | **CC BY-NC-ND 4.0** (NonCommercial) | [📄](docs/about_datasets/wider_face.md) |

> Доменный разрыв — сквозная причина провалов: датасеты ракурса «уровень земли / каталог / толпа» расходятся с automotive POV CarCV (бортовая съёмка, дистанция 5–50 м, угол 0–30°, day/night/IR). Для WIDER FACE рекомендован automotive-срез (`14--Traffic`, `5--Car_Accident`, `59--people--driving--car`).

---

## ⚖️ Лицензии и ограничения

Перед коммерческой поставкой по каждому пункту требуется **legal review**:

- **NVIDIA TAO / NGC** (TrafficCamNet, VehicleMakeNet, VehicleTypeNet, FaceDetect) — модели под NVIDIA AI / TAO model EULA; право на встраивание и дистрибуцию весов/engine в коммерческий продукт нужно подтвердить.
- **Nomeroff Net** (nomeroff_lpd, nomeroff_ocr) — код **GPL-3.0 (copyleft)**: включение в дистрибутив влечёт обязательства по раскрытию исходников производных работ; лицензия на сами веса (`.pt` / `.ckpt`) не подтверждена.
- **bae_model_f3** — лицензия и происхождение весов **неизвестны**; файл вне репозитория и вне NGC. Без легализации источника — в продукт не включать.
- **Датасеты с NonCommercial** (WIDER FACE, MAD-Cars, Stanford Cars) — только внутренний research-бенчмаркинг; публикация метрик/артефактов вовне требует согласования. **CC BY 4.0** (оба AUTO.RIA) — безопасны для коммерции при атрибуции.

---

## 🖥️ Целевая платформа

```
NVIDIA Jetson Orin Nano 8GB  (целевое устройство продакшн-CarCV)
├─ GPU: 1024 CUDA cores (Ampere) · 40 TOPS (INT8)
├─ CPU: 6-core ARM Cortex-A78AE
├─ RAM: 8GB LPDDR5 (unified)
└─ Стек: JetPack 6.x · DeepStream SDK · TensorRT (FP16/INT8)
```

> ⚠️ **Latency/FPS на Jetson в этом репозитории НЕ измерялись.** Цифры вида «8–10 ms», «30 FPS», «<50 ms» из legacy-доков — **аспирационные**. Валидация выполняется на x86 GPU (ONNX Runtime); точность ONNX = верхняя граница для TensorRT-движка. Отдельный риск — `nomeroff_lpd` (YOLOv11x, самая тяжёлая модель стека): фактическую задержку на Orin Nano необходимо измерить до продакшена. NB: точность `nomeroff_lpd` подтверждена на удалённом прогоне qudata; локально импорт пока блокирован упаковкой `modelhub_client`.

---

## 🚀 Как запустить валидацию

```bash
# 1) Загрузка весов TAO-моделей (TrafficCamNet, Make/Type) в deploy/models/
bash deploy/scripts/download_models.sh

# 2) Загрузка датасетов (см. ссылки и пути в datasets.md)
#    например, для OCR/LP — архивы Nomeroff Net; для Detection — bdd100k.zip

# 3) Прогон эвалуатора нужной модели (пороги и препроцессинг — в EVAL_CONFIGS)
python deploy/evaluation/evaluate.py            # см. eval_<model> внутри

# 4) Сводка по результатам
python deploy/evaluation/aggregate_summary.py
```

По правилам проекта (`CLAUDE.md`): результаты сохраняются в `results/` как **JSON + CSV**, графики — в `plots/` (PNG), воспроизводимый код — в `notebooks/`, итог эксперимента — в `results/SUMMARY.md`.

Ключевые файлы:

- `deploy/evaluation/evaluate.py` — эвалуаторы `eval_<model>`, препроцессинг, декодеры, пороги (`EVAL_CONFIGS`).
- `deploy/evaluation/metrics.py` — расчёт метрик (`compute_detection_metrics`, `compute_ocr_metrics`).
- `deploy/scripts/download_models.sh` — загрузка весов (US-модели LPDNet/LPRNet — как контрольные baseline).
- `results/<model>/metrics.json` (+ `EXPERIMENT.md`) — измеренные результаты локальной кампании; `results/SUMMARY.md` — сводка; `notebooks/local_validation_campaign.ipynb` — воспроизводимый ноутбук.
- `results_collected/.../metrics.json`, `results_collected/FINAL_REPORT.md` — прежний удалённый архив qudata (ориентир).

---

## 📚 Документация

| Документ | Описание |
|----------|----------|
| [`models.md`](models.md) · [`datasets.md`](datasets.md) | Источники истины: реестр моделей и пары «модель × датасет» |
| [`docs/about_models/`](docs/about_models) | Спецификации моделей (архитектура, препроцессинг, валидация, лицензия) |
| [`docs/about_datasets/`](docs/about_datasets) | Спецификации датасетов (структура, аннотации, лицензия, рекомендации) |
| [`results/SUMMARY.md`](results/SUMMARY.md) | Итоговая сводка локальной валидационной кампании — **первичный источник результатов** |
| [`docs/model_research/00_SUMMARY.md`](docs/model_research/00_SUMMARY.md) · [`docs/dataset_research/00_SUMMARY.md`](docs/dataset_research/00_SUMMARY.md) | Исследование открытых моделей/датасетов-**аналогов**: лицензии под коммерцию, edge/domain-fit, кандидаты под дообучение (**не** измеренная валидация) |
| [`docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md`](docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md) | Дизайн валидационной кампании на 5 моделей |
| [`docs/architecture.md`](docs/architecture.md) | Архитектура системы (**legacy-стек**, читать с поправкой на актуальный `models.md`) |
| [`docs/system-design/ML_System_Design_Document.md`](docs/system-design/ML_System_Design_Document.md) | ML System Design (требования §5–6; **legacy-метрики непроверены**) |

---

## 🧭 Статус кампании и дальнейшие шаги

**Закрыто (вердикт зафиксирован):**

- [x] `nomeroff_lpd` → AUTO.RIA Detection — **PASS** (P=0.9056 / R=0.9221, qudata)
- [x] `nomeroff_ocr` → autoriaNumberplateOcrRu/val — **PASS** (char=0.9995 / plate=0.9978, qudata)
- [x] `TrafficCamNet` → BDD100K val — **FAIL** (car F1=0.3111)
- [x] `VehicleMakeNet` → VMMRdb — **FAIL** (Top-1=0.4387)
- [x] `VehicleTypeNet` → Stanford Cars — **FAIL** (Top-1=0.3549), окончательно
- [x] `bae_model_f3` (Color) → MAD-Cars — **FAIL** осторожный (Top-1=0.6205)

**Открытая работа:**

- [ ] `FaceDetect` → получить deployable FaceNet ONNX (`tao_converter` / NGC) и запустить уже готовый `eval_facedetect` (сейчас **UNDEF**)
- [ ] `nomeroff_lpd` / `nomeroff_ocr` → снять блокер `modelhub_client` и **локально воспроизвести** удалённый PASS qudata
- [ ] Синхронизировать расхождения версий в `models.md` (TrafficCamNet/Make/Type) с фактически загружаемыми

---

## 📝 Лицензия проекта

Proprietary © 2026 CARS Team. Лицензии моделей и датасетов — см. раздел [«Лицензии и ограничения»](#-лицензии-и-ограничения) и спеки.

---

**Версия README:** 3.1.0 · **Обновлён:** 2026-06-07 · итог **2 PASS / 4 FAIL / 1 UNDEF** (вердикты PASS / FAIL / UNDEF). 4 FAIL + 1 UNDEF — локальная кампания [`results/SUMMARY.md`](results/SUMMARY.md) (RTX 3090); nomeroff_lpd/ocr PASS — на удалённом прогоне qudata (в SUMMARY помечены DEFERRED как локально не воспроизведённые).
