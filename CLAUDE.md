# CARS Model Evaluation Project

## Описание проекта

**CARS (Computer Automotive Recognition System)** — бортовая система видеоаналитики на NVIDIA Jetson Orin Nano 8GB для автономного распознавания транспортных средств в реальном времени без подключения к облаку.

**Пайплайн:** GStreamer/DeepStream (C ~2400 LOC) + Python-сервис (~600 LOC)
- **PGIE:** TrafficCamNet (ResNet-18 pruned, TensorRT FP16, 960×544) — детекция ТС
- **SGIE×4:** VehicleMakeNet (20 марок), VehicleTypeNet (6 типов), LPDNet, FaceDetect
- **OCR:** LPR_STN_PRE_POST.onnx (STN+LSTM+CTC, русский алфавит, 188×48)
- **Цвет:** bae_model_f3.onnx (ResNet, 15 цветов, 384×384)
- **Хранение:** SQLite (my.db → final.db) + BMP-кропы на NVMe SSD
- **API:** REST HTTP на порту 8080 + Web UI

**Целевые характеристики:** ≥30 FPS @ 1080p, latency <50ms, power <25W

**Применение:** контроль доступа на объекты, мониторинг парковки, патрульные автомобили, логистика.

## Окружение
- GPU: NVIDIA (проверить через `nvidia-smi`)
- Python: 3.10+, venv в ./venv
- Менеджер зависимостей: uv
- Данные валидации: скачиваются под задачу (смотри инструкцию в docs/rules/task.md)
- Модель: скачиваются под задачу (смотри инструкцию в docs/rules/task.md)

## Текущая задача
- Описана в docs/rules/task.md

## Модели для валидации на этом сервере
- TrafficCamNet (ResNet-18 pruned) — primary detector
- Целевые метрики: Precision >0.90, Recall >0.85, F1 >0.87

## Правила
- Все результаты сохранять в results/ как JSON + CSV
- Графики сохранять в plots/ как PNG
- В конце каждого эксперимента создавать summary в results/SUMMARY.md
- Jupyter notebook с воспроизводимым кодом сохранять в notebooks/
- Логировать все эксперименты
