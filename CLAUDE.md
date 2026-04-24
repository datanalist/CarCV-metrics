# CARS Model Evaluation Project

## Окружение
- GPU: NVIDIA (проверить через `nvidia-smi`)
- Python: 3.10+, venv в ./venv
- Данные валидации: /data/validation/
- Модели: /models/

## Текущая задача
Валидация ML-моделей для бортовой системы CARS.

## Модели для валидации на этом сервере
- TrafficCamNet (ResNet-18 pruned) — primary detector
- Целевые метрики: Precision >0.90, Recall >0.85, F1 >0.87

## Правила
- Все результаты сохранять в results/ как JSON + CSV
- Графики сохранять в plots/ как PNG
- В конце каждого эксперимента создавать summary в results/SUMMARY.md
- Jupyter notebook с воспроизводимым кодом сохранять в notebooks/
- Логировать все эксперименты
