# CARS Model Evaluation Project

## Описание проекта
- Изучи в docs/system-design/ML_System_Design_Document.md

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
