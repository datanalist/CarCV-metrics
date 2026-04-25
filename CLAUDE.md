# CARS Model Evaluation Project

## О проекте
- Бортовая система видеоаналитики для распознавания транспортных средств в реальном времени на NVIDIA Jetson Orin Nano 8GB.
- System Design: `docs/system-design/ML_System_Design_Document.md` (смотреть только разделы, относящиеся к текущей задаче).

**Применение:** контроль доступа на объекты, мониторинг парковки, патрульные автомобили, логистика.

## Окружение
- GPU: NVIDIA (проверить через `nvidia-smi`)
- Python: 3.10+, venv в ./venv
- Менеджер зависимостей: uv

## Текущая задача
- Описана в docs/tasks/task.md

## Правила
- Все результаты сохранять в results/ как JSON + CSV
- Графики сохранять в plots/ как PNG
- В конце каждого эксперимента создавать summary в results/SUMMARY.md
- Jupyter notebook с воспроизводимым кодом сохранять в notebooks/
- Логировать все эксперименты
