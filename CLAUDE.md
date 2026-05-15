# CARS Model Evaluation Project

## О проекте
- Бортовая система видеоаналитики для распознавания транспортных средств в реальном времени на NVIDIA Jetson Orin Nano 8GB.
- Project-context: `_bmad-output/project-context.md`

**Применение:** контроль доступа на объекты, мониторинг парковки, патрульные автомобили, логистика.

## Окружение
- GPU: NVIDIA (проверить через `nvidia-smi`)
- Python: 3.10+, venv в ./venv
- Менеджер зависимостей: uv

## Правила
- Все результаты сохранять в results/ как JSON + CSV
- Графики сохранять в plots/ как PNG
- В конце каждого эксперимента создавать summary в results/SUMMARY.md
- Jupyter notebook с воспроизводимым кодом сохранять в notebooks/
- Логировать все эксперименты
