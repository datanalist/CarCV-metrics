# CARS Evaluation Agent — qudata2

## Роль
Ты — агент оценки ML-моделей на сервере qudata2 (2× RTX 4090).
Твоя задача: запустить валидацию моделей TrafficCamNet, VehicleMakeNet и Color,
собрать метрики и сохранить результаты.

## Окружение
- GPU: 2× NVIDIA RTX 4090 (24 GB each)
- Python: venv активируется через `source venv/bin/activate`
- Все команды запускать из корня проекта (~/)

## Модели для оценки

### 1. TrafficCamNet (детектор ТС)
- Файл: `models/trafficcamnet/resnet18_trafficcamnet_pruned.onnx`
- Датасет: `data/bdd100k/` (10K изображений)
- Целевые метрики (из System Design §3.3):
  - Precision > 0.90
  - Recall > 0.85
  - F1 > 0.87
- Класс для оценки: только `car` (class_id=0)
- Input: 960×544 BGR, normalize по умолчанию
- Output: boxes [cx, cy, w, h, confidence, class_id]

### 2. VehicleMakeNet (классификация марки)
- Файл: `models/vehiclemakenet/resnet18_pruned.onnx`
- Labels: `models/vehiclemakenet/labels.txt`
- Датасет: `data/mad_cars/sample_5k.json` + `data/mad_cars/images/`
- Целевые метрики:
  - Top-1 Accuracy > 0.70
  - Top-3 Accuracy > 0.85
- Input: 224×224 RGB, ImageNet normalization
- Output: softmax scores по 20 классам

### 3. Color Recognition (цвет автомобиля)
- Файл: `models/color/` — если модель отсутствует, пропустить и записать N/A
- Датасет: `data/mad_cars/sample_5k.json` (поле `color`, 16 классов)
- Целевая метрика: Accuracy > 0.75

## Порядок выполнения

```bash
source venv/bin/activate

# 1. Убедиться что модели и данные на месте
python evaluation/evaluate.py --check

# 2. Запустить оценку всех моделей
python evaluation/evaluate.py --models trafficcamnet vehiclemakenet color \
    --output results/ --plots plots/

# 3. Сгенерировать summary
python evaluation/evaluate.py --summary
```

## Формат результатов
- JSON: `results/{model_name}/metrics.json`
- CSV: `results/{model_name}/per_class_metrics.csv`
- PNG: `plots/{model_name}_confusion_matrix.png`, `plots/{model_name}_pr_curve.png`
- Summary: `results/SUMMARY.md`

## Правила
1. Все ошибки логировать в `logs/eval_qudata2.log`
2. Если модель не загружается — записать ошибку и продолжить со следующей
3. Сохранять промежуточные результаты после каждой модели
4. В конце обязательно создать `results/SUMMARY.md`
5. Не изменять файлы моделей и датасетов

## Целевые пороги (pass/fail)
```
TrafficCamNet: Precision≥0.90, Recall≥0.85, F1≥0.87  → PASS/FAIL
VehicleMakeNet: Top1≥0.70, Top3≥0.85                  → PASS/FAIL
Color: Accuracy≥0.75                                    → PASS/FAIL (если есть модель)
```
