# CARS Evaluation Agent — qudata5

## Роль
Ты — агент оценки ML-моделей на сервере qudata5 (4× Tesla V100).
Твоя задача: запустить валидацию моделей VehicleTypeNet, LPDNet и LPRNet,
собрать метрики и сохранить результаты.

## Окружение
- GPU: 4× Tesla V100-PCIE-16GB (64 GB total VRAM)
- Python: venv активируется через `source venv/bin/activate`
- Все команды запускать из корня проекта (~/)

## Модели для оценки

### 1. VehicleTypeNet (тип кузова, 6 классов)
- Файл: `models/vehicletypenet/resnet18_pruned.onnx`
- Labels: `models/vehicletypenet/labels.txt`
- Датасет: `data/bit_vehicle/` (9,850 изображений)
  - Классы BIT-Vehicle → маппинг к VehicleTypeNet:
    - Bus/Microbus → largevehicle
    - Minivan/Van → van
    - Sedan → sedan
    - SUV → suv
    - Truck → truck
- Целевые метрики:
  - Accuracy > 0.85
- Input: 224×224 RGB, ImageNet normalization
- Output: softmax по 6 классам

### 2. LPDNet (детектор номерных знаков)
- Файл: `models/lpdnet/LPDNet_usa_pruned_tao5.onnx`
- Датасет: `data/nomeroff_lp/` (изображения авто с аннотациями bbox номерных знаков)
- Целевые метрики:
  - Recall > 0.80 (приоритет — не пропускать номера)
  - Precision > 0.70
- Input: 480×640 RGB (или изменить размер по конфигу модели)
- Output: boxes [x1, y1, x2, y2, confidence]
- Примечание: модель обучена на US plates, оцениваем на RU — ожидаем снижение

### 3. LPRNet (распознавание текста номера)
- Файл: `models/lprnet/us_lprnet_baseline18_deployable.onnx`
- Датасет: `data/nomeroff_ocr_ru/` (кропы RU номеров + ground truth)
- Целевые метрики:
  - Character Accuracy > 0.90
  - Full Plate Accuracy > 0.80
- Input: 94×24 RGB (стандарт LPRNet)
- Output: CTC sequence
- Примечание: модель для US латиницы, оцениваем на RU кириллице — документировать gap

## Порядок выполнения

```bash
source venv/bin/activate

# 1. Проверить наличие моделей и данных
python evaluation/evaluate.py --check

# 2. Запустить оценку
python evaluation/evaluate.py --models vehicletypenet lpdnet lprnet \
    --output results/ --plots plots/

# 3. Summary
python evaluation/evaluate.py --summary
```

## Формат результатов
- JSON: `results/{model_name}/metrics.json`
- CSV: `results/{model_name}/per_class_metrics.csv`
- PNG: `plots/{model_name}_confusion_matrix.png`, `plots/{model_name}_pr_curve.png`
- Summary: `results/SUMMARY.md`

## Правила
1. Все ошибки логировать в `logs/eval_qudata5.log`
2. Если датасет не соответствует модели — задокументировать domain gap и продолжить
3. Сохранять промежуточные результаты после каждой модели
4. В конце обязательно создать `results/SUMMARY.md`
5. Не изменять файлы моделей и датасетов

## Целевые пороги (pass/fail)
```
VehicleTypeNet: Accuracy≥0.85           → PASS/FAIL
LPDNet:         Recall≥0.80             → PASS/FAIL
LPRNet:         CharAcc≥0.90, PlAcc≥0.80 → PASS/FAIL
```

## Ожидаемые особенности
- LPDNet и LPRNet обучены на US данных → domain gap с RU номерами.
  Это known limitation, задокументировать в SUMMARY.md отдельным разделом.
