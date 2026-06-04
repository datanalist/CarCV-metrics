# Эксперимент: VehicleTypeNet × Stanford Cars

Дата: 2026-06-04 · Ветка: exp/eval-vehicletypenet · GPU: RTX 3090

## 1. Выписка из docs (шаг 1 протокола)
- Модель: TAO classification, backbone **ResNet-18 (pruned)**, версия фактической загрузки
  `pruned_onnx_v1.1.0` (в `models.md` записана `pruned_v1.0.2` — расхождение версий зафиксировано
  в docs, авторитетна фактически загруженная версия). Голова — 6 выходных классов (типы кузова).
- Препроцессинг (`preprocess_tao_bgr` из `evaluate.py`): вход **224×224** (W×H), 3 канала,
  **BGR без свопа в RGB**, **без масштаба** (`net-scale-factor = 1.0`, без /255), только поканальное
  вычитание offsets **(104, 117, 124)** в порядке B;G;R. Тензор **NCHW**.
- Выход: вектор логитов размерности 6 → softmax → top-3 (`probs.argsort()[::-1][:3]`); top-1 = первый.
- Классы (порядок `labels.txt` = индекс выхода): `coupe` (0), `largevehicle` (1), `sedan` (2),
  `suv` (3), `truck` (4), `van` (5). В эвалуаторе нормализуются (`strip().lower()`).
- Маппинг Stanford → тип кузова делает сам `eval_vehicletypenet` через `derive_typenet_label`
  (таблица `STANFORD_BODY_KEYWORDS`, приоритет от специфичных слов к общим; первое совпавшее
  ключевое слово выигрывает): `convertible→coupe`, `hatchback→sedan`, `minivan→van`,
  `crew cab→truck`, `extended cab→truck`, `regular cab→truck`, `cargo van→van`, `supercab→truck`,
  `wagon→sedan`, `sedan→sedan`, `coupe→coupe`, `suv→suv`, `hummer→suv`, `van→van`, `cab→truck`.
  Если ни одно ключевое слово не совпало — образец пропускается (`skipped_no_mapping`).
  Класс `largevehicle` среди ground-truth Stanford не появляется (нет ключевого слова на него).
- Порог PASS (`evaluate.py`, `eval_vehicletypenet`): **Top-1 accuracy ≥ 0.85**.
- **Источник меток: TRAIN-сплит Stanford Cars** (`cars_train_annos.mat` + `cars_meta.mat`, в них ЕСТЬ метки).
  Официальный test-сплит без меток (`cars_test_annos_withlabels.mat` локально отсутствует;
  есть только `cars_test_annos.mat` без `class`), поэтому для оценки используется TRAIN.
- Риски / расхождения: доменный разрыв + шумные суррогатные метки (Stanford размечен по make/model,
  тип кузова выводится по ключевым словам — приблизительно). Прежний прогон давал Top1≈0.36 (FAIL) —
  результат переизмеряется; FAIL ожидается валидным окончательным результатом, не повод для тюнинга.

## 2. Подготовка
Конвертер `deploy/scripts/prep_stanford_cars.py` (Stanford devkit `.mat` → `test.json` + симлинк `images/`).
`test.json`: `[{file_name, label:"<Make> <Model> <BodyType> <Year>"}]`; тип кузова выводит сам
`eval_vehicletypenet` (`derive_typenet_label`).

Команда (TRAIN-сплит — у него есть метки классов):
```bash
.venv/bin/python deploy/scripts/prep_stanford_cars.py \
  --meta-mat "/home/mk/Загрузки/DATASETS/Stanford Cars Dataset/car_devkit/devkit/cars_meta.mat" \
  --annos-mat "/home/mk/Загрузки/DATASETS/Stanford Cars Dataset/car_devkit/devkit/cars_train_annos.mat" \
  --images-dir "/home/mk/Загрузки/DATASETS/Stanford Cars Dataset/cars_train/cars_train" \
  --out-dir data/prep/vehicletypenet
```
Результат: `test.json: 8144 записей · 196 классов · images → .../cars_train/cars_train`
(симлинк `data/prep/vehicletypenet/images` на 8144 jpg). `data/prep/` в git не коммитится (gitignore).
Реальный layout `cars_train_annos.mat` имеет именованные поля
`('bbox_x1','bbox_y1','bbox_x2','bbox_y2','class','fname')` — `parse_annos` использует доступ по
именам `class`/`fname` без адаптации.

Прогон: `.venv/bin/python deploy/scripts/run_local.py --models vehicletypenet`.
Provider: **CUDAExecutionProvider** (`Loaded resnet18_pruned.onnx on CUDAExecutionProvider`),
RTX 3090, 7577 изображений за ~39 с (≈193 img/s).
Из 8144 записей `eval_vehicletypenet` отобрал **7577** (по совпавшему ключевому слову типа кузова) и
**пропустил 567** (`skipped_no_mapping` — нет ключевого слова в имени класса Stanford).
Audit-таблица соответствий: `data/prep/vehicletypenet/type_mapping.csv` (182 уникальных класса).

## 3. Измеренные метрики
Источник: `results/vehicletypenet/metrics.json` · #images = 7577 · #skipped (no mapping) = 567.

| Метрика | Значение | Порог | Статус |
|---------|----------|-------|--------|
| Top-1 accuracy | **0.3549** | ≥ 0.85 | **FAIL** |
| Top-3 accuracy | 0.6932 | — | — |

Per-class Top-1 (по выведенным меткам типа кузова):

| Класс | Top-1 |
|-------|-------|
| `truck` | 0.6137 (лучший) |
| `suv` | 0.3807 |
| `van` | 0.3757 |
| `coupe` | 0.3282 |
| `sedan` | 0.2867 (худший) |
| `largevehicle` | — (нет ground-truth на Stanford) |

Confusion-матрица: `plots/vehicletypenet_confusion.png`.

## 4. Вердикт
Порог: Top-1 accuracy ≥ 0.85.
- Top-1 0.3549 < 0.85 → **FAIL**

**Вердикт: FAIL** (окончательный валидный результат, не повод для тюнинга).

Причина: доменный разрыв + шумные суррогатные метки. Stanford Cars размечен по make/model, а не по
типу кузова; тип кузова выводится по ключевым словам из имени класса (`derive_typenet_label`) —
приблизительно (например, любой `convertible→coupe`, `wagon→sedan`). Класс `largevehicle` на Stanford
не представлен. Числа близки к прежнему прогону (Top1≈0.3575 на 7483 образцах) — переизмерено на TRAIN-
сплите (7577 образцов), FAIL подтверждён. Задача Type считается отвалидированной на доступном
(суррогатном) датасете и выведена из активных задач кампании.
