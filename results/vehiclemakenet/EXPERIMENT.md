# Эксперимент: VehicleMakeNet × VMMRdb

Дата: 2026-06-04 · Ветка: exp/eval-vehiclemakenet · GPU: (впишет контроллер)

## 1. Выписка из docs (шаг 1 протокола)
- Модель: NVIDIA TAO image classification, backbone **ResNet-18 (pruned)**, фактически загружаемая
  версия `pruned_onnx_v1.1.0` (в `models.md` записана `pruned_v1.0.2` — расхождение версий зафиксировано
  в docs, авторитетна фактически загружаемая `download_models.sh`). Голова — **20 выходных классов** марок.
- Препроцессинг (`preprocess_tao_bgr` из `evaluate.py`): вход **224×224**, 3 канала,
  **BGR без свопа в RGB**, **без масштаба** (без `/255`), только поканальное вычитание
  offsets **(104, 117, 124)** в порядке B;G;R. Тензор **NCHW**.
- Выход: вектор логитов размерности 20 → softmax → top-3 (`probs.argsort()[::-1][:3]`); top-1 = первый.
- **20 NGC-марок** (`NGC_MAKES` / `labels.txt`, нижний регистр, порядок = индекс выхода):
  acura, audi, bmw, chevrolet, chrysler, dodge, ford, gmc, honda, hyundai, infiniti, jeep, kia,
  lexus, mazda, mercedes, nissan, subaru, toyota, volkswagen. Все 20 — US/EU-рынок.
- **Источник данных — VMMRdb** (`/home/mk/Загрузки/DATASETS/VMMRdb`): один каталог = один класс,
  имя каталога вида `make_model_year` (напр. `acura_cl_1997`). Отдельных файлов аннотаций нет —
  метка берётся из имени каталога. **make = первый токен по `_`** (`cls_dir.name.split("_")[0]`),
  затем нормализуется `normalize_brand`. Пробельные марки остаются одним first-token при `split("_")`:
  `mercedes benz` → `normalize_brand` → `mercedes` (подстрочное совпадение, попадает в 20 NGC). ✓
- **OOD-skip.** Каталоги, чья марка после `normalize_brand` не входит в 20 NGC (Tesla, LandRover,
  RollsRoyce, AM General и ~50 других токенов), пропускаются и считаются отдельно (`skipped_ood`,
  исключаются из метрик).
- Пороги PASS (`evaluate.py`, `eval_vehiclemakenet`, `thresholds`): **Top-1 accuracy ≥ 0.70**,
  **Top-3 accuracy ≥ 0.85**. (Пороги 0.90/0.97 из `vehiclemakenet_eval.yaml` относятся к
  другому эксперименту — finetuned-модели на 105 классов, не к этому 20-классовому baseline.)
- Совместимость путей данных: эвалуатор сохраняет рабочую ветку **MAD-Cars** (`sample_5k.json` +
  `images/`) и добавляет ветку **VMMRdb** (каталоги-классы). Выбор источника автоматический:
  есть `sample_5k.json` → MAD-Cars; иначе каталоги-классы → VMMRdb.
- Риски / расхождения:
  - VMMRdb крупнее и иного домена (каталожные/объявленческие ракурсы, не automotive POV) — итог
    **вероятно FAIL** (валидный окончательный результат, не повод для тюнинга).
  - Сильный дисбаланс классов (long-tail) и **большое число OOD-каталогов** → крупный `skipped_ood`.
  - Возможный train/test leak (VehicleMakeNet обучен на US-данных, VMMRdb — тоже US) → метрики
    могут быть оптимистично смещены вверх.
  - Полный прогон ~222k изображений (in-distribution срез после OOD-skip) — длительный GPU-прогон.

## 2. Подготовка
Отдельная подготовка данных не требуется: VMMRdb используется как есть (каталоги-классы), метка
выводится из имени каталога. Чтение реализует `discover_vmmrdb_samples(data_dir, NGC_MAKES_LOWER)`
в `deploy/evaluation/evaluate.py` — возвращает список `[(img_path, brand)]` только по 20 NGC-маркам
(`*.jpg` внутри каждого in-distribution каталога), OOD-каталоги пропускаются.

Команда прогона (запускает контроллер): `.venv/bin/python deploy/scripts/run_local.py --models vehiclemakenet`
с `data_dir`, указывающим на каталог VMMRdb.

## 3. Измеренные метрики
Прогон ~222k изображений запускается контроллером в фоне; метрики и вердикт впишутся после завершения.

## 4. Вердикт
Прогон ~222k изображений запускается контроллером в фоне; метрики и вердикт впишутся после завершения.
