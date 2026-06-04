# Эксперимент: TrafficCamNet × BDD100K

Дата: 2026-06-04 · Ветка: exp/eval-trafficcamnet · GPU: RTX 3090

## 1. Выписка из docs (шаг 1 протокола)
- Модель: DetectNet_v2 (backbone ResNet-18 pruned), версия `pruned_onnx_v1.0.4`.
  Препроцессинг: resize до **960×544** (W×H) без сохранения пропорций (`maintain-aspect-ratio=0`);
  BGR→RGB (`model-color-format=0`); масштаб `/255` (`net-scale-factor=0.00392156862745098`);
  offsets = `0;0;0` (без вычитания среднего). Тензор NCHW (`transpose(2,0,1)[None]`).
  Вход: `3×544×960`. Выходы: `output_cov/Sigmoid` `[C,gh,gw]` (coverage) +
  `output_bbox/BiasAdd` `[4C,gh,gw]` (смещения границ). Декодер `detectnet_v2_decode`:
  stride=16, bbox_norm=35.0, финальный NMS IoU=0.5.
  Классы (порядок `labels.txt` = индекс): `car` (0), `bicycle` (1), `person` (2), `road_sign` (3).
- Датасет: BDD100K val — 10 000 изображений (`bdd100k_labels_images_val.json`).
  GT-формат: массив `{name, labels:[{category, box2d:{x1,y1,x2,y2}}]}`, координаты абсолютные в пикселях.
  Категории BDD: `car, truck, bus, bike, motor, person, rider, traffic light, traffic sign`.
  Маппинг BDD→TrafficCamNet делает сам `eval_trafficcamnet` через `TRAFFICCAMNET_GT_VOCAB`:
  `car`→{car}; `person`→{person, pedestrian}; `bicycle`→{bike, bicycle, rider};
  `road_sign`→{traffic sign, sign, traffic_sign, trafficsign}.
  (BDD `bike` и `rider` засчитываются как `bicycle`; `traffic sign` → `road_sign`.)
- Пороги (PASS): P≥0.90 R≥0.85 F1≥0.87 ; conf_thr=0.2 (cross-domain)
- Риски / расхождения с legacy: cross-domain BDD→ожидаемо низкие метрики; прежний прогон давал FAIL (валидно).
  Расхождение версии: `models.md` указывает `pruned_quantized_v2.0.1`/`pruned_v1.0.3`, фактически локально и
  в `download_models.sh` используется `pruned_onnx_v1.0.4` — авторитетна локальная версия.
  Примечание: docs упоминают для in-domain BDD100K `conf_thr=0.4`; согласно ТЗ и EVAL_CONFIGS используется
  `conf_thr=0.2` (значение из cfg) — фиксируется фактически использованный порог.

## 2. Подготовка
Конвертер `deploy/scripts/prep_bdd100k.py` (BDD val JSON → labels.json + симлинк images/).
Команда:
```bash
.venv/bin/python deploy/scripts/prep_bdd100k.py \
  --bdd-json "/home/mk/Загрузки/DATASETS/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json" \
  --images-dir "/home/mk/Загрузки/DATASETS/bdd100k/bdd100k/bdd100k/images/100k/val" \
  --out-dir data/prep/trafficcamnet
```
Результат: `labels.json: 10000 записей · images → .../images/100k/val` (симлинк на 10 000 jpg).
Маппинг BDD→TrafficCamNet выполняет сам `eval_trafficcamnet` через `TRAFFICCAMNET_GT_VOCAB`;
конвертер сохраняет сырые BDD-категории, приводя только `box2d`→`bbox2d`.
`data/prep/` в git не коммитится (gitignore).

Прогон: `.venv/bin/python deploy/scripts/run_local.py --models trafficcamnet`.
Provider: **CUDAExecutionProvider** (`Loaded resnet18_trafficcamnet_pruned.onnx on CUDAExecutionProvider`),
RTX 3090, ~10 000 изображений за ~3 мин (≈53 img/s).

## 3. Измеренные метрики
Источник: `results/trafficcamnet/metrics.json` · `conf_thr=0.2` · #images = 10 000.

| Класс | precision | recall | f1 | ap | num_gt | num_pred | num_tp |
|-------|-----------|--------|-----|-----|--------|----------|--------|
| **car** (primary) | 0.3909 | 0.2584 | 0.3111 | 0.1483 | 102 506 | 67 753 | 26 487 |
| person | 0.2485 | 0.1188 | 0.1607 | 0.0788 | 13 262 | 6 337 | 1 575 |
| bicycle | 0.1458 | 0.0604 | 0.0854 | 0.0579 | 1 656 | 686 | 100 |
| road_sign | 0.1714 | 0.0182 | 0.0330 | 0.0334 | 34 908 | 3 716 | 637 |
| **macro (4 класса)** | 0.2392 | 0.1140 | 0.1476 | 0.0796 | — | — | — |

## 4. Вердикт
Пороги (по primary-классу `car`): P≥0.90, R≥0.85, F1≥0.87.
- precision 0.3909 < 0.90 → FAIL
- recall 0.2584 < 0.85 → FAIL
- f1 0.3111 < 0.87 → FAIL

**Вердикт: FAIL** (cross-domain, conf_thr=0.2).

Причина: доменный разрыв. TrafficCamNet обучен на видах с дорожных камер (вид сверху/под углом),
BDD100K val — бортовая съёмка с уровня дороги. Метрики на in-domain BDD100K заметно выше прежнего
суррогата COCO val2017 (там car P=0.0819, R=0.0528, F1=0.0642), но порогов кампании не достигают.
FAIL из-за доменного разрыва — валидный окончательный результат, не повод для тюнинга.
