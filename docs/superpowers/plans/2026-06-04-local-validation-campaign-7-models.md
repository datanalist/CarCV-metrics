# План реализации: локальная валидационная кампания, 7 моделей

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Честно измерить 7 пар *модель × датасет* локально на RTX 3090 (по ветке git на эксперимент), зафиксировать метрики и вердикт PASS/FAIL/UNDEF по утверждённым порогам; FAIL/UNDEF — валидные окончательные результаты.

**Architecture:** Базис — существующий `deploy/evaluation/evaluate.py` (универсальный evaluator detection/classification/OCR). Новый код минимален: локальный раннер `run_local.py` (накладывает абсолютные пути из `configs/local_paths.yaml` на `EVAL_CONFIGS` и зовёт `cfg["eval_fn"](cfg)`), 2 новых эвалуатора (`eval_color`, `eval_facedetect`), ветка чтения VMMRdb в `eval_vehiclemakenet`, идемпотентные конвертеры данных. Результаты/графики уходят в репозиторные `results/` и `plots/` через симлинки `deploy/results`→`../results`, `deploy/plots`→`../plots` (эвалуаторы пишут в `ROOT/results`, `ROOT/plots`, где `ROOT = deploy/`).

**Tech Stack:** Python 3.13 (`.venv`, uv), onnxruntime 1.24 (CUDA/TensorRT EP), torch 2.11, opencv 4.13, numpy 2.4, PyYAML, scipy (для `.mat`), pytest (TDD нового кода), nomeroff-net (для LPD/OCR, установка рискованна), matplotlib.

---

## Карта файлов (что создаём / меняем)

**Создаём:**
- `deploy/scripts/run_local.py` — локальный раннер (overlay путей + вызов `eval_fn`).
- `configs/local_paths.yaml` — единственное место с абсолютными локальными путями.
- `deploy/scripts/prep_bdd100k.py` — конвертер BDD100K val JSON → `labels.json` + симлинк `images/`.
- `deploy/scripts/prep_stanford_cars.py` — `cars_meta.mat` + `cars_train_annos.mat` → `test.json` + симлинк `images/`.
- `deploy/scripts/get_facenet_onnx.sh` — попытка получить FaceNet ONNX (NGC → экспорт etlt → иначе UNDEF).
- `deploy/tests/conftest.py` — `sys.path` для импорта `evaluation/` и `scripts/`.
- `deploy/tests/test_run_local.py`, `test_prep_bdd100k.py`, `test_prep_stanford.py`, `test_vmmrdb_reader.py`, `test_eval_color.py`, `test_eval_facedetect.py` — TDD нового кода.
- `results/<model>/EXPERIMENT.md` для каждой из 7 моделей.
- `notebooks/local_validation_campaign.ipynb` — воспроизводимый прогон.

**Меняем:**
- `deploy/evaluation/evaluate.py` — добавить `HEX_TO_CARS_COLOR`, `COLOR_CLASSES`, `preprocess_color`, `load_madcars_color_index`, `eval_color`; `parse_wider_gt`, `eval_facedetect`; `discover_vmmrdb_samples` + ветку VMMRdb в `eval_vehiclemakenet`; записи `color` и `facedetect` в `EVAL_CONFIGS`; защитить `check_all()` от конфигов без `model_path`.
- `.gitignore` — добавить `*.etlt`, `/data/prep/`, симлинки `/deploy/results`, `/deploy/plots`.
- `deploy/requirements.txt` и `pyproject.toml` — добавить `scipy>=1.11`.

**Не трогаем:** `deploy/evaluation/metrics.py`, `aggregate_summary.py`, `visualize.py`, `deploy/scripts/run_remote.py` (и весь SSH/qudata-тулинг).

---

## Опорные факты из кодовой базы (проверено 2026-06-04, не выдумывать заново)

- `evaluate.py`: `ROOT = Path(__file__).parent.parent` = `/home/mk/CarCV-metrics/deploy`. Все `cfg[...]`-пути резолвятся `ROOT / cfg[...]`. **Абсолютный путь в overlay перекрывает join** (`Path("/a") / "/b" == Path("/b")`) — поэтому накладывать абсолютные пути безопасно.
- `deploy/` — **не пакет** (нет `__init__.py`); `evaluate.py` импортит `from metrics import ...`, `from visualize import ...` как соседние модули (script-dir на `sys.path`).
- `eval_fn(cfg: dict) -> dict`; успех → `{"metrics": {...}, "thresholds": <check_thresholds(...)>}`; пишет `(ROOT/cfg["results_dir"] / "metrics.json")`. Ошибка → `{"error": "..."}`.
- `check_thresholds(metrics: dict, thresholds: dict) -> {key: {"value","threshold","status": "PASS"/"FAIL"}}` (из `metrics.py`).
- `compute_detection_metrics(predictions, ground_truths, iou_threshold=0.5, conf_threshold=0.35)`: `predictions=[{"image_id","boxes":[[x1,y1,x2,y2,conf]]}]`, `ground_truths=[{"image_id","boxes":[[x1,y1,x2,y2]]}]` → `DetectionMetrics(precision,recall,f1,ap,map50,num_gt,num_pred,num_tp)` (`ap`/`map50` = AP@0.5, 11-точечная).
- `compute_classification_metrics(predictions, ground_truths)`: `predictions=[{"image_id","top_k":["lbl",...]}]`, `ground_truths=[{"image_id","label":"lbl"}]` → `ClassificationMetrics(top1_accuracy, top3_accuracy, num_samples, per_class_accuracy{lbl:acc})`.
- Хелперы в `evaluate.py`: `softmax` (77), `nms`/`nms_iou` (90–109), `detectnet_v2_decode(cov,bbox,target_cls,conf_thr=0.4,stride=16,bbox_norm=35.0,img_w,img_h,scale_w,scale_h)` (112, внутри уже `nms(iou_thr=0.5)`), `get_ort_session` (47), `load_labels` (82, понимает `\n` и `;`), `preprocess_tao_bgr` (63), `softmax` (77), `TRAFFICCAMNET_GT_VOCAB` (154), `NGC_MAKES_LOWER` (302), `normalize_brand` (305), `STANFORD_BODY_KEYWORDS`+`derive_typenet_label` (424).
- **Нет** в коде: `eval_color`, `eval_facedetect`, `HEX_TO_CARS_COLOR`, билдер confusion-матрицы, CSV-врайтер per-class (кроме инлайна в makenet/typenet).
- `metrics.json` схема (verbatim прецедент): top-level `metrics` + `thresholds`; classification-семейство выводится по наличию `top1_accuracy`, detection — по `{precision,recall,f1}` (`aggregate_summary.infer_family`).
- `aggregate_summary.py` сканирует `results_collected/**/metrics.json`; финальный `results/SUMMARY.md` удобнее строить через `evaluate.py --summary` (читает `ROOT/results/<name>/metrics.json` для всех ключей `EVAL_CONFIGS`).

### Проверенные локальные пути (для `local_paths.yaml`)
| Ключ | Абсолютный путь | Статус |
|---|---|---|
| TrafficCamNet ONNX | `/home/mk/CarCV/models/trafficcamnet_pruned_onnx_v1.0.4/resnet18_trafficcamnet_pruned.onnx` | ✅ + `labels.txt` рядом (`car/bicycle/person/road_sign`) |
| VehicleMakeNet ONNX | `/home/mk/CarCV/models/vehiclemakenet_pruned_onnx_v1.1.0/resnet18_pruned.onnx` | ✅ + `labels.txt` (20 марок через `;`) |
| VehicleTypeNet ONNX | `/home/mk/CarCV/models/vehicletypenet_pruned_onnx_v1.1.0/resnet18_pruned.onnx` | ✅ + `labels.txt` (6 типов) |
| Color ONNX | `/home/mk/CarCV/models/bae_model_f3.onnx` | ✅ (67 MB, логиты) |
| nomeroff_lpd веса | `/home/mk/CarCV/models/nomeroff_net/object_detection/yolov26x-keypoints-2026-01-21.pt` (+ `yolov11x-keypoints-2026-01-21.onnx`) | ✅ (обновлено в `models.md`) |
| FaceNet | `/home/mk/Загрузки/facenet_pruned_quantized_v2.0.1/model.etlt` (зашифр.) | ⚠️ ONNX нет — получить или UNDEF |
| BDD100K val labels | `/home/mk/Загрузки/DATASETS/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json` | ✅ (10000 записей) |
| BDD100K val images | `/home/mk/Загрузки/DATASETS/bdd100k/bdd100k/bdd100k/images/100k/val` | ✅ (10000 jpg) |
| WIDER val | `/home/mk/Загрузки/DATASETS/Wider Face/WIDER_val` (gt txt + `images/<cat>/`) | ✅ (3226 img; пробел в пути → кавычки) |
| VMMRdb | `/home/mk/Загрузки/DATASETS/VMMRdb` | ✅ (9171 каталог `<make>_<model>_<year>`) |
| Stanford devkit | `/home/mk/Загрузки/DATASETS/Stanford Cars Dataset/car_devkit/devkit` (`cars_meta.mat`, `cars_train_annos.mat` с метками) | ✅ (тест без меток — берём train) |
| Stanford train images | `/home/mk/Загрузки/DATASETS/Stanford Cars Dataset/cars_train/cars_train` | ✅ (8144 jpg) |
| MAD-Cars | `/home/mk/CarCV/data/external/ymad_cars` (`images_index.jsonl`, `images/`) | ✅ (9996 jpg; поля `car_id,brand,model,color`) |
| AutoRia detection val | `/home/mk/Загрузки/DATASETS/nomeroff/autoriaNumberplateDataset-2021-05-12` (`val/via_region_data.json`) | ✅ |
| AutoRia OCR val | `/home/mk/Загрузки/DATASETS/nomeroff/autoriaNumberplateOcrRu-2021-09-01` (`val/img`+`val/ann`) | ✅ (4893) |

### Известные блокеры окружения
- `scipy` и `nomeroff_net` **не установлены** в `.venv` (ставить через `uv pip install`).
- `cars_test_annos_withlabels.mat` отсутствует на диске → классификация Stanford только на train split (с метками); зафиксировать в EXPERIMENT.md.
- Пути с пробелами (`Wider Face`, `Stanford Cars Dataset`) — в YAML/симлинках обязательно кавычки.
- `.gitignore` не покрывает `*.etlt` и `data/prep/` — добавить.
- `docs/about_models/*` и `docs/about_datasets/*` (нужны для шага 1 протокола) лежат на ветке `docs/dataset-specs`, не на `main`.

---

## Протокол одного эксперимента (повторяется в каждой модельной задаче)
1. **Read docs** — прочитать `docs/about_models/<model>.md` + `docs/about_datasets/<dataset>.md`; выписать в `results/<model>/EXPERIMENT.md` препроцессинг, классы/маппинг, пороги, риски и расхождения с legacy.
2. **Подготовка** — веса (overlay пути) + конвертер/sampling данных.
3. **Прогон** — `.venv/bin/python deploy/scripts/run_local.py --models <name>`.
4. **Фиксация** — дописать в `EXPERIMENT.md` измеренные числа + вердикт PASS/FAIL/UNDEF; коммит; merge в `main`.

Шаблон `EXPERIMENT.md` (создавать первым шагом каждой модельной задачи):
```markdown
# Эксперимент: <Model> × <Dataset>

Дата: 2026-06-04 · Ветка: exp/eval-<model> · GPU: RTX 3090

## 1. Выписка из docs (шаг 1 протокола)
- Модель: <docs/about_models/...> — препроцессинг (resize, цветовой порядок, mean/std, offsets), вход/выход, классы.
- Датасет: <docs/about_datasets/...> — формат GT, классы/маппинг, размер val.
- Пороги (PASS): <...>
- Риски / расхождения с legacy: <...>

## 2. Подготовка
- Веса: <abs path> · Данные: <abs path / prep>
- Конвертер/команда: <...>

## 3. Измеренные метрики
<вставить ключевые числа из results/<model>/metrics.json>

## 4. Вердикт
**PASS / FAIL / UNDEF** — <обоснование по порогам>
```

---

# Task 0: База — `exp/local-base` (раннер, конфиг путей, окружение)

**Files:**
- Create: `deploy/scripts/run_local.py`, `configs/local_paths.yaml`, `deploy/tests/conftest.py`, `deploy/tests/test_run_local.py`
- Modify: `.gitignore`, `deploy/requirements.txt`, `pyproject.toml`

- [ ] **Шаг 0.1: Подготовить ветки и доступность docs**

`docs/about_models/*` и `docs/about_datasets/*` нужны протоколу — они на `docs/dataset-specs`. Сначала влить docs-ветку в `main`, затем базовую ветку от обновлённого `main`.
```bash
cd /home/mk/CarCV-metrics
git status -sb                      # ожидается: M CLAUDE.md (рабочее изменение пользователя)
git stash push -m wip-claudemd CLAUDE.md   # временно убрать незакоммиченное, если мешает merge
git checkout main && git merge --no-ff docs/dataset-specs -m "docs: спеки моделей/датасетов + дизайн локальной кампании"
git checkout -b exp/local-base
```
Удалить/переименовать устаревшие ветки эпохи qudata (во избежание конфликта имён с новыми):
```bash
git branch -D exp/eval-trafficcamnet 2>/dev/null || true
git branch -m exp-eval-trafficcamnet-unprunned exp/legacy-trafficcamnet-unprunned 2>/dev/null || true
```
Ожидаемо: на ветке `exp/local-base`, `ls docs/about_models` показывает 8 `.md`.

- [ ] **Шаг 0.2: Установить недостающие зависимости**

```bash
.venv/bin/python -m pip --version 2>/dev/null || echo "pip отсутствует — используем uv"
uv pip install --python .venv/bin/python "scipy>=1.11" pytest
.venv/bin/python -c "import scipy, pytest; print('scipy', scipy.__version__, 'pytest', pytest.__version__)"
```
Ожидаемо: версии печатаются без ошибок.

- [ ] **Шаг 0.3: Симлинки результатов/графиков в репозиторные каталоги**

Эвалуаторы пишут в `ROOT/results` и `ROOT/plots` (`ROOT = deploy/`). Делаем их симлинками на репозиторные `results/`, `plots/` — без правок эвалуаторов.
```bash
cd /home/mk/CarCV-metrics
mkdir -p results plots
[ -e deploy/results ] || ln -s ../results deploy/results
[ -e deploy/plots ]   || ln -s ../plots   deploy/plots
ls -l deploy/results deploy/plots          # должны быть симлинками на ../results, ../plots
```

- [ ] **Шаг 0.4: Обновить `.gitignore`**

Добавить в конец `/home/mk/CarCV-metrics/.gitignore`:
```gitignore
# Локальная кампания: зашифрованные веса, prep-данные, deploy-симлинки
*.etlt
/data/prep/
/deploy/results
/deploy/plots
```
(`results/` и `plots/` верхнего уровня НЕ игнорируются — JSON/CSV/MD/PNG коммитятся.)

- [ ] **Шаг 0.5: Тест-каркас — `deploy/tests/conftest.py`**

```python
# deploy/tests/conftest.py
import sys
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1]          # /home/mk/CarCV-metrics/deploy
sys.path.insert(0, str(DEPLOY / "evaluation"))         # import evaluate / metrics / visualize
sys.path.insert(0, str(DEPLOY / "scripts"))            # import run_local / prep_*
```

- [ ] **Шаг 0.6: Написать падающий тест раннера — `deploy/tests/test_run_local.py`**

```python
import run_local


def test_overlay_config_only_allowed_keys():
    base = {"model_path": "models/x.onnx", "data_dir": "data/x",
            "results_dir": "results/x", "eval_fn": object()}
    overlay = {"model_path": "/abs/x.onnx", "data_dir": "/abs/data",
               "eval_fn": "EVIL", "junk": 1}
    out = run_local.overlay_config(base, overlay)
    assert out["model_path"] == "/abs/x.onnx"
    assert out["data_dir"] == "/abs/data"
    assert out["eval_fn"] is base["eval_fn"]      # eval_fn не перетирается overlay'ем
    assert "junk" not in out
    assert base["model_path"] == "models/x.onnx"  # исходник не мутирован


def test_select_models_all_expands():
    cfgs = {"a": {}, "b": {}}
    assert run_local.select_models(["all"], cfgs) == ["a", "b"]
    assert run_local.select_models(["b"], cfgs) == ["b"]


def test_load_paths_missing_returns_empty(tmp_path):
    assert run_local.load_paths(tmp_path / "nope.yaml") == {}
```

- [ ] **Шаг 0.7: Запустить тест — убедиться, что падает**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_run_local.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'run_local'`.

- [ ] **Шаг 0.8: Реализовать `deploy/scripts/run_local.py`**

```python
#!/usr/bin/env python3
"""Локальный раннер валидации (замена SSH-оркестрации run_remote.py).

Берёт EVAL_CONFIGS[name] из evaluate.py, накладывает абсолютные локальные пути
из configs/local_paths.yaml и вызывает cfg["eval_fn"](cfg). Результаты пишутся
в репозиторные results/<model>/ и plots/ (через симлинки deploy/results, deploy/plots).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]        # /home/mk/CarCV-metrics
EVAL_DIR = REPO_ROOT / "deploy" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))
from evaluate import EVAL_CONFIGS  # noqa: E402

DEFAULT_PATHS = REPO_ROOT / "configs" / "local_paths.yaml"
OVERLAY_KEYS = ("model_path", "labels_path", "data_dir", "results_dir",
                "conf_thr", "eval_classes")


def load_paths(path: Path) -> dict:
    """configs/local_paths.yaml → dict (пусто, если файла нет)."""
    if not Path(path).exists():
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def overlay_config(base_cfg: dict, overlay: dict) -> dict:
    """Копия base_cfg с наложенными разрешёнными ключами overlay (eval_fn неприкосновенен)."""
    cfg = dict(base_cfg)
    for k, v in (overlay or {}).items():
        if k in OVERLAY_KEYS:
            cfg[k] = v
    return cfg


def select_models(names: list[str], configs: dict) -> list[str]:
    if "all" in names:
        return list(configs.keys())
    return names


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CARS local evaluation runner")
    p.add_argument("--models", nargs="+",
                   choices=list(EVAL_CONFIGS.keys()) + ["all"], default=["all"])
    p.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    args = p.parse_args(argv)

    paths = load_paths(args.paths)
    rc = 0
    for name in select_models(args.models, EVAL_CONFIGS):
        cfg = overlay_config(EVAL_CONFIGS[name], paths.get(name, {}))
        print(f"─── local eval: {name} ───")
        result = cfg["eval_fn"](cfg)
        if "error" in result:
            print(f"{name}: ERROR/UNDEF → {result['error']}")
            rc = max(rc, 0)   # UNDEF не делает прогон фатальным
        else:
            print(f"{name}: {result.get('thresholds')}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Шаг 0.9: Запустить тест — убедиться, что проходит**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_run_local.py -q`
Expected: PASS (3 passed). Импорт `evaluate` подтянет cv2/onnxruntime — это нормально.

- [ ] **Шаг 0.10: Создать `configs/local_paths.yaml`**

```yaml
# configs/local_paths.yaml
# Абсолютные локальные пути к датасетам и весам (RTX 3090, сверено 2026-06-04).
# Накладывается поверх EVAL_CONFIGS в deploy/scripts/run_local.py (OVERLAY_KEYS).
# Веса/данные в git НЕ коммитятся. Пути с пробелами — в кавычках.

trafficcamnet:
  model_path: /home/mk/CarCV/models/trafficcamnet_pruned_onnx_v1.0.4/resnet18_trafficcamnet_pruned.onnx
  labels_path: /home/mk/CarCV/models/trafficcamnet_pruned_onnx_v1.0.4/labels.txt
  data_dir: /home/mk/CarCV-metrics/data/prep/trafficcamnet   # prep: labels.json + images/ (симлинк)

vehicletypenet:
  model_path: /home/mk/CarCV/models/vehicletypenet_pruned_onnx_v1.1.0/resnet18_pruned.onnx
  labels_path: /home/mk/CarCV/models/vehicletypenet_pruned_onnx_v1.1.0/labels.txt
  data_dir: /home/mk/CarCV-metrics/data/prep/vehicletypenet   # prep: test.json + images/ (Stanford train, с метками)

vehiclemakenet:
  model_path: /home/mk/CarCV/models/vehiclemakenet_pruned_onnx_v1.1.0/resnet18_pruned.onnx
  labels_path: /home/mk/CarCV/models/vehiclemakenet_pruned_onnx_v1.1.0/labels.txt
  data_dir: "/home/mk/Загрузки/DATASETS/VMMRdb"               # каталоги <make>_<model>_<year>

color:
  model_path: /home/mk/CarCV/models/bae_model_f3.onnx
  data_dir: /home/mk/CarCV/data/external/ymad_cars            # images_index.jsonl + images/

facedetect:
  model_path: /home/mk/CarCV-metrics/data/prep/facedetect/facenet.onnx   # получить (NGC/экспорт); иначе UNDEF
  data_dir: "/home/mk/Загрузки/DATASETS/Wider Face/WIDER_val"            # gt txt + images/<category>/

nomeroff_lpd:
  model_path: /home/mk/CarCV/models/nomeroff_net/object_detection/yolov26x-keypoints-2026-01-21.pt  # информационно; nomeroff-net грузит модель внутри
  data_dir: "/home/mk/Загрузки/DATASETS/nomeroff/autoriaNumberplateDataset-2021-05-12"

nomeroff_ocr:
  data_dir: "/home/mk/Загрузки/DATASETS/nomeroff/autoriaNumberplateOcrRu-2021-09-01"
```

- [ ] **Шаг 0.11: Добавить `scipy` в зависимости**

В `deploy/requirements.txt` добавить строку `scipy>=1.11`. В `pyproject.toml` в список зависимостей (рядом с `pyyaml>=6.0`) добавить `"scipy>=1.11"`.

- [ ] **Шаг 0.12: Smoke-проверка раннера (без данных модели — ожидаем аккуратную ошибку, не краш)**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python deploy/scripts/run_local.py --models trafficcamnet`
Expected: печатает `─── local eval: trafficcamnet ───`, затем `trafficcamnet: ERROR/UNDEF → dataset not found` (данные ещё не подготовлены) — раннер не падает, путь к ONNX/labels резолвится.

- [ ] **Шаг 0.13: Коммит и merge базы в `main`**

```bash
cd /home/mk/CarCV-metrics
git add deploy/scripts/run_local.py configs/local_paths.yaml deploy/tests/ .gitignore deploy/requirements.txt pyproject.toml
git commit -m "feat(local): локальный раннер run_local.py + local_paths.yaml + тест-каркас"
git checkout main && git merge --no-ff exp/local-base -m "merge: база локальной кампании (run_local, local_paths, симлинки)"
```
(Симлинки `deploy/results|plots` не коммитятся — они в `.gitignore`; создаются шагом 0.3 на каждой машине.)

---

# Task 1: TrafficCamNet × BDD100K — `exp/eval-trafficcamnet`

Готовый `eval_trafficcamnet` уже есть. Нужен конвертер BDD100K val JSON → формат пайплайна. Эта задача проверяет локальный пайплайн end-to-end.

**Files:**
- Create: `deploy/scripts/prep_bdd100k.py`, `deploy/tests/test_prep_bdd100k.py`, `results/trafficcamnet/EXPERIMENT.md`

- [ ] **Шаг 1.1: Ветка + шаг 1 протокола (read docs)**

```bash
cd /home/mk/CarCV-metrics && git checkout main && git checkout -b exp/eval-trafficcamnet
```
Прочитать `docs/about_models/trafficcamnet.md` + `docs/about_datasets/bdd100k.md`, создать `results/trafficcamnet/EXPERIMENT.md` по шаблону. Зафиксировать: вход 3×544×960, `/255`, BGR→RGB, offsets=0 (из `nvinfer_config.txt`), классы `car/bicycle/person/road_sign`, маппинг GT `TRAFFICCAMNET_GT_VOCAB` (BDD `bike|rider`→bicycle, `traffic sign`→road_sign), пороги P≥0.90 R≥0.85 F1≥0.87, `conf_thr=0.2` (cross-domain). Риск: cross-domain BDD→ожидаем низкие метрики (легаси-прогон давал FAIL — валидно).

- [ ] **Шаг 1.2: Падающий тест конвертера — `deploy/tests/test_prep_bdd100k.py`**

```python
import prep_bdd100k


def test_convert_bdd_to_labels_maps_box2d_and_filters():
    bdd = [{
        "name": "img1.jpg",
        "labels": [
            {"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
            {"category": "traffic sign", "box2d": {"x1": 5, "y1": 6, "x2": 7, "y2": 8}},
            {"category": "drivable area", "poly2d": [[0, 0]]},   # без box2d → отбрасывается
            {"category": "lane"},                                # без box2d → отбрасывается
        ],
    }]
    out = prep_bdd100k.convert_bdd_to_labels(bdd)
    assert len(out) == 1
    rec = out[0]
    assert rec["file_name"] == "img1.jpg"
    assert rec["image_id"] == "img1.jpg"
    cats = [d["category"] for d in rec["detections"]]
    assert cats == ["car", "traffic sign"]
    assert rec["detections"][0]["bbox2d"] == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
```

- [ ] **Шаг 1.3: Запустить — убедиться, что падает**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_prep_bdd100k.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'prep_bdd100k'`.

- [ ] **Шаг 1.4: Реализовать `deploy/scripts/prep_bdd100k.py`**

```python
#!/usr/bin/env python3
"""BDD100K val JSON → labels.json формата пайплайна + симлинк images/.

Формат на выходе: [{file_name, image_id, detections:[{category, bbox2d:{x1,y1,x2,y2}}]}].
Маппинг BDD→TrafficCamNet делает сам eval_trafficcamnet через TRAFFICCAMNET_GT_VOCAB —
здесь сохраняем сырые BDD-категории, только приводя box2d→bbox2d.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_bdd_to_labels(bdd_records: list) -> list:
    out = []
    for rec in bdd_records:
        dets = []
        for lab in rec.get("labels", []):
            box = lab.get("box2d")
            if not box:
                continue
            dets.append({
                "category": lab.get("category", ""),
                "bbox2d": {"x1": box["x1"], "y1": box["y1"],
                           "x2": box["x2"], "y2": box["y2"]},
            })
        out.append({"file_name": rec["name"], "image_id": rec["name"],
                    "detections": dets})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd-json", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    records = json.loads(args.bdd_json.read_text())
    labels = convert_bdd_to_labels(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "labels.json").write_text(json.dumps(labels))

    link = args.out_dir / "images"
    if not link.exists():
        link.symlink_to(args.images_dir)
    print(f"labels.json: {len(labels)} записей · images → {args.images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Шаг 1.5: Запустить тест — PASS**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_prep_bdd100k.py -q`
Expected: PASS (1 passed).

- [ ] **Шаг 1.6: Подготовить данные (шаг 2 протокола)**

```bash
cd /home/mk/CarCV-metrics
.venv/bin/python deploy/scripts/prep_bdd100k.py \
  --bdd-json "/home/mk/Загрузки/DATASETS/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json" \
  --images-dir "/home/mk/Загрузки/DATASETS/bdd100k/bdd100k/bdd100k/images/100k/val" \
  --out-dir data/prep/trafficcamnet
ls data/prep/trafficcamnet           # labels.json + images (симлинк)
```
Expected: `labels.json: 10000 записей · images → ...val`.

- [ ] **Шаг 1.7: Прогон (шаг 3 протокола)**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python deploy/scripts/run_local.py --models trafficcamnet`
Expected: лог `Loaded resnet18_trafficcamnet_pruned.onnx on CUDAExecutionProvider`, прогресс-бар по 10000 img, финал `trafficcamnet: {...status...}`; создан `results/trafficcamnet/metrics.json`.

- [ ] **Шаг 1.8: Зафиксировать вердикт (шаг 4) + коммит/merge**

Прочитать `results/trafficcamnet/metrics.json`, вписать P/R/F1 (car + macro) в `EXPERIMENT.md`, проставить **PASS/FAIL** по порогам (ожидаемо FAIL в cross-domain — валидно).
```bash
git add deploy/scripts/prep_bdd100k.py deploy/tests/test_prep_bdd100k.py results/trafficcamnet/
git commit -m "feat(eval): TrafficCamNet × BDD100K — конвертер + прогон + вердикт"
git checkout main && git merge --no-ff exp/eval-trafficcamnet -m "merge: TrafficCamNet × BDD100K"
```

---

# Task 2: VehicleTypeNet × Stanford Cars — `exp/eval-vehicletypenet`

`eval_vehicletypenet` уже умеет читать `data_dir/test.json` (записи `{file_name,label}`, тип кузова выводит `derive_typenet_label`). Нужен конвертер из devkit `.mat`. Тест Stanford без меток → берём **train split** (с метками); зафиксировать в EXPERIMENT.md.

**Files:**
- Create: `deploy/scripts/prep_stanford_cars.py`, `deploy/tests/test_prep_stanford.py`, `results/vehicletypenet/EXPERIMENT.md`

- [ ] **Шаг 2.1: Ветка + read docs**

```bash
cd /home/mk/CarCV-metrics && git checkout main && git checkout -b exp/eval-vehicletypenet
```
Прочитать `docs/about_models/vehicletypenet.md` + `docs/about_datasets/stanford_cars.md`; создать `results/vehicletypenet/EXPERIMENT.md`. Зафиксировать: вход 224, `preprocess_tao_bgr` offsets=(104,117,124), классы `coupe/largevehicle/sedan/suv/truck/van`, маппинг `STANFORD_BODY_KEYWORDS` (напр. `convertible→coupe`, `wagon/hatchback→sedan`, `cab→truck`), порог Top-1≥0.85. **Источник меток: train split** (`cars_train_annos.mat`+`cars_meta.mat`) т.к. `cars_test_annos_withlabels.mat` отсутствует. Риск: легаси давал Top1≈0.36 (FAIL) — меряем заново, FAIL валиден.

- [ ] **Шаг 2.2: Падающий тест конвертера — `deploy/tests/test_prep_stanford.py`**

```python
import numpy as np
import prep_stanford_cars


def test_build_stanford_records_joins_class_names():
    # class_names[idx] (0-based); annos: (fname, class_1based)
    class_names = ["Acura RL Sedan 2012", "Audi TT Coupe 2012"]
    annos = [("00001.jpg", 1), ("00002.jpg", 2), ("00003.jpg", 1)]
    recs = prep_stanford_cars.build_stanford_records(class_names, annos)
    assert recs == [
        {"file_name": "00001.jpg", "label": "Acura RL Sedan 2012"},
        {"file_name": "00002.jpg", "label": "Audi TT Coupe 2012"},
        {"file_name": "00003.jpg", "label": "Acura RL Sedan 2012"},
    ]


def test_parse_mat_class_names_and_annos(tmp_path):
    from scipy.io import savemat
    savemat(tmp_path / "meta.mat",
            {"class_names": np.array([["Acura RL Sedan 2012", "Audi TT Coupe 2012"]],
                                     dtype=object)})
    # annos: поля bbox + class + fname; нам нужны class и fname
    annos = np.zeros((2,), dtype=[("class", "O"), ("fname", "O")])
    annos[0] = (np.array([[1]]), np.array(["00001.jpg"]))
    annos[1] = (np.array([[2]]), np.array(["00002.jpg"]))
    savemat(tmp_path / "annos.mat", {"annotations": annos})

    names = prep_stanford_cars.parse_class_names(tmp_path / "meta.mat")
    assert names[0] == "Acura RL Sedan 2012"
    pairs = prep_stanford_cars.parse_annos(tmp_path / "annos.mat")
    assert pairs[0] == ("00001.jpg", 1)
```

- [ ] **Шаг 2.3: Запустить — убедиться, что падает**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_prep_stanford.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'prep_stanford_cars'`.

- [ ] **Шаг 2.4: Реализовать `deploy/scripts/prep_stanford_cars.py`**

```python
#!/usr/bin/env python3
"""Stanford Cars devkit (.mat) → test.json + симлинк images/.

test.json: [{file_name, label:"<Make> <Model> <BodyType> <Year>"}].
Тип кузова выводит сам eval_vehicletypenet (derive_typenet_label).
По умолчанию используем TRAIN split (с метками) — у официального test нет меток.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.io import loadmat


def parse_class_names(meta_mat: Path) -> list[str]:
    m = loadmat(str(meta_mat))
    raw = m["class_names"][0]
    return [str(x[0]) for x in raw]


def parse_annos(annos_mat: Path) -> list[tuple[str, int]]:
    """Возвращает [(fname, class_1based)] из cars_*_annos.mat."""
    m = loadmat(str(annos_mat))
    ann = m["annotations"][0]
    out = []
    for rec in ann:
        # поля Stanford: bbox_x1,y1,x2,y2, class, fname (имена/порядок устойчивы)
        cls = int(rec["class"][0][0]) if "class" in rec.dtype.names else int(rec[-2][0][0])
        fname = str(rec["fname"][0]) if "fname" in rec.dtype.names else str(rec[-1][0])
        out.append((fname, cls))
    return out


def build_stanford_records(class_names: list[str],
                           annos: list[tuple[str, int]]) -> list[dict]:
    return [{"file_name": fn, "label": class_names[cls - 1]} for fn, cls in annos]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-mat", required=True, type=Path)
    ap.add_argument("--annos-mat", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    names = parse_class_names(args.meta_mat)
    annos = parse_annos(args.annos_mat)
    recs = build_stanford_records(names, annos)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "test.json").write_text(json.dumps(recs))
    link = args.out_dir / "images"
    if not link.exists():
        link.symlink_to(args.images_dir)
    print(f"test.json: {len(recs)} записей · {len(names)} классов · images → {args.images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Шаг 2.5: Запустить тест — PASS**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_prep_stanford.py -q`
Expected: PASS (2 passed). Если поля `.mat` называются иначе — починить `parse_annos` по фактической `rec.dtype.names` (вывести `print(loadmat(...)["annotations"][0][0].dtype.names)`), тест ловит регресс.

- [ ] **Шаг 2.6: Подготовить данные**

```bash
cd /home/mk/CarCV-metrics
.venv/bin/python deploy/scripts/prep_stanford_cars.py \
  --meta-mat "/home/mk/Загрузки/DATASETS/Stanford Cars Dataset/car_devkit/devkit/cars_meta.mat" \
  --annos-mat "/home/mk/Загрузки/DATASETS/Stanford Cars Dataset/car_devkit/devkit/cars_train_annos.mat" \
  --images-dir "/home/mk/Загрузки/DATASETS/Stanford Cars Dataset/cars_train/cars_train" \
  --out-dir data/prep/vehicletypenet
```
Expected: `test.json: 8144 записей · 196 классов · images → .../cars_train`.

- [ ] **Шаг 2.7: Прогон + вердикт + коммит/merge**

```bash
.venv/bin/python deploy/scripts/run_local.py --models vehicletypenet
```
Expected: прогресс по ~8144 img; `results/vehicletypenet/metrics.json` + `plots/vehicletypenet_confusion.png` + `data/prep/vehicletypenet/type_mapping.csv`. Вписать Top-1 в `EXPERIMENT.md`, вердикт PASS/FAIL.
```bash
git add deploy/scripts/prep_stanford_cars.py deploy/tests/test_prep_stanford.py results/vehicletypenet/
git commit -m "feat(eval): VehicleTypeNet × Stanford Cars (train split) — конвертер + прогон + вердикт"
git checkout main && git merge --no-ff exp/eval-vehicletypenet -m "merge: VehicleTypeNet × Stanford Cars"
```

---

# Task 3: VehicleMakeNet × VMMRdb — `exp/eval-vehiclemakenet`

Нужно добавить ветку чтения VMMRdb (каталоги `<make>_<model>_<year>`) в `eval_vehiclemakenet`, сохранив существующий MAD-Cars путь. Марка = первый токен; `normalize_brand`; OOD (не в 20 NGC-марках) — пропуск.

**Files:**
- Modify: `deploy/evaluation/evaluate.py` (добавить `discover_vmmrdb_samples`, переписать тело `eval_vehiclemakenet` на сбор `samples` + ветку VMMRdb)
- Create: `deploy/tests/test_vmmrdb_reader.py`, `results/vehiclemakenet/EXPERIMENT.md`

- [ ] **Шаг 3.1: Ветка + read docs**

```bash
cd /home/mk/CarCV-metrics && git checkout main && git checkout -b exp/eval-vehiclemakenet
```
Прочитать `docs/about_models/vehiclemakenet.md` + `docs/about_datasets/vmmrdb.md`; создать `results/vehiclemakenet/EXPERIMENT.md`. Зафиксировать: вход 224 `preprocess_tao_bgr` offsets=(104,117,124), 20 NGC-марок, маппинг VMMRdb (марка=первый токен каталога), OOD-пропуск, пороги Top-1≥0.70 Top-3≥0.85. Риск: VMMRdb крупнее/иной домен → возможен FAIL (валиден); многие каталоги OOD → большой `skipped_ood`.

- [ ] **Шаг 3.2: Падающий тест ридера — `deploy/tests/test_vmmrdb_reader.py`**

```python
import evaluate


def test_discover_vmmrdb_samples_filters_ood(tmp_path):
    (tmp_path / "honda_civic_2015").mkdir()
    (tmp_path / "honda_civic_2015" / "a.jpg").write_bytes(b"x")
    (tmp_path / "toyota_camry_2016").mkdir()
    (tmp_path / "toyota_camry_2016" / "b.jpg").write_bytes(b"x")
    (tmp_path / "tesla_model3_2018").mkdir()          # OOD (нет в 20 NGC)
    (tmp_path / "tesla_model3_2018" / "c.jpg").write_bytes(b"x")

    samples = evaluate.discover_vmmrdb_samples(tmp_path, evaluate.NGC_MAKES_LOWER)
    brands = sorted({b for _, b in samples})
    assert brands == ["honda", "toyota"]              # tesla отфильтрована
    assert all(p.suffix == ".jpg" for p, _ in samples)
    assert len(samples) == 2
```

- [ ] **Шаг 3.3: Запустить — убедиться, что падает**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_vmmrdb_reader.py -q`
Expected: FAIL — `AttributeError: module 'evaluate' has no attribute 'discover_vmmrdb_samples'`.

- [ ] **Шаг 3.4: Добавить `discover_vmmrdb_samples` в `evaluate.py`**

Вставить сразу после `normalize_brand` (после строки 318):
```python
def discover_vmmrdb_samples(data_dir: Path, ngc_makes_lower) -> list:
    """VMMRdb: каталоги <make>_<model>_<year>. brand = первый токен (normalize_brand).
    OOD (марка не в 20 NGC) пропускается. Возвращает [(img_path, brand)]."""
    samples = []
    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        brand = normalize_brand(cls_dir.name.split("_")[0])
        if brand not in ngc_makes_lower:
            continue
        for img in cls_dir.glob("*.jpg"):
            samples.append((img, brand))
    return samples
```

- [ ] **Шаг 3.5: Переписать тело `eval_vehiclemakenet` (строки 321–394) на сбор `samples` + ветку VMMRdb**

Заменить всё тело функции `eval_vehiclemakenet` на:
```python
def eval_vehiclemakenet(cfg: dict) -> dict:
    model_path = ROOT / cfg["model_path"]
    labels_path = ROOT / cfg["labels_path"]
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"VehicleMakeNet model not found: {model_path}")
        return {"error": "model not found"}

    labels = load_labels(labels_path)
    labels_norm = [l.strip().lower() for l in labels]
    sess = get_ort_session(str(model_path))
    input_name = sess.get_inputs()[0].name

    # Источник данных: MAD-Cars (sample_5k.json) ИЛИ VMMRdb (каталоги-по-классам)
    madcars_meta = data_dir / "sample_5k.json"
    samples: list[tuple[Path, str]] = []
    if madcars_meta.exists():
        source = "mad_cars"
        with open(madcars_meta) as f:
            sample = json.load(f)
        images_dir = data_dir / "images"
        for item in sample:
            samples.append((images_dir / item["file_name"],
                            normalize_brand(item["brand"])))
    elif data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()):
        source = "vmmrdb"
        samples = discover_vmmrdb_samples(data_dir, NGC_MAKES_LOWER)
    else:
        log.error(f"VehicleMakeNet: no sample_5k.json and no class dirs in {data_dir}")
        return {"error": "dataset not found"}

    if not samples:
        return {"error": "no images"}

    predictions, ground_truths = [], []
    skipped_oo_dist = 0
    for img_path, gt_label in tqdm(samples, desc="VehicleMakeNet", unit="img"):
        if gt_label not in NGC_MAKES_LOWER:
            skipped_oo_dist += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        inp = preprocess_tao_bgr(img, size=224, offsets=(104.0, 117.0, 124.0))
        out = sess.run(None, {input_name: inp})[0][0]
        probs = softmax(out)
        top_k = [labels_norm[i] for i in probs.argsort()[::-1][:3]]
        predictions.append({"image_id": str(img_path), "top_k": top_k})
        ground_truths.append({"image_id": str(img_path), "label": gt_label})

    log.info(f"VehicleMakeNet[{source}]: {len(predictions)} imgs, "
             f"skipped {skipped_oo_dist} out-of-distribution (not in 20 NGC makes)")

    m = compute_classification_metrics(predictions, ground_truths)
    thresholds = {"top1_accuracy": 0.70, "top3_accuracy": 0.85}
    status = check_thresholds(m.to_dict(), thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status,
              "source": source, "skipped_ood": skipped_oo_dist}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    import csv
    with open(results_dir / "per_class_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy"])
        for cls, acc in sorted(m.per_class_accuracy.items()):
            w.writerow([cls, acc])

    if m.per_class_accuracy:
        plot_per_class_accuracy(
            m.per_class_accuracy, "VehicleMakeNet", 0.70,
            str(ROOT / "plots" / "vehiclemakenet_per_class.png"),
        )

    log.info(f"VehicleMakeNet: Top1={m.top1_accuracy:.3f} Top3={m.top3_accuracy:.3f}")
    return result
```

- [ ] **Шаг 3.6: Запустить тест — PASS**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_vmmrdb_reader.py -q`
Expected: PASS (1 passed).

- [ ] **Шаг 3.7: Прогон + вердикт + коммит/merge**

```bash
.venv/bin/python deploy/scripts/run_local.py --models vehiclemakenet
```
Expected: `VehicleMakeNet[vmmrdb]: N imgs, skipped M out-of-distribution`; `results/vehiclemakenet/metrics.json` (+ `source:"vmmrdb"`, `skipped_ood`) + per-class CSV/PNG. Вписать Top-1/Top-3 + `skipped_ood` в `EXPERIMENT.md`, вердикт.
```bash
git add deploy/evaluation/evaluate.py deploy/tests/test_vmmrdb_reader.py results/vehiclemakenet/
git commit -m "feat(eval): VehicleMakeNet × VMMRdb — ветка чтения каталогов-по-классам + прогон"
git checkout main && git merge --no-ff exp/eval-vehiclemakenet -m "merge: VehicleMakeNet × VMMRdb"
```

---

# Task 4: Color (bae_model_f3) × MAD-Cars — `exp/eval-color`

Новый эвалуатор `eval_color`: EfficientNet-B3, вход `[N,3,384,384]`, выход `[N,15]` **логиты** (softmax в постобработке). 15 классов в алфавитном порядке. GT — `HEX_TO_CARS_COLOR[row.color]`, dedup по `car_id`.

**Files:**
- Modify: `deploy/evaluation/evaluate.py` (добавить `COLOR_CLASSES`, `COLOR_MEAN/STD`, `HEX_TO_CARS_COLOR`, `preprocess_color`, `load_madcars_color_index`, `eval_color`; запись `color` в `EVAL_CONFIGS`)
- Create: `deploy/tests/test_eval_color.py`, `results/color/EXPERIMENT.md`

- [ ] **Шаг 4.1: Ветка + read docs**

```bash
cd /home/mk/CarCV-metrics && git checkout main && git checkout -b exp/eval-color
```
Прочитать `docs/about_models/bae_model_f3.md` + `docs/about_datasets/mad_cars.md` + System Design §6.5; создать `results/color/EXPERIMENT.md`. Зафиксировать: вход 384×384, BGR→RGB `/255`, mean=[0.43,0.40,0.39] std=[0.27,0.26,0.26] (нестандартные, непроверяемы), NCHW; **выход — логиты → softmax в постобработке**; 15 классов алфавит (`beige…yellow`), маппинг индекс→класс **непроверенный**; GT из hex (`HEX_TO_CARS_COLOR`); dedup по `car_id`; подаём resize 384×384 **без** TrafficCamNet-кропа. Пороги: overall≥0.80, best(black/white/red/blue)≥0.90, challenging(beige/tan/gold/silver)≥0.70. Caveats: `tan` отсутствует в данных (покрытие 14/15), `pink`≈0.9%.

- [ ] **Шаг 4.2: Падающий тест — `deploy/tests/test_eval_color.py`**

```python
import numpy as np
import evaluate


def test_color_classes_alphabetical_15():
    assert evaluate.COLOR_CLASSES == [
        "beige", "black", "blue", "brown", "gold", "green", "grey", "orange",
        "pink", "purple", "red", "silver", "tan", "white", "yellow"]
    assert len(evaluate.COLOR_CLASSES) == 15


def test_hex_to_cars_color_mapping():
    h = evaluate.HEX_TO_CARS_COLOR
    assert h["000000"] == "black"
    assert h["ffffff"] == "white"
    assert h["9966cc"] == "purple"
    assert h["0088ff"] == "blue"      # второй синий hex
    assert "tan" not in h.values()    # tan отсутствует в данных (покрытие 14/15)


def test_preprocess_color_shape_and_norm():
    img = np.full((10, 10, 3), 127, dtype=np.uint8)   # BGR
    inp = evaluate.preprocess_color(img, size=384)
    assert inp.shape == (1, 3, 384, 384)
    assert inp.dtype == np.float32
    # значение = ((127/255) - mean)/std по каналам RGB
    expected_r = ((127 / 255.0) - 0.43) / 0.27
    assert np.allclose(inp[0, 0].mean(), expected_r, atol=1e-3)


def test_load_madcars_color_index_dedups_by_car_id(tmp_path):
    jl = tmp_path / "images_index.jsonl"
    jl.write_text(
        '{"image_id":"0_0","image_path":"images/0_0.jpg","car_id":"0","color":"000000"}\n'
        '{"image_id":"0_1","image_path":"images/0_1.jpg","car_id":"0","color":"000000"}\n'
        '{"image_id":"1_0","image_path":"images/1_0.jpg","car_id":"1","color":"ffffff"}\n')
    rows = evaluate.load_madcars_color_index(jl, dedup=True)
    assert len(rows) == 2                       # один view на car_id
    assert {r["car_id"] for r in rows} == {"0", "1"}
```

- [ ] **Шаг 4.3: Запустить — убедиться, что падает**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_eval_color.py -q`
Expected: FAIL — `AttributeError: module 'evaluate' has no attribute 'COLOR_CLASSES'`.

- [ ] **Шаг 4.4: Добавить константы и хелперы Color в `evaluate.py`**

Вставить рядом с другими константами (после `IMAGENET_STD`, ~строка 44):
```python
COLOR_CLASSES = [
    "beige", "black", "blue", "brown", "gold", "green", "grey", "orange",
    "pink", "purple", "red", "silver", "tan", "white", "yellow",
]
COLOR_MEAN = np.array([0.43, 0.40, 0.39], dtype=np.float32)
COLOR_STD = np.array([0.27, 0.26, 0.26], dtype=np.float32)

# MAD-Cars hex → класс цвета CARS (docs/about_datasets/mad_cars.md §HEX_TO_CARS_COLOR).
# 16 hex → 14 имён (нет tan; два hex для blue и red).
HEX_TO_CARS_COLOR = {
    "000000": "black", "ffffff": "white", "9c9999": "grey", "cacecb": "silver",
    "0000ff": "blue", "0088ff": "blue", "ff0000": "red", "cc0033": "red",
    "926547": "brown", "34ba2b": "green", "ffefd5": "beige",
    "ff9966": "orange", "9966cc": "purple", "fde910": "yellow",
    "ffcc00": "gold", "ffc0cb": "pink",
}


def preprocess_color(img_bgr: np.ndarray, size: int = 384) -> np.ndarray:
    """Color (bae_model_f3): BGR→RGB, /255, (x-mean)/std, NCHW. mean/std из System Design §6.5."""
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - COLOR_MEAN) / COLOR_STD
    return img.transpose(2, 0, 1)[None]


def load_madcars_color_index(jsonl_path, dedup: bool = True) -> list:
    """images_index.jsonl → список строк; dedup по car_id (1 view/car) для headline-метрики."""
    rows, seen = [], set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if dedup:
                cid = r.get("car_id")
                if cid in seen:
                    continue
                seen.add(cid)
            rows.append(r)
    return rows
```

- [ ] **Шаг 4.5: Добавить `eval_color` в `evaluate.py`** (рядом с другими `eval_*`, до `EVAL_CONFIGS`)

```python
def eval_color(cfg: dict) -> dict:
    model_path = ROOT / cfg["model_path"]
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"Color model not found: {model_path}")
        return {"error": "model not found"}

    index_path = data_dir / "images_index.jsonl"
    if not index_path.exists():
        log.error(f"MAD-Cars index not found: {index_path}")
        return {"error": "dataset not found"}

    rows = load_madcars_color_index(index_path, dedup=True)
    sess = get_ort_session(str(model_path))
    input_name = sess.get_inputs()[0].name

    predictions, ground_truths = [], []
    skipped = 0
    cm = np.zeros((len(COLOR_CLASSES), len(COLOR_CLASSES)), dtype=int)

    for r in tqdm(rows, desc="Color", unit="img"):
        gt = HEX_TO_CARS_COLOR.get((r.get("color") or "").lower())
        if gt is None:
            skipped += 1
            continue
        img_path = data_dir / r["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        inp = preprocess_color(img, size=384)
        logits = sess.run(None, {input_name: inp})[0][0]
        probs = softmax(logits)            # граф отдаёт логиты — softmax в постобработке
        order = probs.argsort()[::-1]
        top_k = [COLOR_CLASSES[i] for i in order[:3]]
        predictions.append({"image_id": r["image_id"], "top_k": top_k})
        ground_truths.append({"image_id": r["image_id"], "label": gt})
        cm[COLOR_CLASSES.index(gt), COLOR_CLASSES.index(top_k[0])] += 1

    if not predictions:
        return {"error": "no images"}

    m = compute_classification_metrics(predictions, ground_truths)
    md = m.to_dict()
    pca = md["per_class_accuracy"]
    best = ["black", "white", "red", "blue"]
    challenging = ["beige", "tan", "gold", "silver"]
    best_min = min([pca[c] for c in best if c in pca], default=0.0)
    chal_min = min([pca[c] for c in challenging if c in pca], default=0.0)

    metrics_for_thr = {"overall": md["top1_accuracy"],
                       "best_group_min": best_min,
                       "challenging_group_min": chal_min}
    thresholds = {"overall": 0.80, "best_group_min": 0.90,
                  "challenging_group_min": 0.70}
    status = check_thresholds(metrics_for_thr, thresholds)

    covered = sorted({g["label"] for g in ground_truths})
    result = {
        "metrics": {**md, "best_group_min": best_min,
                    "challenging_group_min": chal_min,
                    "coverage_classes": covered,
                    "coverage": f"{len(covered)}/15"},
        "thresholds": status, "skipped": skipped,
        "model": "bae_model_f3 EfficientNet-B3 (color)",
    }
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    import csv
    with open(results_dir / "per_class_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy"])
        for cls in COLOR_CLASSES:
            w.writerow([cls, pca.get(cls, "")])

    if cm.sum() > 0:
        plot_confusion_matrix(cm, COLOR_CLASSES, "Color (bae_model_f3)",
                              str(ROOT / "plots" / "color_confusion.png"))
    if pca:
        plot_per_class_accuracy(pca, "Color (bae_model_f3)", 0.80,
                                str(ROOT / "plots" / "color_per_class.png"))

    log.info(f"Color: Top1={m.top1_accuracy:.3f} best_min={best_min:.3f} "
             f"chal_min={chal_min:.3f} coverage={len(covered)}/15 skipped={skipped}")
    return result
```

- [ ] **Шаг 4.6: Зарегистрировать `color` в `EVAL_CONFIGS`** (внутри dict, ~строка 1005)

```python
    "color": {
        "model_path": "models/color/bae_model_f3.onnx",   # перекрывается local_paths.yaml
        "data_dir": "data/mad_cars",                        # перекрывается local_paths.yaml
        "results_dir": "results/color",
        "eval_fn": eval_color,
    },
```

- [ ] **Шаг 4.7: Запустить тест — PASS**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_eval_color.py -q`
Expected: PASS (4 passed).

- [ ] **Шаг 4.8: Прогон + вердикт + коммит/merge**

```bash
.venv/bin/python deploy/scripts/run_local.py --models color
```
Expected: `Color: Top1=... best_min=... chal_min=... coverage=14/15`; `results/color/metrics.json` + `plots/color_confusion.png` + per-class CSV. Вписать числа + покрытие 14/15 в `EXPERIMENT.md`; вердикт (вероятен «осторожный»/заниженный — валиден из-за непроверенного маппинга индексов и нестандартных mean/std).
```bash
git add deploy/evaluation/evaluate.py deploy/tests/test_eval_color.py results/color/
git commit -m "feat(eval): Color (bae_model_f3) × MAD-Cars — новый eval_color + прогон"
git checkout main && git merge --no-ff exp/eval-color -m "merge: Color × MAD-Cars"
```

---

# Task 5: nomeroff_lpd × AutoRia detection — `exp/eval-nomeroff-lpd`

`eval_nomeroff_lpd` готов (через `nomeroff-net pipeline`). Главный риск — установка `nomeroff-net` на Python 3.13 / torch 2.11. Если не встаёт — **deferred** (не «выдуманный PASS»). Локальные веса есть (`yolov26x-keypoints*.pt` + `yolov11x-keypoints*.onnx`).

**Files:**
- Create: `results/nomeroff_lpd/EXPERIMENT.md`

- [ ] **Шаг 5.1: Ветка + read docs**

```bash
cd /home/mk/CarCV-metrics && git checkout main && git checkout -b exp/eval-nomeroff-lpd
```
Прочитать `docs/about_models/nomeroff_lpd.md` + `docs/about_datasets/autoria_numberplate_detection_ru.md`; создать `results/nomeroff_lpd/EXPERIMENT.md`. Зафиксировать: формат GT VIA (`val/via_region_data.json`, `regions[].shape_attributes` rect/polygon), пороги P≥0.70 R≥0.80, conf_thr=0.3 (в коде), локальные веса `yolov26x-keypoints-2026-01-21.pt`. Расхождение: `models.md` указывает локальный `.pt`, но `eval_nomeroff_lpd` грузит модель внутри `pipeline("number_plate_localization")` (свой кэш) — зафиксировать, чем именно меряли.

- [ ] **Шаг 5.2: Установить `nomeroff-net` (рискованно)**

```bash
cd /home/mk/CarCV-metrics
uv pip install --python .venv/bin/python nomeroff-net 2>&1 | tail -20 || echo "INSTALL FAILED"
.venv/bin/python -c "import nomeroff_net; print('nomeroff_net OK')" 2>&1 | tail -3
```
Если импорт **успешен** → шаг 5.3. Если **падает** (несовместимость py3.13/torch2.11) → зафиксировать в `EXPERIMENT.md` вердикт **deferred** (с текстом ошибки), закоммитить EXPERIMENT.md, merge, перейти к Task 7 (OCR — Task 6 — также deferred на той же причине).

- [ ] **Шаг 5.3: Прогон**

```bash
.venv/bin/python deploy/scripts/run_local.py --models nomeroff_lpd
```
Expected: `Nomeroff LPD: processed N images, skipped M`, `results/nomeroff_lpd/metrics.json` с P/R/F1. (Если `pipeline` грузит дефолтные веса, а не локальный `.pt` — отметить в EXPERIMENT.md; опционально направить nomeroff-net на локальные веса через её кэш/конфиг, но без долгих раскопок.)

- [ ] **Шаг 5.4: Вердикт + коммит/merge**

Вписать P/R/F1 (или deferred) в `EXPERIMENT.md`, вердикт PASS/FAIL/deferred.
```bash
git add results/nomeroff_lpd/
git commit -m "feat(eval): nomeroff_lpd × AutoRia detection — прогон/вердикт"
git checkout main && git merge --no-ff exp/eval-nomeroff-lpd -m "merge: nomeroff_lpd × AutoRia detection"
```

---

# Task 6: nomeroff_ocr × AutoRia OCR — `exp/eval-nomeroff-ocr`

`eval_nomeroff_ocr` готов (включая обход `torch.load(weights_only=False)` для torch≥2.6). Зависит от той же установки `nomeroff-net`.

**Files:**
- Create: `results/nomeroff_ocr/EXPERIMENT.md`

- [ ] **Шаг 6.1: Ветка + read docs**

```bash
cd /home/mk/CarCV-metrics && git checkout main && git checkout -b exp/eval-nomeroff-ocr
```
Прочитать `docs/about_models/nomeroff_ocr.md` + `docs/about_datasets/autoria_numberplate_ocr_ru.md`; создать `results/nomeroff_ocr/EXPERIMENT.md`. Зафиксировать: вход — готовые кропы (`val/img`+`val/ann`), GT из `description`/`name` (upper), метрики char_acc (позиционное совпадение, не edit-distance — отметить!) и full_plate_acc, пороги char≥0.90 plate≥0.80.

- [ ] **Шаг 6.2: Прогон (если nomeroff-net установлен в Task 5)**

```bash
.venv/bin/python deploy/scripts/run_local.py --models nomeroff_ocr
```
Expected: `Nomeroff OCR: CharAcc=... PlateAcc=...`, `results/nomeroff_ocr/metrics.json`. Если nomeroff-net не установлен → вердикт **deferred** (как в Task 5).

- [ ] **Шаг 6.3: Вердикт + коммит/merge**

```bash
git add results/nomeroff_ocr/
git commit -m "feat(eval): nomeroff_ocr × AutoRia OCR — прогон/вердикт"
git checkout main && git merge --no-ff exp/eval-nomeroff-ocr -m "merge: nomeroff_ocr × AutoRia OCR"
```

---

# Task 7: FaceDetect × WIDER FACE — `exp/eval-facedetect`

Самый рискованный. Локально только зашифрованный `model.etlt`. Новый `eval_facedetect` (DetectNet_v2, вход 736×416, 1 класс `face`, `detectnet_v2_decode`). Парсер `wider_face_val_bbx_gt.txt`. Easy/Med/Hard официальный split **отсутствует** → headline = overall AP@0.5 + automotive-срез; Easy/Med/Hard либо неофициальная аппроксимация, либо UNDEF.

**Files:**
- Create: `deploy/scripts/get_facenet_onnx.sh`, `deploy/tests/test_eval_facedetect.py`, `results/facedetect/EXPERIMENT.md`
- Modify: `deploy/evaluation/evaluate.py` (добавить `parse_wider_gt`, `eval_facedetect`; запись `facedetect` в `EVAL_CONFIGS`; защитить `check_all`)

- [ ] **Шаг 7.1: Ветка + read docs**

```bash
cd /home/mk/CarCV-metrics && git checkout main && git checkout -b exp/eval-facedetect
```
Прочитать `docs/about_models/facedetect.md` + `docs/about_datasets/wider_face.md`; создать `results/facedetect/EXPERIMENT.md`. Зафиксировать: вход 736×416 BGR→RGB `/255` без offsets, `detectnet_v2_decode(target_cls=0,stride=16,bbox_norm=35.0)`+NMS 0.5; формат txt (`x1 y1 w h blur expression illumination invalid occlusion pose` — `invalid` ПЕРЕД occlusion!), bbox xywh→xyxy; пороги AP@0.5 Easy≥0.80 Med≥0.70 Hard≥0.50 + automotive-срез (`14--Traffic`,`5--Car_Accident`,`59--people--driving--car`); риск: ONNX нет → вероятен UNDEF; официального Easy/Med/Hard split нет.

- [ ] **Шаг 7.2: Падающий тест парсера — `deploy/tests/test_eval_facedetect.py`**

```python
import evaluate


def test_parse_wider_gt_xywh_to_xyxy_and_attrs(tmp_path):
    txt = tmp_path / "gt.txt"
    txt.write_text(
        "0--Parade/0_Parade_1.jpg\n"
        "2\n"
        "10 20 30 40 1 0 0 0 2 0 \n"
        "5 5 0 0 0 0 0 1 0 0 \n"          # w=h=0 → отбрасывается
        "14--Traffic/14_t.jpg\n"
        "1\n"
        "0 0 8 8 0 0 0 0 0 0 \n")
    gt = evaluate.parse_wider_gt(txt)
    assert set(gt.keys()) == {"0--Parade/0_Parade_1.jpg", "14--Traffic/14_t.jpg"}
    boxes = gt["0--Parade/0_Parade_1.jpg"]["boxes"]
    assert boxes == [[10, 20, 40, 60]]    # [x1,y1,x1+w,y1+h]; нулевой бокс отброшен
    attrs = gt["0--Parade/0_Parade_1.jpg"]["attrs"]
    assert attrs[0]["blur"] == 1 and attrs[0]["invalid"] == 0 and attrs[0]["occlusion"] == 2
```

- [ ] **Шаг 7.3: Запустить — убедиться, что падает**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_eval_facedetect.py -q`
Expected: FAIL — `AttributeError: module 'evaluate' has no attribute 'parse_wider_gt'`.

- [ ] **Шаг 7.4: Добавить `parse_wider_gt` в `evaluate.py`**

```python
def parse_wider_gt(txt_path) -> dict:
    """wider_face_val_bbx_gt.txt → {rel_path: {"boxes":[[x1,y1,x2,y2]], "attrs":[{...}]}}.
    Поля строки: x1 y1 w h blur expression illumination invalid occlusion pose
    (invalid стоит ПЕРЕД occlusion/pose — порядок из readme.txt). xywh→xyxy.
    """
    lines = Path(txt_path).read_text().splitlines()
    entries, i = {}, 0
    while i < len(lines):
        name = lines[i].strip()
        i += 1
        if not name:
            continue
        n = int(lines[i].strip())
        i += 1
        boxes, attrs = [], []
        count = n if n > 0 else 1          # при n==0 идёт строка-заполнитель из 10 нулей
        for _ in range(count):
            parts = lines[i].split()
            i += 1
            if n == 0:
                continue
            x, y, w, h = (float(v) for v in parts[:4])
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            attrs.append({"blur": int(parts[4]), "invalid": int(parts[7]),
                          "occlusion": int(parts[8])})
        entries[name] = {"boxes": boxes, "attrs": attrs}
    return entries
```

- [ ] **Шаг 7.5: Добавить `eval_facedetect` в `evaluate.py`**

```python
FACEDETECT_AUTOMOTIVE = {"14--Traffic", "5--Car_Accident", "59--people--driving--car"}


def eval_facedetect(cfg: dict) -> dict:
    model_path = ROOT / cfg["model_path"]
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"FaceNet ONNX unavailable: {model_path} → UNDEF")
        return {"error": "model not found (FaceNet ONNX unavailable) → UNDEF"}

    gt_txt = data_dir / "wider_face_val_bbx_gt.txt"
    images_root = data_dir / "images"
    if not gt_txt.exists():
        return {"error": "dataset not found"}

    gt = parse_wider_gt(gt_txt)
    sess = get_ort_session(str(model_path))
    input_name = sess.get_inputs()[0].name
    H, W = 416, 736   # FaceNet вход 736×416 (W×H)

    predictions, ground_truths = [], []
    for rel, info in tqdm(gt.items(), desc="FaceDetect", unit="img"):
        img_path = images_root / rel
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        inp = cv2.resize(img, (W, H)).astype(np.float32)
        inp = inp[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0   # BGR→RGB NCHW
        outputs = sess.run(None, {input_name: inp})
        cov, bbox = outputs[0][0], outputs[1][0]
        boxes = detectnet_v2_decode(
            cov, bbox, target_cls=0, conf_thr=0.4, stride=16, bbox_norm=35.0,
            img_w=W, img_h=H, scale_w=orig_w / W, scale_h=orig_h / H,
        )
        predictions.append({"image_id": rel, "boxes": boxes})
        ground_truths.append({"image_id": rel, "boxes": info["boxes"]})

    if not predictions:
        return {"error": "no images"}

    m = compute_detection_metrics(predictions, ground_truths,
                                  iou_threshold=0.5, conf_threshold=0.4)

    def slice_metrics(cats):
        p = [x for x in predictions if x["image_id"].split("/")[0] in cats]
        g = [x for x in ground_truths if x["image_id"].split("/")[0] in cats]
        return compute_detection_metrics(p, g, iou_threshold=0.5,
                                         conf_threshold=0.4).to_dict() if p else {}

    auto = slice_metrics(FACEDETECT_AUTOMOTIVE)

    md = m.to_dict()
    md["automotive_slice"] = auto
    # Официального Easy/Med/Hard split нет локально → headline = overall AP@0.5.
    thresholds = {"map50": 0.50}   # консервативный headline-порог (Hard-уровень)
    status = check_thresholds(md, thresholds)
    status["easy_medium_hard"] = {
        "value": None, "threshold": "Easy≥0.80/Med≥0.70/Hard≥0.50",
        "status": "UNDEF", "note": "официальный WIDER eval_tools split недоступен локально",
    }

    result = {"metrics": md, "thresholds": status,
              "model": "FaceNet DetectNet_v2 (face)",
              "note": "AP@0.5 overall + automotive; Easy/Med/Hard UNDEF (нет официального split)"}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info(f"FaceDetect: P={m.precision:.3f} R={m.recall:.3f} AP@0.5={m.map50:.3f}")
    return result
```

- [ ] **Шаг 7.6: Зарегистрировать `facedetect` + защитить `check_all`**

В `EVAL_CONFIGS` (рядом с `color`):
```python
    "facedetect": {
        "model_path": "models/facedetect/facenet.onnx",   # перекрывается local_paths.yaml
        "data_dir": "data/wider_face",                      # перекрывается local_paths.yaml
        "results_dir": "results/facedetect",
        "eval_fn": eval_facedetect,
    },
```
В `check_all()` (строка ~1012) заменить безусловный доступ `cfg["model_path"]` на безопасный, чтобы конфиги без `model_path` (nomeroff_*) не роняли `--check`:
```python
        model_path = cfg.get("model_path")
        if model_path is None:
            log.info(f"{name}: model loaded internally (skip path check)")
            continue
        model_path = ROOT / model_path
```
(адаптировать под фактический код вокруг строки 1012; цель — `.get` вместо `[]`).

- [ ] **Шаг 7.7: Запустить тест — PASS**

Run: `cd /home/mk/CarCV-metrics && .venv/bin/python -m pytest deploy/tests/test_eval_facedetect.py -q`
Expected: PASS (1 passed).

- [ ] **Шаг 7.8: Получить FaceNet ONNX — `deploy/scripts/get_facenet_onnx.sh`**

```bash
#!/usr/bin/env bash
# Попытка получить deployable FaceNet ONNX. 3 шага; при неудаче — UNDEF.
set -u
OUT=/home/mk/CarCV-metrics/data/prep/facedetect
mkdir -p "$OUT"
ETLT=/home/mk/Загрузки/facenet_pruned_quantized_v2.0.1/model.etlt

# (1) Прямая загрузка deployable ONNX с NGC (как download_models.sh, без ключа):
URL="https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/deployable_v1.0/files/model.onnx"
echo "[1/3] NGC deployable ONNX…"
wget -q "$URL" -O "$OUT/facenet.onnx" && [ -s "$OUT/facenet.onnx" ] && { echo "OK NGC"; exit 0; }
rm -f "$OUT/facenet.onnx"

# (2) Экспорт etlt→onnx стандартными ключами (если установлен tao/tlt-конвертер):
echo "[2/3] etlt→onnx экспорт…"
for KEY in tlt_encode nvidia_tlt; do
  if command -v tao_converter >/dev/null 2>&1; then
    tao_converter -k "$KEY" -e "$OUT/facenet.onnx" "$ETLT" 2>/dev/null && [ -s "$OUT/facenet.onnx" ] && { echo "OK export ($KEY)"; exit 0; }
  fi
done

# (3) Не вышло — UNDEF (валидный окончательный результат).
echo "[3/3] FAILED — FaceNet ONNX недоступен → UNDEF"
exit 1
```
Run: `bash deploy/scripts/get_facenet_onnx.sh`
- Если ONNX получен → шаг 7.9 (прогон).
- Если нет → пропустить прогон; в `EXPERIMENT.md` вердикт **UNDEF** (без долгих раскопок TAO-тулинга).

- [ ] **Шаг 7.9: Прогон (если ONNX есть) или фиксация UNDEF**

```bash
.venv/bin/python deploy/scripts/run_local.py --models facedetect
```
Expected (есть ONNX): `FaceDetect: P=... R=... AP@0.5=...`, `results/facedetect/metrics.json` (overall + automotive_slice; Easy/Med/Hard UNDEF).
Expected (нет ONNX): `facedetect: ERROR/UNDEF → model not found (FaceNet ONNX unavailable) → UNDEF` — записать metrics.json вручную/через прогон с пометкой UNDEF в EXPERIMENT.md.

- [ ] **Шаг 7.10: Вердикт + коммит/merge**

```bash
git add deploy/evaluation/evaluate.py deploy/scripts/get_facenet_onnx.sh deploy/tests/test_eval_facedetect.py results/facedetect/
git commit -m "feat(eval): FaceDetect × WIDER FACE — eval_facedetect + парсер + получение ONNX (или UNDEF)"
git checkout main && git merge --no-ff exp/eval-facedetect -m "merge: FaceDetect × WIDER FACE"
```

---

# Task 8: Финальная агрегация на `main`

**Files:**
- Create/Update: `results/SUMMARY.md`, `notebooks/local_validation_campaign.ipynb`, обновление памяти кампании

- [ ] **Шаг 8.1: Сгенерировать `results/SUMMARY.md`**

```bash
cd /home/mk/CarCV-metrics && git checkout main
ls deploy/results deploy/plots                      # симлинки на месте (иначе шаг 0.3)
.venv/bin/python deploy/evaluation/evaluate.py --summary
```
Expected: `evaluate.py --summary` читает `results/<name>/metrics.json` всех ключей `EVAL_CONFIGS` (включая `color`, `facedetect`) и пишет `results/SUMMARY.md`. Проверить, что в нём 7 моделей с вердиктами PASS/FAIL/UNDEF/deferred.

- [ ] **Шаг 8.2: Воспроизводимый notebook — `notebooks/local_validation_campaign.ipynb`**

Создать ноутбук с ячейками (по правилам CLAUDE.md): (1) markdown-обзор кампании; (2) для каждой модели — команда `run_local.py --models <name>` и загрузка `results/<model>/metrics.json` в таблицу; (3) сводная таблица 7 пар с вердиктами; (4) вывод PNG из `plots/`. Smoke-проверка: `.venv/bin/jupyter nbconvert --to notebook --execute notebooks/local_validation_campaign.ipynb --output local_validation_campaign.ipynb` (или пометить ячейки прогона как «запускать вручную», т.к. полный прогон долог — тогда ноутбук читает уже готовые `metrics.json`).

- [ ] **Шаг 8.3: Обновить память кампании**

Обновить `/home/mk/.claude/projects/-home-mk-CarCV-metrics/memory/validation-campaign-status.md`: измеренные PASS/FAIL/UNDEF/deferred по 7 моделям (TrafficCamNet, VehicleTypeNet, VehicleMakeNet, Color, nomeroff_lpd, nomeroff_ocr, FaceDetect) + пороги. Добавить указатель в `MEMORY.md`, если меняется хук.

- [ ] **Шаг 8.4: Финальный коммит**

```bash
git add results/SUMMARY.md notebooks/local_validation_campaign.ipynb plots/
git commit -m "docs(results): финальная агрегация локальной кампании (7 пар) — SUMMARY + notebook"
```

---

## Self-Review (выполнено автором плана)

**Покрытие спеки:** все 7 пар — Tasks 1–7; локальный раннер + конфиг путей + .gitignore — Task 0; финальная агрегация (SUMMARY + notebook + память) — Task 8; ветка-на-эксперимент + merge в main — git-шаги каждой задачи; протокол «read docs → подготовка → прогон → фиксация» — шаги `.1`/`.6-.8` каждой модельной задачи; два новых эвалуатора (`eval_color`, `eval_facedetect`) с утверждёнными порогами — Tasks 4, 7; ветка VMMRdb — Task 3; конвертеры BDD/Stanford — Tasks 1, 2; риски (FaceNet UNDEF, WIDER split, nomeroff на py3.13 deferred, Stanford test-labels, Color caveats, git-гигиена) — отражены в read-docs/прогонных шагах и `.gitignore`. Вне охвата (SSH/qudata, тюнинг порогов, Face embedding, latency) — не планировалось.

**Скан заглушек:** конкретный код во всех шагах правки кода; команды с ожидаемым выводом; пути абсолютные и сверены по диску. Единственные намеренно-условные места — установка `nomeroff-net` (deferred при неудаче) и получение FaceNet ONNX (UNDEF при неудаче) — это утверждённые в спеке валидные исходы, а не недосказанность.

**Согласованность типов:** сигнатуры новых функций (`overlay_config`, `convert_bdd_to_labels`, `build_stanford_records`/`parse_class_names`/`parse_annos`, `discover_vmmrdb_samples`, `preprocess_color`/`load_madcars_color_index`/`COLOR_CLASSES`/`HEX_TO_CARS_COLOR`/`eval_color`, `parse_wider_gt`/`eval_facedetect`) совпадают между определением и использующими их тестами/эвалуаторами; форматы `predictions`/`ground_truths` соответствуют контракту `compute_*_metrics`; схема `metrics.json` (`metrics`+`thresholds`) совместима с `aggregate_summary.infer_family` (classification по `top1_accuracy`, detection по `precision/recall/f1`).
