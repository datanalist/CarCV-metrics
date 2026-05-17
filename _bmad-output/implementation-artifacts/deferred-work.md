# Deferred Work

Откладывается из текущей итерации (split из QQ-цикла `validation-campaign-2026-05-17`).

---

## Goal 2 — Campaign config + execution (deferred 2026-05-17)

**Зачем:** Запустить серию валидационных экспериментов на двух удалённых GPU-серверах qudata, опираясь на инфраструктуру из Goal 1 (`run_remote.py` + расширенная YAML-схема).

**Hosts:**
- qudata-1: `ssh4.qudata.ai:19356` (root)
- qudata-2: `ssh9.qudata.ai:19478` (root)

**Scope:**
- Карта «модель × датасет × сервер» (распределение всех 5 моделей `trafficcamnet`, `vehiclemakenet`, `vehicletypenet`, `lpdnet`, `lprnet` по двум qudata-хостам).
- Опора на `_bmad-output/planning-artifacts/research-datasets-validation.md` §6.5 (шорт-лист):
  - `trafficcamnet` ← BDD100K val (10K) + COCO val 2017 (5K)
  - `vehiclemakenet` ← Stanford Cars test (8,041) + MAD-Cars sampled (~4K)
  - `vehicletypenet` ← Stanford Cars + Make→Type mapping
  - `lpdnet` ← Kaggle `c/car-plates-recognition` или собственная разметка
  - `lprnet` ← AY000554 HF (val+test 7,736)
- Скачивание моделей (NVIDIA TAO/NGC), подготовка датасетов на серверах (`deploy/scripts/download_models.sh`, `download_datasets_*.sh` уже существуют — могут потребовать апдейта).
- Реальный прогон `deploy → run → collect` против двух qudata.

**Зависит от:** Goal 1 (orchestrator + dry-run mode + YAML-схема с host/port/user).

---

## Goal 3 — Aggregation: SUMMARY.md generator + post-processing notebook (deferred 2026-05-17)

**Зачем:** Автоматизировать сборку `results/SUMMARY.md` (правило из `CLAUDE.md`) и предоставить notebook для пост-обработки.

**Scope:**
- Скрипт-агрегатор `results_collected/{host}/results/*.json` → единый `results/SUMMARY.md` с per-модельными метриками (Precision/Recall/F1/mAP@0.5 для detection; Top-1/Top-3/Accuracy для classification; Full Plate Accuracy/Character Accuracy для LPR).
- Сегментация метрик по `weather × timeofday × scene` для BDD100K (метаданные доступны).
- `notebooks/post_processing.ipynb` — воспроизводимый ноутбук для plots в `plots/` (PNG): confusion matrices, PR-curves, error breakdown.

**Зависит от:** Goal 2 (наличие реальных результатов в `results_collected/`).

---

## Источник решения

QQ-цикл `validation-campaign-2026-05-17`, multi-goal check на step-01 → пользователь выбрал [S] Split. Текущая итерация сужена до Goal 1.
