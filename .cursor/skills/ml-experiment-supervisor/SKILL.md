---
name: ml-experiment-supervisor
description: "Audit ML experiments for purity, industry compliance, contradictory results, and reproducibility. Use proactively when reviewing experiment reports, before merging experiment PRs, or when verifying experiment quality."
---

# ML Experiment Supervisor

Аудит ML-экспериментов на соответствие индустриальным практикам, отсутствие противоречий и воспроизводимость. Основан на REFORMS (Princeton), AAAI Reproducibility Checklist, LakeFS Three Pillars, и рекомендациях Google/Microsoft.

## Workflow Overview

1. **Собрать артефакты** — notebook, report, scripts, configs, results
2. **Проверить три столпа воспроизводимости** — Data, Code/Params, Environment
3. **Пройти чеклисты** — Study design, Computational reproducibility, Data quality, Modeling, Leakage, Metrics, Artifacts
4. **Поиск противоречий** — report vs architecture vs notebook vs results
5. **Отчёт** — issues (critical / warning / info), рекомендации

---

## Phase 1: Gather Artifacts

Для эксперимента `{experiment-name}` собери:

| Артефакт | Путь | Проверка |
|----------|------|----------|
| Report | `docs/experiments/{experiment-name}.md` | Структура, полнота |
| Notebook | `notebooks/{experiment-name}.ipynb` | Воспроизводимость кода |
| Config | `configs/experiment/{experiment-name}.yaml` | Hydra (если применимо) |
| Results | `results/{...}/` | metrics.json, predictions.pkl |
| Scripts | `scripts/{task-name}/` | Переиспользуемый код |
| Plan | `docs/rules/plan.md` | Соответствие задаче |

Читай: `dl-experiments.mdc`, `report-template.md`, `architecture.md`, `edge-ai.mdc` (целевые пороги).

---

## Phase 2: Three Pillars of Reproducibility (LakeFS)

Любое изменение в одном из трёх — меняет результаты. Проверить стабильность:

| Столп | Чек-пойнты |
|-------|-------------|
| **Input Data** | Dataset path, split (train/val/test), фильтры, версионирование. Нет ли данных из test в train? |
| **Code & Params** | Hyperparameters в конфиге, seeds зафиксированы, версия кода (git hash). Все параметры задокументированы? |
| **Execution Environment** | Python/uv, библиотеки (versions), GPU/CPU, precision (FP16/FP32). Есть requirements или lockfile? |

---

## Phase 3: REFORMS / Industry Checklist

### Study Design
- [ ] Чётко указаны цель и гипотеза
- [ ] Описаны все допущения (assumptions)
- [ ] Мнения/спекуляции отделены от фактов и результатов

### Computational Reproducibility
- [ ] Random seeds зафиксированы (random, numpy, torch)
- [ ] Код доступен, команды для воспроизведения в отчёте
- [ ] Вычислительная среда описана (hardware, OS, libs)

### Data Quality
- [ ] Описание датасета (источник, split, кол-во примеров)
- [ ] Preprocessing задокументирован
- [ ] Нет data leakage: нормализация/аугментация только на train

### Modeling
- [ ] Финальные hyperparameters перечислены
- [ ] Выбор метрик обоснован
- [ ] Модель и путь указаны однозначно

### Data Leakage (критично)
- [ ] Test set не использован при preprocessing/tuning
- [ ] Нет совместной нормализации train+test
- [ ] Оversampling/sampling — только после split

### Metrics & Uncertainty
- [ ] ≥3 метрик
- [ ] Есть меры вариации (std, CI, error bars), если runs > 1
- [ ] Статистические тесты при сравнении моделей (если применимо)
- [ ] Сравнение с целевыми порогами (edge-ai.mdc)

### Generalizability
- [ ] Ограничения и границы применимости упомянуты
- [ ] Edge cases (ночь, дождь, blur) — отражены в error analysis

---

## Phase 4: Contradiction Detection

Проверить согласованность между источниками.

### Report vs Notebook
- Метрики в report совпадают с вычисленными в notebook?
- Пути к данным и моделям совпадают?
- Даты и статус (Completed/In Progress) корректны?

### Report vs architecture.md
- Current Metrics в architecture — совпадают с последним экспериментом по этой модели?
- Если не совпадают — обновлён ли architecture или report устарел?

### Report vs results/
- `metrics.json` — значения совпадают с report?
- Файлы predictions/figures существуют по указанным путям?

### Plan vs Report
- Цель в plan соответствует результату в report?
- Experiment отмечен как completed в plan — если report Completed?

### Типичные противоречия
- Разные значения одной метрики в разных местах
- Указание пути к модели/датасету, которого нет
- Report говорит "PASS", но architecture показывает старые (хуже) метрики
- Разные числа классов/samples в report и в notebook

---

## Phase 5: Artifact Clarity & Reproducibility Blockers

### Что мешает наглядности
- Отсутствующие визуализации (confusion matrix, per-class, error examples) — по dl-experiments.mdc обязательны
- Метрики без контекста (нет target, нет сравнения)
- Выводы без ссылок на конкретные ячейки/графики

### Что мешает воспроизводимости
- Жёстко прописанные пути в коде вместо config
- Отсутствие seeds или нестабильный порядок данных
- Нет команды `uv run python ...` или `jupyter` в разделе "How to Reproduce"
- Временный код не удалён, но dependencies не задокументированы

### Проектные конвенции (CarCV)
- Hydra для конфигов — все гиперпараметры в yaml
- Артефакты: metrics.json, predictions.pkl, figures в notebooks/{name}/
- Report по шаблону report-template.md

---

## Phase 6: Output Report

Формат отчёта аудита:

```markdown
# Audit: {experiment-name}
**Date:** {YYYY-MM-DD}

## Summary
{1-2 предложения: PASS / ISSUES FOUND}

## Critical Issues (block reproducibility or validity)
- [ ] Issue 1
- [ ] Issue 2

## Warnings (should fix)
- [ ] Warning 1

## Info (recommendations)
- [ ] Suggestion 1

## Contradictions Found
| Location A | Location B | Conflict |
|------------|------------|----------|
| report: mAP=0.85 | notebook cell: mAP=0.82 | Different values |

## Checklist Score
{ X / Y items passed }

## Recommendations
{ Prioritized list of actions }
```

---

## Rules

- Не модифицировать артефакты — только читать и отчёт писать
- Язык отчёта — как у пользователя
- При неясности — указать в отчёте как "Requires clarification"
- Ссылаться на конкретные файлы и строки при issues
- Использовать пороги из edge-ai.mdc для проверки PASS/FAIL консистентности
