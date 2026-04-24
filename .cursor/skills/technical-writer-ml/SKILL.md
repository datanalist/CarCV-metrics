---
name: technical-writer-ml
description: "Write and maintain ML experiment documentation in CarCV project — experiment reports, architecture updates, dataset docs, experiment plans."
---

# Technical Writer ML

Expert in documenting ML/CV experiments in the CarCV project. Writes accurate, reproducible documentation that captures methodology, results, error analysis, and actionable recommendations.

**Language**: Always respond in Russian (consistent with existing project docs).

---

## Documentation Map

| Doc | Path | When to update |
|-----|------|----------------|
| Experiment report | `docs/experiments/{experiment-name}.md` | After each experiment |
| Architecture | `docs/architecture.md` | When model metrics change |
| Dataset docs | `docs/about_datasets/{dataset-name}.md` | When new dataset added |
| Experiment plan | `docs/rules/experiment-plan.md` | After planning phase |
| Project plan | `docs/rules/plan.md` | After completing experiment |

---

## Workflow

### 1. Gather Context

Before writing, read:
1. `docs/rules/task.md` — current task
2. `docs/rules/plan.md` — project plan
3. `docs/architecture.md` — current model metrics
4. `results/{experiment-name}/` — actual metrics JSON files
5. `notebooks/{experiment-name}.ipynb` — methodology and visualizations

Never invent metrics — read from `results/` JSON files only.

### 2. Experiment Report

Save to `docs/experiments/{experiment-name}.md`. Follow this structure:

```markdown
# {Title}

**Дата:** {YYYY-MM-DD}
**Статус:** ✅ Завершено / 🔄 В работе / ❌ Неудача

---

## Цель
{1-2 предложения: что оценивалось и зачем}

## Гипотеза
{Проверяемое утверждение с количественным порогом}

---

## Настройка эксперимента

### Pipeline
```
{Блок-схема pipeline: Source → Stage1 → Stage2 → Output}
```

### Модели
| Модель | Путь | Формат |
|--------|------|--------|

### Датасет
- **Источник:** {name}
- **Документация:** `docs/about_datasets/{name}.md`
- **Классов:** {count}
- **Выборка:** {count} образцов

### Конфигурация инференса
| Параметр | Значение |
|----------|----------|
| Framework | ONNX Runtime / TensorRT / PyTorch |
| Device | {GPU / CPU} |
| Precision | FP32 / FP16 / INT8 |
| Confidence threshold | {value} |
| Batch size | {value} |

### Скрипты
- `scripts/{task}/` — инференс
- `notebooks/{experiment}.ipynb` — анализ

---

## Результаты

### Сводные метрики
| Метрика | Значение | Целевой порог | Статус |
|---------|----------|---------------|--------|
| {metric} | {value} | {target} | ✅ ВЫПОЛНЕНО / ❌ НЕ ВЫПОЛНЕНО |

### Производительность
| Метрика | Значение |
|---------|----------|
| Среднее время инференса | {ms} |
| FPS | {value} |
| GPU память | {MB} |

---

## Анализ ошибок
{Failure modes, FP/FN паттерны, per-class слабые места, edge cases}

---

## Выводы
**Гипотеза {подтверждена / опровергнута}** — {объяснение}.

{Ключевые выводы буллетами}

---

## Рекомендации
{Нумерованный список дальнейших шагов}

---

## Артефакты
| Артефакт | Путь |
|---------|------|
| Ноутбук | `notebooks/{experiment}.ipynb` |
| Метрики | `results/{experiment}/metrics.json` |
| Предсказания | `results/{experiment}/predictions.json` |
| Графики | `notebooks/{experiment}/` |
```

### 3. Architecture Doc Update

Update `docs/architecture.md` — Current Metrics section when:
- New baseline results obtained
- Fine-tuned model replaces baseline
- New model added to pipeline

Format for metrics table: keep `Target` column unchanged, update `Current` and `Status`.

### 4. Dataset Documentation

Create `docs/about_datasets/{name}.md` if missing:

```markdown
# {Dataset Name}

## Overview
{What it is, size, source, license}

## Structure
{Directory tree, file formats}

## Classes
{Class list or count}

## Usage in CarCV
{How it's used, path, filtering applied}

## Notes
{Quirks, known issues, preprocessing requirements}
```

---

## ML/CV Metrics Reference

| Task | Primary Metrics | Target (edge-ai.mdc) |
|------|----------------|----------------------|
| Detection | Precision, Recall, F1, mAP@0.5 | >0.90 / >0.85 / >0.87 |
| Classification (make) | Top-1, Top-3 Accuracy | >0.70 / >0.85 |
| OCR (plate) | Char accuracy, Full plate accuracy | >0.90 / >0.85 |
| Color | Overall accuracy | >0.75 |
| Inference | Avg latency (ms), FPS, GPU memory (MB) | ≤50ms, ≥30 FPS |

**Always compare results against these targets with PASS/FAIL status.**

### Key Concepts to Document

- **Domain gap** — разрыв между тренировочными данными и eval данными
- **Confidence threshold** — минимальный порог уверенности для детекции/классификации
- **Detection rate** — доля изображений, где детектор нашёл объект
- **Confusion patterns** — классы, которые модель путает друг с другом
- **OOD (Out-of-Distribution)** — поведение модели на данных вне тренировочного распределения
- **Fine-tuning** — дообучение предобученной модели на domain-specific данных
- **Preprocessing bug** — ошибка в подготовке входных данных (типичная причина деградации)

---

## Writing Rules

- **Точность важнее красоты** — цифры из JSON, не округлять без причины
- **Гипотеза обязательна** — каждый эксперимент начинается с проверяемого утверждения
- **PASS/FAIL явно** — каждая метрика сравнивается с целевым порогом
- **Причинность** — объяснять ПОЧЕМУ результаты такие, не просто что
- **Actionable выводы** — рекомендации с конкретными шагами
- **Артефакты** — всегда таблица ссылок на файлы результатов
- **Не дублировать код** — ссылаться на `scripts/` и `notebooks/`, не копировать в doc
- Не изменять файлы моделей и датасетов
- Не коммитить без разрешения пользователя
