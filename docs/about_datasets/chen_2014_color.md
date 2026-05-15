# Vehicle Color Recognition Dataset (Chen 2014)

## Общая информация

**Vehicle Color Recognition Dataset** — корпус для классификации цвета автомобилей, опубликован P. Chen, X. Bai, W. Liu в работе «Vehicle Color Recognition on Urban Road by Feature Context» (IEEE TITS, 2014). Содержит 15,601 кадров frontal-view, снятых HD-камерой на городских дорогах в условиях освещённости, дымки и пересвета.

**Локальное расположение:** _(скачивать при необходимости)_ — см. раздел «Получение».

**Оригинальная страница проекта:** `http://122.205.5.5:8071/~pchen/project.html` (IP-based URL, на 2026-05-15 нестабилен; см. зеркала).

---

## Структура датасета

```
chen_color/
├── black/                  # ~1,950 изображений
├── blue/                   # ~1,950
├── cyan/                   # ~1,950
├── gray/                   # ~1,950
├── green/                  # ~1,950
├── red/                    # ~1,950
├── white/                  # ~1,950
└── yellow/                 # ~1,950
```

Структура — одна папка на класс цвета (folder-per-class layout), типично для классификационных датасетов. Точное распределение по классам — после распаковки.

> **Примечание:** оригинальные имена файлов могут содержать non-English символы — рекомендуется переименовать при загрузке.

---

## Статистика

| Параметр | Значение |
|----------|----------|
| Всего изображений | 15,601 |
| Классов цвета | 8 |
| Разрешение | 1920×1080 (frontal view) |
| Камера | HD camera |
| Сцена | городская дорога |
| Сложности | illumination variation, haze, overexposure |

---

## Классы цветов

| # | Класс | Совпадение с CARS §6.5 (15) |
|---|-------|------------------------------|
| 1 | black | ✅ |
| 2 | blue | ✅ |
| 3 | cyan | ❌ (нет в CARS) |
| 4 | gray | ✅ (CARS: `grey`) |
| 5 | green | ✅ |
| 6 | red | ✅ |
| 7 | white | ✅ |
| 8 | yellow | ✅ |

**Покрытие: 7/15 цветов CARS.**

**Отсутствуют в Chen 2014:** `beige`, `brown`, `gold`, `orange`, `pink`, `purple`, `silver`, `tan` — 8 цветов.

---

## Формат изображений

- **Разрешение:** 1920×1080 (соответствует §5.1 CARS!).
- **Цветовое пространство:** RGB JPEG (предположительно).
- **Содержание:** frontal view автомобиля на городской дороге (camera-fixed, urban traffic).

⚠️ Это полные кадры, **не кропы** автомобилей. Для подачи в `bae_model_f3.onnx` (вход 384×384) требуется предварительная детекция автомобиля (TrafficCamNet) и crop.

---

## Формат аннотаций

Folder-per-class. Метка цвета извлекается из имени родительской папки:

```python
from pathlib import Path

def get_label(image_path: str) -> str:
    return Path(image_path).parent.name  # 'black', 'blue', ...
```

Нет bbox-аннотаций.

---

## Использование для CarCV

### Применимость

- ✅ **Color Recognition baseline** для 7 «лёгких» цветов из 15 CARS — `black`, `blue`, `green`, `grey`, `red`, `white`, `yellow`.
- ✅ Разрешение 1920×1080 точно соответствует §5.1 CARS.
- ✅ Urban road conditions, illumination variation, haze — реалистичные шумы, близкие к ожидаемой эксплуатации CARS.
- ✅ Нет лика с CARS color-моделью `bae_model_f3.onnx` (модель — кастомная ResNet, не из Chen 2014 training set).

### Ограничения

- ❌ **Покрытие 7/15 цветов CARS.** 8 цветов (`beige`, `brown`, `gold`, `orange`, `pink`, `purple`, `silver`, `tan`) отсутствуют → для полной валидации §6.5 требуется дополнительный корпус.
- ❌ **Класс `cyan` отсутствует в CARS** — при использовании Chen 2014 для cross-evaluation нужно отбрасывать cyan-кадры.
- ❌ **License не указана явно.** Использование для internal benchmark — стандартная академическая практика; публикация метрик во внешних материалах — flag для legal review.
- ❌ Полные кадры (не crops) — требуется предварительная детекция и crop через TrafficCamNet pipeline. Это вносит зависимость метрики цвета от точности detection.
- ❌ Оригинальный URL `122.205.5.5:8071` нестабилен → использовать Google Drive зеркало (haze-free version).

### Рекомендации для validation §6.5

1. Использовать Chen 2014 для **7 цветов**: `black, blue, green, grey, red, white, yellow`.
2. **Маппинг:** `cars_color = chen_color` для совпадающих, отбрасывать `cyan`.
3. **Crop pipeline:**
   - Применить TrafficCamNet (или ground-truth bbox если доступно) → crop автомобиля.
   - Resize crop → 384×384 (нормализация ImageNet mean=[0.43, 0.40, 0.39], std=[0.27, 0.26, 0.26] из §6.5).
   - Inference `bae_model_f3.onnx` → predicted color.
4. **Метрики по цветам:**
   - Best classes (`black`, `white`, `red`, `blue`) — Acc >0.90 (per §6.5).
   - Прочие 3 (`green`, `grey`, `yellow`) — Acc >0.80 (overall threshold).
5. **Для 8 отсутствующих цветов** (`beige`, `brown`, `gold`, `orange`, `pink`, `purple`, `silver`, `tan`):
   - Собрать собственную разметку из CARS pilot-кадров.
   - Целевой объём: **100 примеров на цвет = 800 кадров** (CP3-расширение).
   - Сохранять в той же folder-per-class структуре для единого pipeline.

### Альтернатива / дополнение

- **VeRi-776** имеет атрибуты цвета (~10 colors, включая `brown`, `silver`, `orange`) — может частично закрыть gap, **но non-commercial лицензия**.
- **VehicleID** — color attrs, но license unclear.
- **Synthetic data** через color augmentation базы (HSV/LAB shift) — для challenging colors как aug, не как ground-truth.

---

## Получение

**Оригинальный URL** (на 2026-05-15 нестабилен):
- `http://122.205.5.5:8071/~pchen/project.html` (IP-based, академический хостинг)

**Haze-free processed зеркало** (Google Drive):
- `https://drive.google.com/open?id=1DKYmI1R-3yOYXaE0VYWxypppOUOxnBGx`

**Action:** уточнить актуальный download URL у авторов (`pchen@...` контакт в оригинальной статье) или искать новые зеркала через PapersWithCode.

---

## Лицензия

Лицензия в публикациях Chen et al. 2014 явно не указана. Стандартная академическая практика — research use по согласованию с авторами. Для CARS:
- ✅ Internal benchmark — допустимо.
- ⚠️ Публикация метрик в коммерческом продукте — flag для legal review (см. §5.1 / §7.2 research-документа).

---

## Ссылки

- [Chen P., Bai X., Liu W. — «Vehicle Color Recognition on Urban Road by Feature Context» (IEEE TITS, 2014)](https://www.researchgate.net/publication/266082848_Vehicle_Color_Recognition_on_Urban_Road_by_Feature_Context)
- [Rachmadi, Purnama — «Vehicle Color Recognition using CNN» (arXiv 1510.07391)](https://arxiv.org/pdf/1510.07391) — использует данный датасет.
- [GitHub: Vehicle-Color-Identification (jwhabi)](https://github.com/jwhabi/Vehicle-Color-Identification)

---

## История изменений

- **2026-05-15** — Создана документация в рамках research-датасетов для валидации ML-стека CARS (см. `_bmad-output/planning-artifacts/research-datasets-validation.md`).
