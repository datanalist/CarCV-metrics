# AUTO.RIA Numberplate OCR RU Dataset — `autoriaNumberplateOcrRu-2021-09-01`

## Общая информация

**AUTO.RIA Numberplate OCR RU Dataset** — корпус нормализованных кропов российских автомобильных номерных знаков (ГОСТ-Р 50577) с OCR-аннотациями в формате проекта **Nomeroff Net** (стиль Supervisely/JSON). Это «сырой» обучающий датасет проекта Nomeroff Net / AUTO.RIA.com, на котором обучался кастомный OCR-модуль фреймворка (исторически — RNN-архитектура). Данные предоставлены сообществом **Habr.com** (особая благодарность — Aziz Temirkhanov, German Cyganov, согласно `license.txt`). Версия датасета — снимок от **2021-09-01** (архив опубликован 01.09.2021).

Это **первоисточник** относительно очищенного HuggingFace-производного `AY000554/Car_plate_OCR_dataset` (см. соседний файл [`car_plate_ocr_dataset_ru.md`](car_plate_ocr_dataset_ru.md)). Различия описаны ниже в разделе «Использование для CarCV».

**Локальное расположение:** `/home/mk/Загрузки/DATASETS/nomeroff/autoriaNumberplateOcrRu-2021-09-01`

**Прямая ссылка на архив:** https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip (1 499 299 647 байт ≈ 1.5 GB)

**Каталог датасетов Nomeroff Net:** https://nomeroff.net.ua/datasets/

**Исходный проект:** https://nomeroff.net.ua/ · GitHub: https://github.com/ria-com/nomeroff-net

---

## Структура датасета

```
/home/mk/Загрузки/DATASETS/nomeroff/autoriaNumberplateOcrRu-2021-09-01/
├── license.txt                 # CC BY 4.0 (дословный текст)
├── train/
│   ├── ann/                    # 49 382 JSON (по одному на кроп)
│   └── img/                    # 49 382 PNG (кропы пластины)
├── val/
│   ├── ann/                    #  4 893 JSON
│   └── img/                    #  4 893 PNG
└── test/
    ├── ann/                    #  2 845 JSON
    └── img/                    #  2 845 PNG
```

Каждый split содержит ДВЕ подпапки: `ann/` (аннотации) и `img/` (изображения). Имена файлов в `img/` и `ann/` соответствуют попарно (один stem — одна аннотация на один кроп). Stem может быть как таймстампом (`11_11_2014_10_42_11_230_0`), так и текстом номера (`B828EE69`). **Ground-truth текст всегда берётся из поля `description` внутри JSON, а НЕ из имени файла и НЕ из поля `name`** (см. предостережение в разделе «Формат аннотаций»).

---

## Статистика

| Компонент | Количество | Размер (прибл.) | Доля |
|-----------|------------|-----------------|------|
| Train (img) | 49 382 PNG | ~1.3 GB | 86.5% |
| Train (ann) | 49 382 JSON | ~195 MB | — |
| Val (img) | 4 893 PNG | ~139 MB | 8.6% |
| Val (ann) | 4 893 JSON | ~20 MB | — |
| Test (img) | 2 845 PNG | ~76 MB | 5.0% |
| Test (ann) | 2 845 JSON | ~12 MB | — |
| **Всего кропов** | **57 120** | **~1.8 GB** (распакованный; архив ~1.5 GB) | 100% |

Размер кропа (поле `size`) сильно варьируется: ширина **59–882 px**, высота **13–191 px**; медиана ≈ **229×49 px**, среднее ≈ 241×52 px (диапазон p5–p95: ширина 122–403, высота 26–87). Подавляющее большинство кропов — RU-пластины с `state_id=2`, `region_id=6`, `count_lines=0` (однострочные стандартные знаки): `state_id=2` встречается в 57 118 из 57 120 файлов, `region_id=6` — в 57 119.

---

## Формат изображений

- **Тип контента:** crop номерного знака (не полный кадр), нормализованный/выровненный выходом детектора-источника.
- **Формат:** PNG (все 57 120 файлов — `.png`).
- **Разрешение:** переменное, медиана ≈ 229×49 px (см. «Статистика»); точный размер каждого кропа продублирован в поле `size` соответствующей аннотации.
- **Цветовое пространство:** в большинстве кропов **RGBA** (4 канала); часть файлов (преимущественно с именами-номерами) — RGB. При препроцессинге привести к 3-канальному RGB (отбросить альфа-канал).
- **Именование:** stem файла = таймстамп (`11_11_2014_10_42_11_230_0.png`) либо текст номера (`B828EE69.png`); ground-truth не извлекается из имени.

---

## Формат аннотаций

Один JSON на кроп (стиль Supervisely/Nomeroff). Bbox-разметки нет — это OCR-only корпус.

### Пример (типовой, файл-таймстамп `train/ann/11_11_2014_10_42_11_230_0.json`)

```json
{
    "tags": [],
    "objects": [],
    "state_id": "2",
    "region_id": "6",
    "size": {"width": 172, "height": 37},
    "moderation": {"isModerated": 1},
    "description": "A088KK60",
    "name": "11_11_2014_10_42_11_230_0",
    "count_lines": "0"
}
```

В типовом (большинство) файле `name` равно stem-у файла (таймстамп), а `region_id`/`state_id`/`count_lines` хранятся как **строки**. Есть и второй стиль (файлы с именами-номерами), где `name` совпадает с `description`, а числовые поля хранятся как `int`:

```json
{
    "tags": [],
    "objects": [],
    "description": "B828EE69",
    "name": "B828EE69",
    "region_id": 6,
    "state_id": 2,
    "count_lines": 0,
    "size": {"width": 188, "height": 41},
    "moderation": {"isModerated": 1}
}
```

Поле `name` совпадает с `description` лишь примерно в 80% файлов (train: 39 320 из 49 382); в остальных `name` = имя файла. **Опираться на `name` как на эталон нельзя.**

Часть аннотаций (≈4411 в train, есть и в val/test) дополнительно содержит поле `predicted` — предсказание исходной модели на этапе модерации/самообучения (НЕ ground-truth):

```json
{
    "predicted": "A029HKB5",
    "description": "A029PK35",
    "tags": [],
    "name": "A029PK35_0",
    "objects": [],
    "size": {"width": 148, "height": 32},
    "region_id": "6",
    "state_id": "2",
    "moderation": {"isModerated": 1}
}
```

### Поля

| Поле | Тип | Семантика |
|------|-----|-----------|
| `description` | string | **Ground-truth текст номера** (напр. `"A088KK60"`). Использовать как единственный эталон OCR. |
| `name` | string | Идентификатор кропа: чаще = stem файла (таймстамп), иногда = `description`. **НЕ использовать как эталон.** |
| `region_id` | int \| string \| null | Код типа/региона пластины в нотации Nomeroff (для RU доминирует `6`). Встречается как `int`, `"str"`, редко отсутствует. |
| `state_id` | int \| string \| null | Код государства в нотации Nomeroff (RU = `2`). Встречается как `int`, `"str"`, редко отсутствует. |
| `count_lines` | int \| string \| null | Число строк номера. Фактические значения в этом датасете: `0` (51 900 — однострочные) и единичный `1`; ~5219 файлов поле не содержат (`null`). Значение `2` (двухстрочные) в данном корпусе практически отсутствует. |
| `size` | object | `{width, height}` — размер кропа в пикселях. |
| `moderation` | object | `{isModerated}` — флаг прохождения модерации. |
| `predicted` | string (опц.) | Предсказание исходной модели (присутствует не во всех файлах). Не использовать как эталон. |
| `tags` | array | Зарезервировано (в RU-датасете пусто). |
| `objects` | array | Зарезервировано (bbox-объекты; в RU-датасете пусто). |

**Внимание:** типы `region_id`/`state_id`/`count_lines` не унифицированы между файлами (int / str / отсутствие) — при парсинге приводить к единому типу и обрабатывать `None`.

### Алфавит (ГОСТ-Р 50577)

```python
ALPHABET = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','E','K','M','H','O','P','C','T','Y','X']
```

22 содержательных символа: 10 цифр + 12 латинских литер, визуально идентичных кириллическим А/В/Е/К/М/Н/О/Р/С/Т/У/Х (стандарт ГОСТ-Р 50577 для гражданских регистрационных знаков РФ). Совпадает с целевым алфавитом OCR-модели CARS `nomeroff_ocr`.

---

## Использование для CarCV

Целевая модель: **`nomeroff_ocr`** (resnet18, чекпойнт `anpr_ocr_ru_2023_02_01_resnet18.ckpt`, RU OCR). Цели §6.4: **Full Plate Accuracy > 0.85**, **Character Accuracy > 0.92**. Прошлый прогон именно на этом датасете дал **PASS** (Character Accuracy = 0.9995).

### Применимость

- ✅ **§6.4 — Character Accuracy (> 0.92)** — валидация на `val + test` = 7 738 кропов; уже подтверждён PASS = 0.9995.
- ✅ **§6.4 — Full Plate Accuracy (> 0.85)** — эталон в `description`, точное сравнение строк.
- ✅ Алфавит датасета полностью соответствует ГОСТ-Р 50577 и алфавиту `nomeroff_ocr` (22/22 содержательных символа).
- ✅ Кропы (медиана ~229×49) сопоставимы по геометрии со входом модели LPR (~188×48) — ресайз без существенного искажения пропорций.
- ✅ Поля `region_id` / `state_id` / `count_lines` позволяют **сегментировать ошибки** по типу/региону/числу строк номера.
- ✅ Лицензия **CC BY 4.0** безопасна для коммерческой валидации и публикации метрик.

### Ограничения

- ❌ **Нет bbox/keypoints-аннотаций** → не валидирует ступень LP Detection (`nomeroff_lpd`) и не покрывает recall детектора пластин.
- ❌ Только кропы, нет полных кадров 1920×1080 → нет end-to-end проверки сценариев §5.1 (day/night/IR, 0–60 км/ч, 5–50 м, 0–30°).
- ⚠️ **Риск пересечения с обучающим корпусом** `nomeroff_ocr`: датасет — первоисточник обучения Nomeroff Net OCR. Сверхвысокий результат (0.9995) может отражать train-leakage; train (49 382) для валидации НЕ использовать.
- ⚠️ Неоднородность аннотаций: `region_id`/`state_id`/`count_lines` встречаются как int, как str и отсутствуют (`null`); `name` лишь в ~80% файлов совпадает с `description`.
- ⚠️ Поле `predicted` — это предсказание исходной модели, НЕ ground-truth; игнорировать при расчёте метрик.
- ⚠️ Изображения преимущественно RGBA (4 канала) — требуется конвертация в RGB перед подачей в модель.
- ⚠️ Снимок 2021 года: новейшие форматы/спецсерии номеров могут быть недопредставлены; `count_lines` фактически только однострочные (`0`), двухстрочные пластины не покрыты.

### Различие с `car_plate_ocr_dataset_ru.md`

В `datasets.md` для задачи OCR выбран **именно этот** сырой датасет AUTO.RIA/Nomeroff. Он отличается от описанного в [`car_plate_ocr_dataset_ru.md`](car_plate_ocr_dataset_ru.md) HuggingFace-производного `AY000554/Car_plate_OCR_dataset`:

| Признак | autoriaNumberplateOcrRu (этот) | AY000554/Car_plate_OCR_dataset |
|---------|--------------------------------|--------------------------------|
| Происхождение | Сырой исходник Nomeroff Net / AUTO.RIA | Очищенный производный (HuggingFace) |
| Кропов всего | 57 120 | 45 514 |
| Структура | `ann/` (JSON) + `img/` (PNG) | плоские `.jpg` по split |
| Ground-truth | поле `description` в JSON | имя файла (stem) |
| Метаданные | `region_id`, `state_id`, `count_lines`, `size` | отсутствуют |
| Формат изображений | PNG (часто RGBA) | JPEG |

### Рекомендации

1. **Бенчмарк** Full Plate Accuracy + Character Accuracy для `nomeroff_ocr` считать на `val + test` (7 738 кропов); `train` исключить из-за риска train-leakage.
2. Эталон брать ТОЛЬКО из поля `description` (НЕ из имени файла, НЕ из `name`, НЕ из `predicted`).
3. Препроцессинг: конвертировать RGBA→RGB, затем resize кропа → вход `nomeroff_ocr` ~188×48 (сохранять aspect ratio, padding), нормализация по конвейеру модели.
4. Расчёт метрик относительно порогов §6.4 (`Full Plate Accuracy > 0.85`, `Character Accuracy > 0.92`):
   ```python
   gt = ann["description"]
   pred = nomeroff_ocr(crop)
   full_plate_match = (pred == gt)
   char_acc = sum(p == g for p, g in zip(pred, gt)) / max(len(gt), 1)
   ```
5. **Сегментировать ошибки** по `region_id` / `state_id` / `count_lines`, чтобы отделить редкие типы номеров от стандартных однострочных (`count_lines=0`).
6. При парсинге аннотаций приводить `region_id`/`state_id`/`count_lines` к единому типу (int) и корректно обрабатывать `null`/отсутствие поля.
7. Сверхвысокий PASS (0.9995) трактовать осторожно: при подозрении на пересечение с train-корпусом дополнить независимым held-out набором (например, `AY000554` test или собственные кадры CARS).

---

## Получение

```bash
# Прямая загрузка архива (~1.5 GB; ~1.8 GB после распаковки)
wget https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip
unzip autoriaNumberplateOcrRu-2021-09-01.zip -d ./data/

# Если ссылка устарела — актуальная версия в каталоге датасетов:
#   https://nomeroff.net.ua/datasets/
```

```python
import json, glob, os
from PIL import Image

ROOT = "/home/mk/Загрузки/DATASETS/nomeroff/autoriaNumberplateOcrRu-2021-09-01"

def load_split(split):
    """Возвращает список (path_to_png, ground_truth_text)."""
    items = []
    for ann_path in glob.glob(os.path.join(ROOT, split, "ann", "*.json")):
        ann = json.load(open(ann_path, encoding="utf-8"))
        stem = os.path.splitext(os.path.basename(ann_path))[0]
        img_path = os.path.join(ROOT, split, "img", stem + ".png")
        items.append((img_path, ann["description"]))  # эталон — description
    return items

def load_rgb(path):
    """Кропы часто RGBA — приводим к RGB перед инференсом."""
    return Image.open(path).convert("RGB")

val = load_split("val")
test = load_split("test")
print(len(val), len(test))  # 4893 2845
```

---

## Лицензия

**CC BY 4.0** — Creative Commons Attribution 4.0 International.

Дословно из `license.txt`:

> AUTO.RIA Numberplate OCR RU Dataset. All data provided by the Habr.com community. Special thanks to Aziz Temirkhanov, German Cyganov. AUTO.RIA Numberplate Dataset is licensed under a Creative Commons Attribution 4.0 International License.

Разрешено:
- ✅ Коммерческое использование («for any purpose, even commercially»).
- ✅ Модификация и распространение производных.
- ✅ Публикация метрик и использование в маркетинговых материалах.

Обязательно:
- Указание авторства: исходный проект **Nomeroff Net / AUTO.RIA.com**, данные сообщества **Habr.com** (Aziz Temirkhanov, German Cyganov); ссылка на текст лицензии; указание факта изменений (если вносились).

Запрещено / ограничения:
- ❌ Распространение без указания авторства (нарушение условия Attribution).
- ❌ Применять технические/юридические ограничения, препятствующие другим пользоваться теми же свободами.

**Вывод для CARS (коммерческий продукт):** CC BY 4.0 — безусловно совместима с коммерческим использованием. Датасет можно применять для валидации `nomeroff_ocr` и публикации метрик при условии корректного указания авторства (Nomeroff Net / AUTO.RIA / Habr.com community).

---

## Ссылки

- [Прямая ссылка на архив](https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip)
- [Каталог датасетов Nomeroff Net](https://nomeroff.net.ua/datasets/)
- [Nomeroff Net (официальный сайт)](https://nomeroff.net.ua/)
- [ria-com/nomeroff-net (GitHub)](https://github.com/ria-com/nomeroff-net)
- [Соседняя спецификация — HuggingFace-производный датасет](car_plate_ocr_dataset_ru.md)
- [CC BY 4.0 (текст лицензии)](http://creativecommons.org/licenses/by/4.0/)
- [ГОСТ Р 50577-2018 (docs.cntd.ru)](https://docs.cntd.ru/document/1200160380)

---

## История изменений

- **2026-06-04** — Создана документация в рамках валидационной кампании на обновлённом datasets.md.
