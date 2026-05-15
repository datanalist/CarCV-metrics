# MAD-Cars (Yandex) — Multi-view Auto Dataset

## Общая информация

**MAD-Cars** — масштабный корпус 360° видеосъёмок автомобилей от Yandex Research. Создан в рамках проекта MADrive («Memory-Augmented Driving Scene Modeling», arXiv:2506.21520). Содержит **~70,000 уникальных автомобилей** со средним числом ~85 кадров на каждый, общим объёмом **5,884,329 frames** в разрешении преимущественно 1920×1080.

**Локальное расположение:** _(скачивается по URL из `url` column; смотри раздел «Получение и sampling»)_.

**HuggingFace:** https://huggingface.co/datasets/yandex/mad-cars

**Проектная страница:** https://yandex-research.github.io/madrive/

**Paper:** [Karpikova et al., «MADrive: Memory-Augmented Driving Scene Modeling», 2025](https://arxiv.org/abs/2506.21520)

**Дата выпуска:** 2025-06-26.

---

## Структура датасета

MAD-Cars представлен в виде **табличного датасета** (parquet, 2 файла) — изображения **не хранятся в архиве**, а загружаются по URL из колонки `url` в Yandex Cloud Storage.

```
mad-cars/
├── default/
│   └── train/
│       ├── 0.parquet        # ~50% строк
│       └── 1.parquet        # ~50% строк
│
└── (изображения на Yandex Cloud)
    https://storage.yandexcloud.net/yandex-research/mad/{car_id}/{view_id}.jpg
```

### Схема таблицы

| Поле | Тип | Описание |
|------|-----|----------|
| `car_id` | int64 | Идентификатор автомобиля (0–230K). |
| `view_id` | int64 | Идентификатор кадра/view для конкретного `car_id` (0–99). **Не привязан к фиксированной позиции камеры** — индекс кадра в 360° обходе. |
| `url` | str | Прямой HTTPS URL JPEG-изображения. Pattern: `https://storage.yandexcloud.net/yandex-research/mad/{car_id}/{view_id}.jpg` |
| `color` | str (16 классов) | **Hex RGB** значение цвета кузова (не имя класса!) — например `000000`, `ffffff`, `9966cc`. |
| `brand` | str (142 классов) | Lowercase brand (марка), например `vaz`, `bmw`, `kia`. |
| `model` | str | Lowercase model name (модель), например `granta`, `m3`. Свободная строка (1–26 символов). |

> **Важно:**
> - Нет `body_type` column.
> - Нет bbox или сегментации.
> - `view_id` НЕ соответствует фиксированному ракурсу — это просто индекс кадра в 360° обходе.

---

## Статистика

| Параметр | Значение |
|----------|----------|
| Всего frames | **5,884,329** |
| Уникальных автомобилей (`car_id`) | ~70,000 |
| Среднее frames на car | ~85 |
| Разрешение | большинство 1920×1080 |
| Уникальных `brand` | **142** |
| Уникальных `color` (hex) | **16** |
| Splits | только `train` |
| Полный размер при загрузке | **~2.9 TB** (≈500 KB × 5.88M frames) |

---

## Полный enumerated список `color` (16 hex RGB)

| Hex | Sample count | Mapping в CARS §6.5 |
|-----|--------------|----------------------|
| `000000` | 1,268,506 | black |
| `ffffff` | 1,358,819 | white |
| `9c9999` | 925,201 | grey |
| `cacecb` | 622,880 | silver |
| `0000ff` | 550,009 | blue |
| `0088ff` | 86,162 | blue (более светлый — объединить с `0000ff` для CARS) |
| `ff0000` | 323,454 | red |
| `cc0033` | 55,474 | red (бордо — объединить с `ff0000` для CARS) |
| `926547` | 242,321 | brown |
| `34ba2b` | 172,957 | green |
| `ffefd5` | 122,763 | beige (papaya whip — кремово-бежевый; **возможно поглощает `tan`**) |
| `ff9966` | 43,203 | orange |
| `9966cc` | 39,502 | purple |
| `fde910` | 38,150 | yellow |
| `ffcc00` | 33,640 | gold |
| `ffc0cb` | 910 | pink |

**Покрытие CARS §6.5 (15 цветов):** **14 из 15** ✅. Единственный gap — `tan` (нет отдельного hex; фактически объединён с beige `ffefd5` или brown `926547`).

⚠️ Класс `pink` представлен очень маленьким числом примеров (910 frames из 5.88M ≈ 0.015%). Для статистической оценки требуется либо oversampling, либо принимать wide confidence interval.

---

## Полный enumerated список `brand` (142, c покрытием CARS §6.3)

20 марок CARS §6.3 (см. ML_System_Design_Document.md) — все присутствуют:

| CARS make | `brand` | Frames |
|-----------|---------|--------|
| Acura | `acura` | 4,140 |
| Audi | `audi` | 204,047 |
| BMW | `bmw` | 383,548 |
| Chevrolet | `chevrolet` | 169,470 |
| Chrysler | `chrysler` | 8,965 |
| Dodge | `dodge` | 15,929 |
| Ford | `ford` | 236,040 |
| GMC | `gmc` | 3,135 |
| Honda | `honda` | 110,280 |
| Hyundai | `hyundai` | 373,767 |
| Infiniti | `infiniti` | 40,247 |
| Jeep | `jeep` | 22,013 |
| Kia | `kia` | 480,231 |
| Lexus | `lexus` | 82,861 |
| Mazda | `mazda` | 149,827 |
| Mercedes | `mercedes` | 287,793 |
| Nissan | `nissan` | 257,681 |
| Subaru | `subaru` | 42,754 |
| Toyota | `toyota` | 398,407 |
| Volkswagen | `volkswagen` | 357,991 |

**Покрытие: 20/20 ✅**. Минимальное представление — GMC (3,135 frames ≈ 35–40 уник. авто после dedup по `car_id`).

Прочие 122 бренда в датасете включают:
- **Российский рынок (potential extension):** vaz (LADA) — 702,785 (largest brand в датасете), uaz — 21,305, gaz — 13,931, moscvich — 3,886, zaz — 7,794, tagaz — 1,423, luaz — 192, derways — 172, vortex — 3,465.
- **Китайские:** chery (49,277), geely (54,090), haval (25,151), changan (27,411), great_wall (9,611), byd (2,778), и многие другие.
- **Прочие глобальные:** mitsubishi, opel, renault, skoda, peugeot, fiat, suzuki, daewoo, …

---

## Использование для CarCV

### Применимость

- ✅ **Color Recognition (§6.5) — primary benchmark, 14/15 классов** покрыты прямыми hex-значениями. Замещает Chen 2014 (7/15).
- ✅ **Vehicle Make (§6.3) — secondary benchmark (cross-distribution к Stanford Cars).** Multi-view per car (~85 views/instance) — тестирует robustness Make-классификатора к ракурсу.
- ⚠️ **Vehicle Type (§6.3) — слабый use case.** Нет `body_type` column; нужно строить mapping `(brand, model) → body_type` на тысячи моделей. Stanford Cars (где BodyType закодирован в имени класса) удобнее.
- ✅ Разрешение 1920×1080 — точное соответствие §5.1 CARS.
- ✅ Дата выпуска 2025-06 — **позже даты обучения NVIDIA TAO моделей**, лик невозможен по времени.

### Ограничения

- ❌ **Лицензия CC BY-NC-SA 4.0** — NonCommercial + ShareAlike. Использовать только как internal benchmark. Запрещено:
  - публиковать метрики в коммерческих whitepapers/маркетинге без отдельного соглашения с Yandex;
  - дообучать коммерческие модели на этих данных;
  - распространять crop-датасет / preprocessed dump под иной лицензией.
- ❌ Нет body_type → нужно строить mapping для Type (см. §6.3 CARS).
- ❌ Нет bbox → если автомобиль не занимает >80% кадра, требуется внешняя детекция или CARS TrafficCamNet pre-step.
- ❌ Размер 2.9 TB → полная скачка непрактична, **обязателен sampling**.
- ⚠️ Один split (`train`) → собственное разбиение на val/test делается на стороне CARS.
- ⚠️ Возможные дубликаты: ~85 views per car_id. Без dedup по `car_id` метрики будут смещены (одна машина → 85 «образцов»).
- ⚠️ `tan` слит с beige/brown → отдельная валидация `tan` требует custom labeling.

### Рекомендации по sampling

**Для Color validation (§6.5):**

```python
import pandas as pd

HEX_TO_CARS_COLOR = {
    "000000": "black", "ffffff": "white", "9c9999": "grey", "cacecb": "silver",
    "0000ff": "blue", "0088ff": "blue", "ff0000": "red", "cc0033": "red",
    "926547": "brown", "34ba2b": "green", "ffefd5": "beige",
    "ff9966": "orange", "9966cc": "purple", "fde910": "yellow",
    "ffcc00": "gold", "ffc0cb": "pink",
}
SAMPLES_PER_HEX = 200  # ≈ 3,200 frames всего

df = pd.read_parquet([
    "https://huggingface.co/api/datasets/yandex/mad-cars/parquet/default/train/0.parquet",
    "https://huggingface.co/api/datasets/yandex/mad-cars/parquet/default/train/1.parquet",
])

# 1 view per car_id, balanced per hex
df_dedup = df.groupby("car_id").first().reset_index()
sample = df_dedup.groupby("color").sample(n=SAMPLES_PER_HEX, random_state=42, replace=False)

# pink имеет всего ~910 frames → возможно <200 уникальных car_id → sample(replace=True) или skip
```

**Для Make validation (§6.3):**

```python
CARS_MAKES_20 = {"acura", "audi", "bmw", "chevrolet", "chrysler", "dodge", "ford",
                 "gmc", "honda", "hyundai", "infiniti", "jeep", "kia", "lexus",
                 "mazda", "mercedes", "nissan", "subaru", "toyota", "volkswagen"}
SAMPLES_PER_MAKE = 200

df_make = df[df["brand"].isin(CARS_MAKES_20)]
df_make_dedup = df_make.groupby("car_id").first().reset_index()
sample = df_make_dedup.groupby("brand").sample(n=SAMPLES_PER_MAKE, random_state=42)
# = 4000 frames ≈ 2 GB
```

**Для multi-view robustness test (опционально):**

Зафиксировать N car_id и взять все ~85 views для каждого. Например, по 5 car_id на каждую из 14 CARS colors = 70 cars × 85 views = ~6000 frames. Это позволит измерить stability цвета по ракурсу.

### Pipeline для валидации Color

1. Скачать sample с Yandex CDN (учитывать тарифы исходящего трафика РФ).
2. Применить детекцию автомобиля (TrafficCamNet) → crop (если на кадре не только автомобиль).
3. Resize crop → 384×384, нормализация ImageNet (mean=[0.43, 0.40, 0.39], std=[0.27, 0.26, 0.26] из §6.5).
4. Inference `bae_model_f3.onnx` → predicted color (1 из 15).
5. Сравнение с `HEX_TO_CARS_COLOR[row.color]`.
6. Метрики:
   - Overall Accuracy (target >0.80, §6.5).
   - Per-color Accuracy: best (black, white, red, blue) >0.90; challenging (beige, tan, gold, silver) >0.70.
   - Confusion matrix 15×15.

---

## Получение

```python
from datasets import load_dataset

# Загрузка таблицы (без изображений)
dataset = load_dataset("yandex/mad-cars", split="train")

# Скачать конкретное изображение
import requests
from PIL import Image
from io import BytesIO

row = dataset[0]
response = requests.get(row["url"], timeout=30)
img = Image.open(BytesIO(response.content))

# Группировка views по car_id
df = dataset.to_pandas()
car_id_to_views = df.groupby("car_id")["url"].agg(list)
```

Прямые URL parquet-файлов (если нужен только табличный slice без `datasets`):

- `https://huggingface.co/api/datasets/yandex/mad-cars/parquet/default/train/0.parquet`
- `https://huggingface.co/api/datasets/yandex/mad-cars/parquet/default/train/1.parquet`

---

## Лицензия

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (**CC BY-NC-SA 4.0**).

Разрешено:
- ✅ Использовать для **некоммерческих** целей.
- ✅ Делиться и адаптировать.

Запрещено / ограничено:
- ❌ Коммерческое использование (включая публикацию метрик в комм. продукте).
- ❌ Распространение производных под несовместимой лицензией (ShareAlike).

Обязательно:
- Указание авторов: Yandex Research (Karpikova et al., 2025).
- Распространение любых производных только под той же CC BY-NC-SA 4.0.

**Для CARS (комм. продукт):** строго **internal benchmark only.** Метрики не публиковать во внешних материалах без отдельного соглашения с Yandex Research.

---

## Ссылки

- [HuggingFace dataset card](https://huggingface.co/datasets/yandex/mad-cars)
- [MADrive: Memory-Augmented Driving Scene Modeling (arXiv:2506.21520)](https://arxiv.org/abs/2506.21520)
- [Project page](https://yandex-research.github.io/madrive/)
- [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## История изменений

- **2026-05-15** — Создана документация в рамках research-датасетов для валидации ML-стека CARS (см. `_bmad-output/planning-artifacts/research-datasets-validation.md`, §3.4, §6.4, §7.7). Добавлен в шорт-лист как Color primary (14/15 покрытия) и Make secondary (multi-view robustness).
