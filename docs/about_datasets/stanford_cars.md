# Stanford Cars Dataset

## Общая информация

**Stanford Cars Dataset** — fine-grained датасет легковых автомобилей, опубликован Krause et al. в работе «3D Object Representations for Fine-Grained Categorization» (3DRR-13, ICCV 2013 workshop). Содержит 16,185 изображений 196 классов в формате *Make Model Year*.

**Локальное расположение:** _(скачивать при необходимости)_ — см. раздел «Получение».

**Оригинальный URL:** https://ai.stanford.edu/~jkrause/cars/car_dataset.html (**на 2026-05-15 возвращает 404; использовать зеркала Kaggle / Academic Torrents**).

---

## Структура датасета

```
stanford_cars/
├── cars_train/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...                 # 8,144 train images
│
├── cars_test/
│   ├── 00001.jpg
│   └── ...                 # 8,041 test images
│
├── devkit/
│   ├── cars_meta.mat       # 196 class names
│   ├── cars_train_annos.mat
│   └── cars_test_annos_withlabels.mat
│
└── (опционально) car_ims.tgz  # все изображения в одном архиве
```

---

## Статистика

| Компонент | Количество |
|-----------|------------|
| Всего изображений | 16,185 |
| Train | 8,144 |
| Test | 8,041 |
| Классов (Make + Model + Year) | 196 |
| Уникальных Make (агрегация) | ~49 |
| Среднее изображений на класс | ~82 (≈50/50 train/test) |

---

## Формат изображений

- JPEG, переменное разрешение (обычно 300–1000 пикселей по большой стороне).
- Цветовое пространство: RGB.
- Содержание: легковые автомобили, frontal / rear / 3-quarter view, преимущественно показательные / каталожные снимки.

---

## Классы (примеры)

196 классов в формате `<Make> <Model> <BodyType> <Year>`:

```
AM General Hummer SUV 2000
Acura RL Sedan 2012
Acura TL Sedan 2012
...
Audi A5 Coupe 2012
Audi R8 Coupe 2012
...
BMW M3 Coupe 2012
...
Volkswagen Golf Hatchback 2012
```

### Совпадение с CARS §6.3 (20 марок)

CARS требует 20 марок: Acura, Audi, BMW, Chevrolet, Chrysler, Dodge, Ford, GMC, Honda, Hyundai, Infiniti, Jeep, Kia, Lexus, Mazda, Mercedes, Nissan, Subaru, Toyota, Volkswagen.

**Покрытие Stanford Cars — 20/20** ✅. Все марки присутствуют в виде нескольких подклассов каждая (Acura RL/TL, Audi A5/R8/S4/..., BMW 1/3/M3/M5/...).

### Совпадение с CARS §6.3 Vehicle Type (6 типов)

CARS требует 6 типов: coupe, largevehicle, sedan, suv, truck, van.

Stanford Cars **не имеет** прямой type-аннотации, но body type закодирован в названии класса (`Sedan`, `Coupe`, `SUV`, `Hatchback`, `Convertible`, `Wagon`, `Cab`/`Truck`, `Van`, …).

**Маппинг Stanford → CARS type** (требует внешнего справочника):

| Stanford body | CARS type |
|---------------|-----------|
| Sedan | sedan |
| Coupe | coupe |
| SUV / Crossover | suv |
| Cab / Crew Cab / Regular Cab / Extended Cab | truck |
| Van / Minivan / Cargo Van | van |
| Hatchback / Wagon | sedan (или отдельная под-метка) |
| (нет прямого) largevehicle | largevehicle через специальные модели (Hummer, Suburban) |
| Convertible | coupe (или отдельно) |

---

## Формат аннотаций

### `cars_train_annos.mat` / `cars_test_annos_withlabels.mat`

MATLAB v5 формат. Структура:

```python
import scipy.io as sio

annos = sio.loadmat('cars_train_annos.mat')['annotations']
# Каждая аннотация:
# [bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id, filename]
```

| Поле | Тип | Описание |
|------|-----|----------|
| `bbox_x1` | int | левая координата bbox (px) |
| `bbox_y1` | int | верхняя координата bbox (px) |
| `bbox_x2` | int | правая координата bbox (px) |
| `bbox_y2` | int | нижняя координата bbox (px) |
| `class` | int (1–196) | индекс класса |
| `fname` | str | имя файла изображения |

### `cars_meta.mat`

Список из 196 названий классов в формате `<Make> <Model> <BodyType> <Year>`.

---

## Использование для CarCV

### Применимость

- ✅ **Vehicle Make (Top-1 >0.70, Top-3 >0.85)** — primary benchmark. 8,041 test images после mapping `class_id → Make` дают 20 макро-классов с балансом ~400 изображений на класс.
- ✅ **Vehicle Type (Acc >0.85)** — secondary через mapping `class_id → body type → CARS type`. Требует ручного справочника mapping (196 строк).
- ✅ BBox-аннотации позволяют использовать crop вместо полного кадра (соответствует §6.3 input 224×224).
- ✅ Нет лика с VehicleMakeNet/VehicleTypeNet (NVIDIA-internal training data).

### Ограничения

- ❌ **License: ImageNet-like / non-commercial research-only.** Использование для внутреннего benchmark CARS допустимо, но **публикация метрик в коммерческих материалах требует legal review.**
- ❌ Snapshots — преимущественно catalog/PR-shots, не дорожная съёмка. Дистрибуция не соответствует §5.1 (distance 5–50 м, angle 0–30°). Возможен domain gap.
- ❌ Класс distribution смещён к US/EU/JP/KR mainstream (соответствует CARS, но без российских марок — что для §6.3 как раз ОК).
- ❌ Класс distribution по году — модели 2000–2012, нет новых поколений (2013+).

### Рекомендации для CP3 / benchmark Make/Type

1. **Mapping table:** создать `docs/datasets/stanford_cars_to_cars_makes.csv` с двумя колонками:
   - `stanford_class_id` (1–196)
   - `cars_make` (one of 20 CARS makes)
   - и отдельно `cars_type` (one of 6 CARS types)
2. **Crop по bbox:** для inference в `VehicleMakeNet` подавать crop из bbox, не полное изображение.
3. **Метрики:**
   - Top-1 Make Accuracy = `mean(pred_make == gt_make)` по test 8,041.
   - Top-3 Make Accuracy = `mean(gt_make ∈ top3(model_logits))`.
   - Type Accuracy = `mean(pred_type == gt_type)` (после mapping).
4. **Confusion matrix** по 20 маркам — выявление систематических путаниц (Honda↔Acura, Lexus↔Toyota, ...).
5. **Сегментировать ошибки** по году выпуска модели — проверка устаревания training data CARS.

---

## Получение

Оригинальный URL `https://ai.stanford.edu/~jkrause/cars/car_dataset.html` на 2026-05-15 возвращает 404.

**Зеркала:**

```bash
# Kaggle (рекомендованное зеркало)
# https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
kaggle datasets download -d jessicali9530/stanford-cars-dataset

# Academic Torrents
# https://academictorrents.com/details/9c90b7f6208d430bff288845d45667ab2670da56
```

**HuggingFace:** `tanganke/stanford_cars` — также рабочее зеркало.

---

## Лицензия

«Similar to ImageNet license» — research-only, **non-commercial**.

Из ImageNet TOS:
- Можно использовать для академических исследований и публикаций.
- Запрещено перепродавать датасет.
- Коммерческое использование (включая публикацию метрик в маркетинговых материалах коммерческих продуктов) требует отдельного соглашения с правообладателем.

**Для CARS (комм. продукт):** Stanford Cars применим как **internal benchmark** (CP3-style оценка точности), но при публикации метрик во внешних артефактах необходим legal review (см. §7.2 research-документа).

---

## Ссылки

- [Krause J., Stark M., Deng J., Fei-Fei L. — 3D Object Representations for Fine-Grained Categorization (3DRR-13, ICCV 2013 workshop)](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- [Kaggle mirror](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- [HuggingFace mirror](https://huggingface.co/datasets/tanganke/stanford_cars)
- [Academic Torrents mirror](https://academictorrents.com/details/9c90b7f6208d430bff288845d45667ab2670da56)

---

## История изменений

- **2026-05-15** — Создана документация в рамках research-датасетов для валидации ML-стека CARS (см. `_bmad-output/planning-artifacts/research-datasets-validation.md`).
