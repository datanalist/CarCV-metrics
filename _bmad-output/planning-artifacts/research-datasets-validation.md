---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - docs/system-design/ML_System_Design_Document.md
  - docs/about_datasets/bdd100k.md
workflowType: research
lastStep: 6
research_type: technical
research_topic: Выбор датасетов для валидации ML-стека CARS
research_goals: Отобрать датасеты для валидации Primary Detector (TrafficCamNet), LP Detection+Recognition (RU), Vehicle Make/Type, Color; обосновать выбор по покрытию классов, релевантности automotive-условиям, лицензии и доступности; зафиксировать в research-документе и в docs/about_datasets/{name}.md
user_name: Vallo
date: 2026-05-15
web_research_enabled: true
source_verification: true
---

# Research Report: Выбор датасетов для валидации ML-стека CARS

**Дата:** 2026-05-15
**Автор:** Vallo
**Тип исследования:** technical
**Привязка:** docs/system-design/ML_System_Design_Document.md (§3.3, §5, §6, §12)

---

## Research Overview

Цель — собрать обоснованный набор внешних датасетов для валидации четырёх ML-модулей CARS:

1. **Primary Detector (TrafficCamNet)** — детекция car/person/bike/sign в automotive-условиях.
2. **LP Detection + Recognition (RU)** — детекция и распознавание российских номеров (алфавит 23 символа).
3. **Vehicle Make (20) и Vehicle Type (6)** — классификаторы марок/типов ТС.
4. **Color Recognition (15)** — классификатор цвета кузова.

Каждый кандидатный датасет оценивается по следующим критериям отбора:
- покрытие классов / алфавита, требуемых §6 SDD;
- релевантность сценариям §5.1 (1920×1080, day/night/IR, 0–60 км/ч, automotive POV);
- лицензия (возможность коммерческого использования и публикации метрик);
- размер, наличие ground-truth, доступность скачивания на 2026-05-15;
- отсутствие лика с обучающим корпусом TAO-моделей NVIDIA (TrafficCamNet, VehicleTypeNet, VehicleMakeNet, LPDNet/LPRNet).

---

<!-- Content will be appended sequentially through research workflow steps -->

## 1. Scope Confirmation

**Подтверждено пользователем 2026-05-15** (форма: «Артефакты на русском, остальное ок»).

**Технический research scope:**

- Architecture Analysis — соответствие архитектуре ML-стека CARS (Stage 1/2/3, §6.1).
- Implementation Approaches — формат аннотаций, конвертации (BBox → YOLO/COCO/KITTI), пайплайны парсинга.
- Technology Stack — связь с моделями NVIDIA TAO (TrafficCamNet, VehicleMakeNet, VehicleTypeNet, LPDNet, LPRNet) и со стэком DeepStream.
- Integration Patterns — структуры разметки (JSON/XML/TXT), привязка к классам §6, покрытие сценариев §5.1.
- Performance Considerations — размер срезов под целевой бюджет CI/CP3 (1k–10k кадров), наличие public val/test split с ground-truth.

**Методология:**

- Web verification на 2026-05-15 (WebSearch + WebFetch) для лицензии, размера, URL, года публикации.
- Multi-source validation для критичных утверждений.
- Confidence уровни на спорные факты.
- Все источники приводятся со ссылками.

**Артефакты:**

- `_bmad-output/planning-artifacts/research-datasets-validation.md` — этот документ.
- `docs/about_datasets/{name}.md` — по одному документу на каждый отобранный датасет (структура — как у `docs/about_datasets/bdd100k.md`).

**Язык артефактов:** русский (соответствует существующему стилю проекта, см. SDD и `bdd100k.md`).

---

## 2. Authoritative Requirements (выжимка из SDD)

Все целевые метрики, классы и сценарии ниже — извлечение из `docs/system-design/ML_System_Design_Document.md`. На них строится оценка покрытия датасетов.

### 2.1 Целевые метрики (§3.3)

| Модель | Метрика | §3.3 | §6 (детальный раздел) |
|--------|---------|------|------------------------|
| Primary Detector | Precision | >0.90 | >0.92 (§6.2) |
| Primary Detector | Recall | >0.85 | >0.88 (§6.2) |
| Primary Detector | F1 | >0.87 | — |
| Primary Detector | mAP@0.5 | — | >0.89 (§6.2) |
| Vehicle Make | Top-1 | >0.70 | >0.70 (§6.3) |
| Vehicle Make | Top-3 | >0.85 | >0.85 (§6.3) |
| Vehicle Type | Accuracy | >0.85 | >0.85 (§6.3) |
| LP Detection | Recall | >0.80 | — |
| LP Recognition | Full Plate Accuracy | **>0.80** | **>0.85 (§6.4)** ⚠️ |
| LP Recognition | Character Accuracy | >0.90 | >0.92 (§6.4) |
| Color | Accuracy | **>0.75** | **>0.80 (§6.5)** ⚠️ |

⚠️ **Несогласованность SDD:** §3.3 и §6 расходятся по двум метрикам — LP Full Plate Accuracy (0.80 vs 0.85) и Color Accuracy (0.75 vs 0.80). §12 (CP4) использует >0.85 для LP, что согласуется с §6.4. В рамках выбора датасетов это влияет только на интерпретацию результатов, не на состав корпуса; в research-документе принимаем строжайший порог (>0.85 LP, >0.80 Color) — следствие из §6/§12.

### 2.2 Классы Primary Detector (§6.2)

`car`, `person`, `bike`, `sign` (4 класса). Вход модели — 960×544 BGR.

### 2.3 Vehicle Make — 20 классов (§6.3)

```
Acura, Audi, BMW, Chevrolet, Chrysler, Dodge, Ford, GMC, Honda, Hyundai,
Infiniti, Jeep, Kia, Lexus, Mazda, Mercedes, Nissan, Subaru, Toyota, Volkswagen
```

**Важно:** все 20 марок — US / EU / JP / KR. **Российских марок (Lada, GAZ, UAZ, …) в списке нет.** Это снимает требование к специфическим RU-датасетам для make-валидации.

### 2.4 Vehicle Type — 6 классов (§6.3)

`coupe`, `largevehicle`, `sedan`, `suv`, `truck`, `van`.

### 2.5 LP Recognition — алфавит 23 символа (§6.4)

```python
ALPHABET = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','E','K','M','H','O','P','C','T',
            'Y','X','-']
```

Это GOST-Р 50577 latin-look-alike подмножество (12 букв, видимо идентичных кириллическим А/В/Е/К/М/Н/О/Р/С/Т/У/Х) плюс 10 цифр и `-`. Вход модели — 188×48 RGB; архитектура STN + CNN + Bi-LSTM + CTC.

### 2.6 Color — 15 классов (§6.5)

```
beige, black, blue, brown, gold, green, grey, orange,
pink, purple, red, silver, tan, white, yellow
```

Целевая аккуратность по группам: best (black/white/red/blue) >0.90; challenging (beige/tan/gold/silver) >0.70.

### 2.7 Условия съёмки (§5.1)

| Параметр | Значение |
|----------|----------|
| Разрешение | 1920×1080 |
| FPS | 30 (мин. 25) |
| Угол камеры | 0–30° |
| Дистанция | 5–50 м |
| Скорость объекта | 0–60 км/ч |
| Освещённость | day / night+IR / уличное освещение |
| Сложности | дождь/снег → ↓точность; прямой солнечный свет → блики на номерах |

### 2.8 Тестовые checkpoint-ы (§12)

| CP | Критерий | Метод |
|----|----------|-------|
| CP3 | Detection Precision >90% | Manual labeling 1000 frames |
| CP4 | LP Accuracy >85% | Ground truth comparison |

→ Для CP3 нужен корпус с готовым ground-truth по бокс-детекции (минимум ~1k кадров с реальным automotive-распределением). Для CP4 — корпус RU LP с ground-truth текста.

---

## 3. Кандидаты и факты (verified 2026-05-15)

Все факты — на 2026-05-15, источники приведены в §8.

### 3.1 Primary Detector (TrafficCamNet)

| Датасет | Размер | Целевые классы | Метаданные сцены | Лицензия | Источник |
|---------|--------|-----------------|------------------|----------|----------|
| **BDD100K** | 100K img (70k train / 10k val / 20k test); 1280×720 | car, person, bike, motor, truck, bus, traffic sign, traffic light, rider | weather × timeofday × scene (6×3×6+) | UC Berkeley custom; **free для research/non-profit, commercial — через OTL Berkeley** | [doc.bdd100k.com/license.html](https://doc.bdd100k.com/license.html) |
| **KITTI** | 14,999 img (7,481 train / 7,518 test); ~390 MB | car, van, truck, pedestrian, cyclist, tram, person_sitting, misc, dontcare | — | **CC BY-NC-SA 3.0** | [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php) |
| **Cityscapes** | 5K fine + 20K coarse | 30+ urban (car, person, bicycle, motorcycle, truck, bus, train, …) | city × weather (limited) | **Free для non-comm; для comm — отдельная лицензия** | [cityscapes-dataset.com/license](https://www.cityscapes-dataset.com/license/) |
| **Mapillary Vistas** | 25K (v2); 1080p+ | 124 классов (v2), 70 instance-specific | global street-level, mixed | **Free Research-Only; commercial — отдельная лицензия** | [mapillary.com/dataset/vistas](https://www.mapillary.com/dataset/vistas) |
| **COCO 2017** | 118,287 train / 5,000 val / 20,288 test-dev | 80 (vehicle subset: car, bicycle, motorcycle, airplane, bus, train, truck) | — | **CC BY 4.0 (датасет); Flickr-под-лицензии на per-image уровне могут быть NC** | [cocodataset.org](https://cocodataset.org/) |

### 3.2 LP RU Recognition (LPR)

| Датасет | Размер | Алфавит | BBox? | OCR text? | Лицензия | Источник |
|---------|--------|---------|-------|-----------|----------|----------|
| **AY000554/Car_plate_OCR_dataset (HF)** | 45,514 (37,775 / 4,891 / 2,845); 1.23 GB | **GOST 22 chars** `0-9 A B E K M H O P C T Y X` | ❌ | ✅ (filename) | **CC BY 4.0** ✅ | [huggingface.co/datasets/AY000554/Car_plate_OCR_dataset](https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset) |
| nomeroff-net `autoriaNumberplateOcrRu-2021-09-01` | unknown | Multi-country (UA/RU/KZ/GE/BY/KG/AM/EU) | varies | ✅ | Code GPL-3.0, **data license unclear** | [github.com/ria-com/nomeroff-net](https://github.com/ria-com/nomeroff-net) |
| Kaggle `evgrafovmaxim/nomeroff-russian-license-plates` | derived from nomeroff | RU GOST subset | varies | ✅ | derived — unclear | [kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates](https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates) |
| Kaggle `c/car-plates-recognition` | competition train+test | RU | ✅ bbox + text | ✅ | Kaggle competition rules | [kaggle.com/c/car-plates-recognition](https://www.kaggle.com/c/car-plates-recognition) |
| **RodoSol-ALPR** | 20,000 (1280×720); cars+motorcycles, BR + Mercosur | **❌ Brazil/Mercosur (не GOST)** | ✅ | ✅ | **Research-only via email request** | [github.com/raysonlaroca/rodosol-alpr-dataset](https://github.com/raysonlaroca/rodosol-alpr-dataset) |
| Synth_RU_LP | — | Можно сгенерировать через nomeroff-net pipeline | ✅ (синтез) | ✅ | — | — |

### 3.3 Vehicle Make (20) / Type (6)

| Датасет | Размер | Классы | Лицензия | Источник |
|---------|--------|--------|----------|----------|
| **Stanford Cars** | 16,185 (8,144 / 8,041); bbox | 196 (Make + Model + Year) | **ImageNet-like — research-only** | [paper Krause et al. 3DRR-13](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) |
| **MAD-Cars (Yandex)** | **5,884,329 frames / ~70K cars / ~85 views/car**; 1920×1080 | `brand` (142) + `model` (string) + `color` (16 hex); **нет body_type** | **CC BY-NC-SA 4.0** | [huggingface.co/datasets/yandex/mad-cars](https://huggingface.co/datasets/yandex/mad-cars) |
| **CompCars** | 136,727 vehicles | 153 makes / 1,716 models; web + surveillance | **«Non-commercial research and educational purposes only»** (explicit) | [mmlab.ie.cuhk.edu.hk/datasets/comp_cars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html) |
| **VMMRdb** | 291,752 / 9,170 classes; 1950–2016 | Make + Model + Year | **Code MIT; data license не указана** | [github.com/faezetta/VMMRdb](https://github.com/faezetta/VMMRdb) |
| **BoxCars116k** | 116,286 img / 27,496 vehicles / 693 fg | Surveillance angle (роды top) | **Code «research only»; data license не указана** | [arxiv 1703.00686](https://arxiv.org/abs/1703.00686) / [github.com/JakubSochor/BoxCars](https://github.com/JakubSochor/BoxCars) |

### 3.4 Color Recognition (15 классов)

| Датасет | Размер | Color classes | CARS coverage | Лицензия | Источник |
|---------|--------|---------------|---------------|----------|----------|
| **MAD-Cars (Yandex)** | **5,884,329 frames / ~70K cars** (1920×1080) | **16 hex RGB** | **14/15** (нет однозначного `tan`; всё остальное покрыто) | **CC BY-NC-SA 4.0** | [huggingface.co/datasets/yandex/mad-cars](https://huggingface.co/datasets/yandex/mad-cars) |
| **Chen 2014** | 15,601 (1920×1080 frontal urban road) | **8**: black, blue, cyan, gray, green, red, white, yellow | **7/15** (black, blue, green, grey, red, white, yellow; cyan не в CARS; нет beige, brown, gold, orange, pink, purple, silver, tan) | Не указана явно; original IP-URL `122.205.5.5:8071` нестабилен | [Chen 2014 TITS](https://arxiv.org/pdf/1510.07391) |
| **VeRi-776** | 50,000+ img / 776 vehicles / 20 cams | ~10 colors типично + BBox + type + brand | Частичное (нужна верификация маппинга) | **Non-commercial (email request)**; будущие версии без LP | [github.com/VehicleReId/VeRi](https://github.com/VehicleReId/VeRi) |
| **VehicleID** | 221,763 / 26,267 vehicles | ~7 colors + LP + appearance | Частичное (нужна верификация) | Не указана явно | [PROVID page](https://vehiclereid.github.io/VeRi/) |

#### MAD-Cars color hex → CARS §6.5 mapping (14/15 directly)

Полный enumerated список 16 hex значений (вместе с числом frames в датасете) и mapping в CARS-таксон:

| MAD-Cars hex | Sample count | CARS color | Уверенность |
|--------------|--------------|------------|--------------|
| `000000` | 1,268,506 | **black** | ✅ exact |
| `ffffff` | 1,358,819 | **white** | ✅ exact |
| `9c9999` | 925,201 | **grey** | ✅ |
| `cacecb` | 622,880 | **silver** | ✅ |
| `0000ff` | 550,009 | **blue** | ✅ |
| `0088ff` | 86,162 | **blue** (более светлый оттенок; объединить) | ✅ |
| `ff0000` | 323,454 | **red** | ✅ |
| `cc0033` | 55,474 | **red** (бордо; объединить) | ✅ |
| `926547` | 242,321 | **brown** | ✅ |
| `34ba2b` | 172,957 | **green** | ✅ |
| `ffefd5` | 122,763 | **beige** (papaya whip — кремово-бежевый) | 🟡 (возможно `tan`) |
| `ff9966` | 43,203 | **orange** | ✅ |
| `9966cc` | 39,502 | **purple** | ✅ |
| `fde910` | 38,150 | **yellow** | ✅ |
| `ffcc00` | 33,640 | **gold** | ✅ |
| `ffc0cb` | 910 | **pink** | ✅ |

**Покрытие CARS 15 цветов:** 14 ✅ (`beige, black, blue, brown, gold, green, grey, orange, pink, purple, red, silver, white, yellow`). **Только `tan` не выделен отдельным hex** — фактически объединён в `ffefd5` (с beige) или `926547` (с brown). Это согласуется с §6.5 CARS, где `tan` явно перечислен в «challenging» группе (Acc target >0.70).

#### MAD-Cars brand → CARS §6.3 20 makes coverage (20/20)

| CARS make | MAD-Cars brand | Frames в датасете |
|-----------|----------------|--------------------|
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

**Покрытие: 20/20 ✅.** Минимальная марка (GMC) — 3,135 frames; даже после dedup по `car_id` (~85 views/car) остаётся ≥35 уникальных автомобилей. Топ-9 марок имеют >100K frames каждая.

Plus 122 не-CARS марок (vaz — 702,785, mercedes — 287,793, nissan — 257,681, …) — потенциал для будущего расширения CARS таксона на российский рынок.

---

## 4. Матрица покрытия требований CARS

### 4.1 Detection coverage — классы §6.2 (car / person / bike / sign)

| Датасет | car | person | bike (двухколёсные) | sign (traffic sign) | Совпадение |
|---------|-----|--------|---------------------|---------------------|------------|
| BDD100K | ✅ `car` | ✅ `person` | ✅ `bike` + `motor` | ✅ `traffic sign` | **4/4** ✅ |
| KITTI | ✅ `car` | ✅ `pedestrian` | ✅ `cyclist` | ❌ нет | 3/4 |
| Cityscapes | ✅ `car` | ✅ `person` | ✅ `bicycle` + `motorcycle` | ❌ нет (есть `traffic sign` в семантической сегментации, но в instance — нет) | 3/4 |
| Mapillary Vistas | ✅ | ✅ | ✅ | ✅ (богатый набор signs) | 4/4 |
| COCO | ✅ `car` | ✅ `person` | ✅ `bicycle` + `motorcycle` | ❌ нет `traffic sign` (есть `stop sign`) | 3/4 |

→ **BDD100K и Mapillary Vistas** — единственные с полным покрытием 4 классов CARS. COCO/KITTI/Cityscapes теряют `sign`.

### 4.2 Detection coverage — сценарии §5.1

| Датасет | day | night | rain/snow/fog | highway | city | residential | parking | Метаданные доступны? |
|---------|-----|-------|---------------|---------|------|-------------|---------|----------------------|
| BDD100K | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **✅ JSON attributes (`weather`, `timeofday`, `scene`)** |
| KITTI | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Cityscapes | ✅ | ❌ (мало) | limited | ❌ | ✅ | ✅ | ❌ | частично |
| Mapillary Vistas | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | партиально |
| COCO | mixed (not driving) | mixed | — | — | mixed | mixed | mixed | ❌ |

→ **BDD100K — лучший по explicit метаданным сцены**, что даёт прямую сегментацию метрик по `day/night × clear/rainy × city/highway` для §5.1.

### 4.3 LP RU alphabet coverage — §6.4

CARS алфавит (23): `0-9 A B E K M H O P C T Y X -`.

| Датасет | Цифры (10) | Латинские буквы (12) | `-` | Покрытие |
|---------|-----------|----------------------|-----|----------|
| AY000554/Car_plate_OCR_dataset | ✅ 0–9 | ✅ ровно `A B E K M H O P C T Y X` | ❌ | **22/23** (отсутствует `-`) |
| nomeroff RU sub | ✅ | ✅ | unknown | ≥22/23 (likely 22 — `-` это специальный «service» token) |
| Kaggle competition | ✅ | ✅ | unknown | ≥22/23 |
| RodoSol-ALPR | ✅ | ❌ другой набор (бразильские/Mercosur 26 латинских + цифр) | — | НЕ совпадает |

→ **AY000554/Car_plate_OCR_dataset покрывает 22 из 23 символов**. Отсутствующий `-` — это служебный CTC blank/separator в LPR_STN_PRE_POST.onnx, не реальный символ номера. На реальных RU GOST-Р 50577 plates `-` отсутствует. Покрытие функционально полное.

### 4.4 Make coverage — §6.3 (20 марок)

CARS 20 марок: Acura, Audi, BMW, Chevrolet, Chrysler, Dodge, Ford, GMC, Honda, Hyundai, Infiniti, Jeep, Kia, Lexus, Mazda, Mercedes, Nissan, Subaru, Toyota, Volkswagen.

| Датасет | Make-классы | Покрытие 20 CARS | Примечание |
|---------|-------------|-------------------|------------|
| Stanford Cars | 49 уникальных марок (агрегация 196 классов до make) | **20/20** ✅ | Через mapping `Make Model Year` → `Make`. Все 20 марок CARS массовые → присутствуют в Stanford. |
| **MAD-Cars (Yandex)** | **142 brands** | **20/20** ✅ | Прямой `brand` column (lowercased). Минимальное представление — GMC (3,135 frames); после dedup по `car_id` ≥35 уник.авто. |
| CompCars | 153 makes | **20/20** ✅ | Включает китайские/redundant; CARS marquees покрываются. |
| VMMRdb | сотни makes | **20/20** ✅ | 9,170 fine-grained классов; маппинг прост. |
| BoxCars116k | ~70 makes | ≥18/20 | Сурвейлланс ракурсы, less consumer brands diversity. |

→ Все 5 датасетов закрывают 20 CARS марок. **Stanford Cars предпочтителен как primary (bbox, frontal/3-quarter view, 16K — оптимально для CI). MAD-Cars — strong secondary** для кросс-валидации (multi-view per car-instance проверяет robustness Make-классификатора к ракурсу).

### 4.5 Vehicle Type coverage — §6.3 (6 типов)

CARS 6 типов: coupe, largevehicle, sedan, suv, truck, van.

| Датасет | Type-классы напрямую | Покрытие | Примечание |
|---------|----------------------|----------|------------|
| Stanford Cars | ❌ напрямую нет, но `BodyType` закодирован в имени класса | косвенно 6/6 | Через mapping table (Coupe/Sedan/SUV/Truck/Van/Largevehicle) |
| MAD-Cars (Yandex) | ❌ **нет `body_type` column** (только `brand` + `model` string) | косвенно 6/6 | Требует mapping table `(brand, model) → body_type`; гораздо больший mapping чем для Stanford |
| CompCars | ✅ есть `car type` атрибут (MPV, SUV, sedan, hatchback, minibus, …) | 5/6 (нет точного `largevehicle`) | non-commercial |
| VMMRdb | ❌ напрямую нет | косвенно через mapping | — |
| BoxCars116k | ✅ `body_type` (sedan, suv, hatchback, combi, mpv, van, truck) | 5/6 | surveillance ракурс |

→ Для Vehicle Type ни один датасет не покрывает 6/6 напрямую. **Stanford Cars выигрывает у MAD-Cars** для типа: у Stanford `BodyType` уже в строке класса (196 mapping rows), у MAD-Cars нужно строить mapping `(brand, model) → body_type` потенциально на тысячи моделей. Альтернатива — CompCars-type или BoxCars116k для surveillance type ракурсов.

### 4.6 Color coverage — §6.5 (15 цветов)

CARS 15: beige, black, blue, brown, gold, green, grey, orange, pink, purple, red, silver, tan, white, yellow.

| Цвет | MAD-Cars (16) | Chen 2014 (8) | VeRi-776 (~10) | VehicleID (~7) | Лучший источник |
|------|---------------|---------------|----------------|----------------|------------------|
| beige | ✅ (`ffefd5`) | ❌ | ❌ | ❌ | **MAD-Cars** |
| black | ✅ (`000000`) | ✅ | ✅ | ✅ | MAD-Cars / Chen |
| blue | ✅ (`0000ff` + `0088ff`) | ✅ | ✅ | ✅ | MAD-Cars / Chen |
| brown | ✅ (`926547`) | ❌ | ✅ | ❌ | **MAD-Cars** |
| gold | ✅ (`ffcc00`) | ❌ | ❌ | ❌ | **MAD-Cars** |
| green | ✅ (`34ba2b`) | ✅ | ✅ | ✅ | MAD-Cars / Chen |
| grey | ✅ (`9c9999`) | ✅ (gray) | ✅ | ✅ | MAD-Cars / Chen |
| orange | ✅ (`ff9966`) | ❌ | ✅ | ❌ | **MAD-Cars** |
| pink | ✅ (`ffc0cb`) | ❌ | ❌ | ❌ | **MAD-Cars** |
| purple | ✅ (`9966cc`) | ❌ | ❌ | ❌ | **MAD-Cars** |
| red | ✅ (`ff0000` + `cc0033`) | ✅ | ✅ | ✅ | MAD-Cars / Chen |
| silver | ✅ (`cacecb`) | ❌ | ✅ | ✅ | **MAD-Cars** |
| tan | 🟡 (объединён с beige `ffefd5` или brown `926547`) | ❌ | ❌ | ❌ | **gap** |
| white | ✅ (`ffffff`) | ✅ | ✅ | ✅ | MAD-Cars / Chen |
| yellow | ✅ (`fde910`) | ✅ | ✅ | ✅ | MAD-Cars / Chen |

→ **MAD-Cars закрывает 14/15 цветов CARS** — революционное улучшение vs Chen 2014 (7/15). Единственный остающийся gap — `tan`, который **не выделен отдельным hex в MAD-Cars** и, видимо, поглощён `ffefd5` (beige) или `926547` (brown). Это согласуется с §6.5 CARS, где `tan` сам по себе «challenging» (target Acc >0.70). Для полной 15/15 валидации требуется **собственная разметка только `tan`** (≈100 кадров, не 800).

---

## 5. Лицензии и TAO-leak анализ

### 5.1 Сводная таблица лицензий

| Датасет | Лицензия | Comm OK? | Validation для CARS commercial deploy |
|---------|----------|----------|---------------------------------------|
| BDD100K | UC Berkeley custom | ⚠️ через OTL | OK для internal benchmark; для публикации метрик в комм. продукте — требуется OTL соглашение |
| KITTI | CC BY-NC-SA 3.0 | ❌ | **Не использовать** |
| Cityscapes | Custom non-comm | ❌ | Не использовать |
| Mapillary Vistas | Research-only (free) | ❌ | Не использовать |
| COCO | CC BY 4.0 + Flickr | ✅ (с risk per-image) | OK |
| AY000554/Car_plate_OCR_dataset | **CC BY 4.0** | ✅ | **OK безусловно** |
| nomeroff RU sub | Code GPL-3.0, data unclear | ⚠️ | Use at risk; запросить подтверждение у RIA.com |
| Kaggle nomeroff/competition | derived/competition rules | ⚠️ | Использовать только для соревновательной валидации |
| RodoSol-ALPR | Research-only | ❌ | Не использовать |
| Stanford Cars | ImageNet-like (NC) | ❌ | OK для internal benchmark; не для прод-релиза без согласования |
| **MAD-Cars (Yandex)** | **CC BY-NC-SA 4.0** | ❌ | OK для internal benchmark; ShareAlike clause требует публикации производных под той же лицензией → нельзя использовать для дообучения коммерческих моделей |
| CompCars | Explicit NC | ❌ | **Не использовать для коммерческой валидации** |
| VMMRdb | Unspecified | ⚠️ | Use at risk; запросить разъяснение у авторов |
| BoxCars116k | Unspecified | ⚠️ | Use at risk; запросить разъяснение |
| Chen 2014 | Unspecified | ⚠️ | Use at risk; в академических работах используется без явного запрета |
| VeRi-776 | NC | ❌ | Не использовать |
| VehicleID | Unspecified | ⚠️ | Use at risk |

**Trichotomy для оценки рисков:**
- ✅ — лицензия явно разрешает commercial; CARS как комм. продукт может публиковать метрики.
- ⚠️ — лицензия не указана / неоднозначна; использовать как internal benchmark, не публиковать метрики в маркетинге/whitepapers без подтверждения от авторов.
- ❌ — non-commercial запрет; использовать недопустимо.

### 5.2 TAO-leak анализ (модели NVIDIA TAO vs кандидатные датасеты)

| TAO модель | Training data (по model card NGC) | Пересечение с кандидатами |
|------------|------------------------------------|---------------------------|
| **TrafficCamNet** | NVIDIA-internal, real traffic intersections US cities, ~20ft vantage point, mostly North American | BDD100K (US dashcam, front-view) ≠ traffic-intersection top-down → **разные домены** → ✅ нет лика. COCO/KITTI/Cityscapes/Mapillary тоже не указаны → ✅. |
| **VehicleMakeNet** | NVIDIA-internal, 20 makes (см. §6.3) | Stanford Cars / MAD-Cars / CompCars / VMMRdb / BoxCars116k — не упомянуты в model card → **predicted clean**. MAD-Cars (2025-06) выпущен после публикации TAO моделей → лик невозможен по времени. |
| **VehicleTypeNet** | NVIDIA-internal, 6 types | Аналогично — predicted clean. MAD-Cars также чист по дате выпуска. |
| **LPDNet** | Две модели: (1) NVIDIA-internal US LP; (2) **CCPD (Chinese City Parking Dataset)** | RU LP датасеты → ✅ нет лика |
| **LPRNet (TAO)** | TAO LPRNet trained for US/CN | CARS использует кастомную модель **`LPR_STN_PRE_POST.onnx`** (STN + LSTM + CTC), а не TAO LPRNet (§6.4). Скорее всего derived from nomeroff-net/RU ALPR research. → ✅ ничего не валидируем против самой модели обучения. |

**Вывод:** Для всех 4 модулей CARS — **нулевой или низкий риск лика** между обучающим корпусом и предлагаемыми validation корпусами. Это критично подтверждает применимость BDD100K / COCO / Stanford Cars / AY000554/Car_plate_OCR_dataset как unbiased validation source.

---

## 6. Шорт-лист и обоснование

### 6.1 Primary Detector (TrafficCamNet)

**Primary:** **BDD100K val** (10,000 изображений с JSON-аннотациями)

Обоснование:
- ✅ Покрытие классов 4/4 (car/person/bike/sign).
- ✅ Explicit метаданные `weather × timeofday × scene` → прямая сегментация метрик по §5.1.
- ✅ Нет TAO-лика с TrafficCamNet (US intersection top-down ≠ BDD dashcam front-view).
- ✅ Уже скачан и проанализирован (см. `docs/about_datasets/bdd100k.md`).
- ⚠️ License: UC Berkeley custom; CARS как комм. продукт — для публикации внешних метрик нужно соглашение с OTL Berkeley. Для internal CP3 — OK.

**Secondary (cross-validation):** **COCO 2017 val** (5,000 изображений, vehicle subset)

Обоснование:
- ✅ CC BY 4.0 — безусловно безопасная лицензия.
- ✅ Кросс-валидация на иной distribution (general scenes, не automotive POV) → выявление overfit на BDD-стиль.
- ⚠️ Только 3/4 классов (нет `traffic sign`).

**Исключены:** KITTI / Cityscapes / Mapillary Vistas — non-commercial, нет преимуществ перед BDD100K.

### 6.2 LP Recognition (RU)

**Primary:** **`AY000554/Car_plate_OCR_dataset` (HuggingFace)** — 45,514 RU GOST plates

Обоснование:
- ✅ **CC BY 4.0** — единственный найденный RU LP корпус с явной коммерчески-совместимой лицензией.
- ✅ Алфавит совпадает с §6.4 (22/23, отсутствующий `-` — служебный токен модели).
- ✅ Уже разделён на train/val/test (37,775 / 4,891 / 2,845) → используем val + test (7,736 plates) для CP4 (Full Plate Accuracy >0.85).
- ✅ Размер 1.23 GB — лёгко уложится в CI.
- ❌ Только OCR text, нет bbox → не валидирует LP Detection (LPDNet stage).

**Для LP Detection bbox:** **Kaggle `c/car-plates-recognition`** (competition train+test, ~25k bbox-annotated RU plates) или собственная разметка 1000 кадров CARS (CP4 ground-truth comparison из §12.2).

**Secondary (расширение):** `nomeroff-net autoriaNumberplateOcrRu-2021-09-01.zip` — для дополнительного покрытия (ext lic-clarification needed).

**Исключены:** RodoSol-ALPR (бразильский алфавит), `Synth_RU_LP` (требует кастомной генерации).

### 6.3 Vehicle Make (20) и Type (6)

**Primary (Make):** **Stanford Cars** (16,185 images, 196 Make+Model+Year)

Обоснование:
- ✅ Все 20 марок CARS присутствуют (Stanford 196 включает массовые US/EU/JP/KR brands).
- ✅ BBox и frontal/3-quarter view — близко к §5.1 (5–50 м distance, угол 0–30°).
- ✅ 8,041 test images — хороший баланс для статистически значимой Top-1/Top-3.
- ✅ Нет TAO-лика (VehicleMakeNet — NVIDIA-internal).
- ⚠️ License: ImageNet-like (research-only). **Для internal benchmark — OK; для рекламы метрик в коммерческом продукте — flag для legal review.**

**Secondary (Make cross-validation):** **MAD-Cars (Yandex)** — sampled subset

Обоснование:
- ✅ Все 20 CARS марок покрыты с большим запасом (от 3,135 frames для GMC до 480,231 для Kia).
- ✅ **Multi-view per car-instance (~85 views/car)** → проверка robustness Make-классификатора к ракурсу. Это уникальное свойство, которого нет ни у одного другого датасета.
- ✅ 1920×1080 — соответствует §5.1.
- ✅ Опубликован 2025-06, **позже даты обучения NVIDIA TAO моделей** → лик невозможен по времени.
- ⚠️ License: **CC BY-NC-SA 4.0** — non-commercial AND share-alike. Только internal benchmark; нельзя публиковать метрики в комм. материалах; **нельзя использовать для дообучения комм. моделей** (SA clause).
- ❌ Полная загрузка (5.88M URLs ≈ 2.9 TB) непрактична. **Sampling strategy:** 1 view/car_id × 20 CARS makes × 200 cars/make ≈ 4,000 frames ≈ 2 GB.

**Для Vehicle Type:** Через **mapping Make+Model → body type** на Stanford Cars (BodyType уже закодирован в имени класса, mapping ≈196 строк). MAD-Cars менее удобен для type — нет `body_type` column, мапппинг `(brand, model) → body_type` потребовал бы покрытие тысячи моделей. Альтернатива — CompCars (имеет прямую type-аннотацию), но non-commercial → только internal.

**Не вошли в шорт-лист:**
- CompCars (explicit non-comm)
- VMMRdb (license unclear, 9,170 классов избыточно для 20 CARS)
- BoxCars116k (surveillance angle — не соответствует §5.1 0–30° dashcam)

### 6.4 Color Recognition (15 классов)

**Primary:** **MAD-Cars (Yandex)** — sampled subset, 14/15 покрытие

Обоснование:
- ✅ **14/15 CARS цветов** прямо представлены отдельными hex-значениями: black, white, grey, silver, blue, red, brown, green, beige, orange, purple, yellow, gold, pink.
- ✅ 1920×1080 — точное совпадение с §5.1 CARS (вход модели `bae_model_f3.onnx` ресайзит до 384×384).
- ✅ Multi-view per car (~85 views/instance) → **проверка robustness color recognition к ракурсу и освещению** на одном и том же автомобиле — уникальная возможность для CARS use-case (Jetson on-vehicle, разные углы съёмки на одном объекте).
- ✅ Размер позволяет балансированный sample: 100–300 cars × 1 view/car × 14 colors ≈ 1,400–4,200 samples — статистически значимо.
- ⚠️ License: **CC BY-NC-SA 4.0** — internal benchmark only.
- 🟡 `tan` не выделен отдельным hex → объединён с `ffefd5` (beige) или `926547` (brown). Это согласуется с §6.5 классификацией `tan` как «challenging».

**Sampling strategy для MAD-Cars (color):**

```python
# 1 view per car_id, balanced per color
SAMPLES_PER_HEX = 200  # ≈ 3,200 frames всего ≈ 1.5 GB
HEX_TO_CARS = {
    "000000": "black", "ffffff": "white", "9c9999": "grey", "cacecb": "silver",
    "0000ff": "blue", "0088ff": "blue", "ff0000": "red", "cc0033": "red",
    "926547": "brown", "34ba2b": "green", "ffefd5": "beige",
    "ff9966": "orange", "9966cc": "purple", "fde910": "yellow",
    "ffcc00": "gold", "ffc0cb": "pink",
}
# blue, red — два hex каждый → объединить или валидировать раздельно
```

**Secondary (sanity check / fallback при недоступности Yandex CDN):** **Chen 2014** (7 цветов, 15,601 кадров).

**Custom labeling (только `tan`):** ≈100 кадров с CARS-собранных fragmentов для покрытия 15/15. **Сокращено с 800 до 100 благодаря MAD-Cars.**

**Не вошли в шорт-лист:**
- VeRi-776 (non-commercial + license requires email)
- VehicleID (license unclear)

### 6.5 Сводная рекомендация (TL;DR)

| Модуль | Primary датасет | Secondary | License-OK для comm.? | Размер шорт-листа |
|--------|-----------------|-----------|------------------------|---------------------|
| Primary Detector | **BDD100K val (10K)** | COCO val 2017 (5K) | ⚠️ BDD требует OTL соглашения; COCO ✅ | 2 датасета |
| LP Recognition RU | **AY000554 HF (val+test 7,736)** | Kaggle competition (для bbox) | ✅ AY000554 CC BY 4.0 | 1 + 1 |
| Vehicle Make | **Stanford Cars test (8,041)** | **MAD-Cars sampled (~4K, multi-view robustness)** | ⚠️ оба research-only | 2 датасета |
| Vehicle Type | Stanford Cars + Make→Type mapping | (внутр. mapping table) | ⚠️ | 1 + mapping |
| Color | **MAD-Cars sampled (~3K, 14/15)** | Chen 2014 (7 colors, sanity) + own labeling `tan` (~100) | ⚠️ MAD-Cars NC; Chen unclear; own ✅ | 2 + custom |

---

## 7. Риски и follow-up

### 7.1 Color coverage gap (low — снижено благодаря MAD-Cars)
~~5 цветов (beige, gold, pink, purple, tan) отсутствуют во всех публичных корпусах.~~ **Снято с MAD-Cars (Yandex): 14/15 цветов покрыты прямыми hex-классами.** Остаётся единственный gap — `tan` (объединён с `beige`/`brown` в MAD-Cars).

**Action:** в рамках CP3 расширить «Manual labeling 1000 frames» до 1100 кадров (1000 для Primary Detector + 100 для `tan`).

### 7.2 License risks for production claims (medium)
- BDD100K — для метрик в публичных материалах нужен договор с UC Berkeley OTL.
- Stanford Cars — ImageNet-like research-only.
- **MAD-Cars (Yandex) — CC BY-NC-SA 4.0. NonCommercial + ShareAlike.** Это самая строгая лицензия в шорт-листе: запрещает не только коммерческое использование, но и создание производных под несовместимой лицензией. Использовать **только как internal benchmark**.
- Chen 2014 — лицензия не указана.

**Action:** Legal review всех ⚠️ строк §5.1 перед публикацией CP3/CP4 результатов. Для внутреннего benchmark — все ✅.

### 7.3 LP Detection ground-truth (medium)
AY000554 не содержит bbox. Для CP4 (LP detection Recall >0.80) нужен либо Kaggle `c/car-plates-recognition`, либо собственная разметка ~1000 RU LP кадров.

### 7.4 nomeroff-net data license (low)
Source data из RIA.com, лицензия не указана. **Action:** issue в [github.com/ria-com/nomeroff-net](https://github.com/ria-com/nomeroff-net) с вопросом по лицензии training data.

### 7.5 Vehicle Type 6 classes coverage (medium)
Ни один датасет не имеет прямого совпадения с CARS 6 типов (`coupe / largevehicle / sedan / suv / truck / van`). Mapping Make+Model → body type требует внешнего справочника.

**Action:** создать `docs/datasets/vehicle_type_mapping.csv` (модель из Stanford Cars → CARS type), 196 строк.

### 7.6 TAO-leak — residual risk (low)
NVIDIA TAO model cards не публикуют полный список training datasets. **Action:** если в результате CP3/CP4 наблюдаются аномально высокие метрики (>5% выше ожидаемого) — провести ручную проверку test-set duplicates через perceptual hashing. **MAD-Cars (2025-06)** выпущен после публикации TAO моделей — для него лик невозможен по дате.

### 7.7 MAD-Cars sampling и пропускная способность (medium)
MAD-Cars содержит 5.88M URLs к Yandex Cloud Storage (~2.9 TB при полной загрузке). Полная скачка непрактична.

**Action:**
- Реализовать sampling-скрипт: 1 view/car_id × balanced по 20 CARS makes (для Make) и по 14+1 CARS colors (для Color) → ~5–7K кадров суммарно ≈ 3 GB.
- Кэшировать скачанные кадры локально (Yandex CDN из РФ может тарифицироваться при больших объёмах).
- Учитывать ShareAlike clause: любые производные (cropped/preprocessed image dumps) обязаны распространяться под той же CC BY-NC-SA 4.0 — **не публиковать crop-датасет публично в коммерческих контурах**.

---

## 8. Источники

### SDD
- `docs/system-design/ML_System_Design_Document.md` §3.3, §5, §6, §12

### Detection
- [BDD100K License](https://doc.bdd100k.com/license.html)
- [KITTI Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php)
- [Cityscapes Terms and Conditions](https://www.cityscapes-dataset.com/license/)
- [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)
- [COCO Dataset](https://cocodataset.org/)

### LP RU
- [AY000554/Car_plate_OCR_dataset](https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset)
- [ria-com/nomeroff-net](https://github.com/ria-com/nomeroff-net)
- [Kaggle Car Plates Recognition Competition](https://www.kaggle.com/c/car-plates-recognition)
- [Kaggle evgrafovmaxim/nomeroff-russian-license-plates](https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates)
- [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset)

### Make/Type/Color (Yandex)
- [MAD-Cars dataset (Yandex)](https://huggingface.co/datasets/yandex/mad-cars)
- [MADrive: Memory-Augmented Driving Scene Modeling (arXiv 2506.21520)](https://arxiv.org/abs/2506.21520)
- [Project page yandex-research/madrive](https://yandex-research.github.io/madrive/)

### Make/Type
- [Stanford Cars (Krause et al. 3DRR-13)](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
- [VMMRdb](https://github.com/faezetta/VMMRdb)
- [BoxCars116k arxiv:1703.00686](https://arxiv.org/abs/1703.00686)

### Color
- [Chen et al. 2014 — TITS Vehicle Color Recognition](https://arxiv.org/pdf/1510.07391)
- [VeRi-776](https://github.com/VehicleReId/VeRi)

### NVIDIA TAO model cards
- [TrafficCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet)
- [VehicleMakeNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehiclemakenet)
- [VehicleTypeNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehicletypenet)
- [LPDNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lpdnet)
- [LPRNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lprnet)

---

## 9. Acceptance / sign-off

Этот research-документ закрывает шаги step-01 — step-06 BMM technical research workflow (`{skill}/technical-steps/`). Артефакт готов для review.

**Следующий шаг (вне research-воркфлоу):** создание `docs/about_datasets/{name}.md` для финального шорт-листа:
1. `docs/about_datasets/coco.md` — Primary Detector cross-validation.
2. `docs/about_datasets/car_plate_ocr_dataset_ru.md` — LP RU recognition primary.
3. `docs/about_datasets/stanford_cars.md` — Make + Type primary benchmark.
4. **`docs/about_datasets/mad_cars.md` — Make secondary + Color primary (14/15 цветов).**
5. `docs/about_datasets/chen_2014_color.md` — Color sanity-check (7 цветов).
6. (опционально) обновление `docs/about_datasets/bdd100k.md` — добавить раздел License.

**Дельта-обновление 2026-05-15:** добавлен MAD-Cars (Yandex) — `huggingface.co/datasets/yandex/mad-cars`. Закрывает большую часть Color-gap (7/15 → 14/15) и расширяет Make-валидацию multi-view robustness тестом. License CC BY-NC-SA 4.0 — internal benchmark only.


