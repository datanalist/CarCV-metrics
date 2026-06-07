# Аналоги датасета «VMMRdb» для задачи «Make — классификация марки ТС (fine-grained)»

> **См. также:** исходный датасет — [vmmrdb.md](../about_datasets/vmmrdb.md) · индекс датасетов — [datasets.md](../../datasets.md) · сводка исследования — [00_SUMMARY.md](00_SUMMARY.md)

> ⚠️ **CARS — КОММЕРЧЕСКИЙ продукт.** Лицензионный статус каждого кандидата — определяющий критерий. По умолчанию лицензия считается `restricted/unclear`, пока обратное не подтверждено первоисточником. Ни один из найденных академических fine-grained датасетов **не пригоден для обучения продакшн-весов без отдельного лицензионного соглашения и legal review**. Факты ниже перепроверены адверсариально через первоисточники (WebSearch/WebFetch) на 2026-06-05.

## Контекст и требования

**Целевая модель — VehicleMakeNet** (NVIDIA TAO, NGC): классификатор марки ТС, backbone **ResNet-18 (pruned)**, вход **crop ТС 224×224×3**, выход — логиты **20 марок US/EU-рынка** (Acura, Audi, BMW, Chevrolet, Chrysler, Dodge, Ford, GMC, Honda, Hyundai, Infiniti, Jeep, Kia, Lexus, Mazda, Mercedes, Nissan, Subaru, Toyota, Volkswagen). В конвейере CARS это **SGIE1** поверх кропов от TrafficCamNet. Препроцессинг — TAO-classifier: **BGR без свопа в RGB, без `/255`, вычитание offsets `(104, 117, 124)` (B;G;R), NCHW** (см. `docs/about_models/vehiclemakenet.md`).

**Исходный датасет валидации — VMMRdb**: каталожные/краудсорсенные фото с метками `make_model_year`, US-перекос, long-tail. Лицензионная проблема VMMRdb: MIT покрывает репозиторий-обёртку, но **не права на сами краудсорсенные фотографии** — та же оговорка применима почти ко всем аналогам.

**Что требуется от аналога:**
- иерархия меток **make** (минимум) или make/model — для маппинга на 20 NGC-марок;
- покрытие именно **US/EU-словаря** NGC (азиатский/китайский перекос снижает ценность);
- по возможности **bbox** — для чистого crop 224×224 под вход модели;
- доменная близость к бортовому/камерному сценарию (дистанция 5–50 м, угол 0–30°) — для cross-domain валидации против каталожного VMMRdb;
- **для коммерческого обучения — лицензия, явно разрешающая commercial use** (и контроль прав на исходные фото).

> **Примечание про face embedding** (для единообразия со стеком CARS): датасет **WIDER FACE** содержит только bbox лиц **без identity-меток**, поэтому он **не годится для обучения face-эмбеддингов** (нужны пары same/different identity, как в VGGFace2/MS-Celeb/Glint). WIDER FACE применим только к задаче **детекции лиц** (FaceDetect), не к эмбеддингам. Здесь это вне основной задачи Make, но фиксируется как сквозное ограничение стека.

## Обзор аналогов

### CompCars (The Comprehensive Cars)
**2015 / ~208k изобр. (web-nature 136 726 целых ТС + 27 618 частей; surveillance-nature 50 000 фронтальных) / Custom non-commercial research agreement CUHK MMLab (комм.: НЕТ) / Своя иерархия make(163)/model(1716)/year + bbox + viewpoint(5) для web-nature; surveillance: bbox+model+color / Домен: web-каталог + surveillance-камеры**

- **Плюсы:**
  - Крупнейший fine-grained car-датасет с явной иерархией make/model/year — прямой аналог задачи Make; 163 марки покрывают все 20 NGC-классов.
  - **Surveillance-подмножество (50k фронтальных кадров с камер, bbox+model+color)** — лучший доменный срез внутри CompCars, ближе к бортовому/камерному сценарию, чем web-nature и VMMRdb.
  - Viewpoint-метки и bbox в web-nature — удобно отбирать ракурс 0–30° и резать crop 224×224.
- **Минусы:**
  - Лицензия строго **non-commercial** (подписываемое соглашение, case-by-case): «not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes» — обучение продакшн-весов CARS недопустимо без соглашения.
  - Распределение марок смещено к **азиатскому/китайскому рынку** — пересечение именно с US/EU-словарём NGC слабее, чем у VMMRdb/Stanford Cars.
  - Доступ не one-click: agreement + одобрение MMLAB; в surveillance отдельной метки «make» нет — выводится из model-иерархии.
- **Применимость для валидации:** хорошая (cross-dataset контроль VehicleMakeNet на не-VMMRdb домене, особенно surveillance-50k). Маппинг на 20 NGC-марок, исключение OOD, Top-1/Top-3 macro. Только внутренний research-benchmark.
- **Применимость для обучения:** технически отличный материал, но **non-commercial блокирует** коммерческое обучение. Только research/ablation или после соглашения с CUHK.

### Stanford Cars (Cars196)
**2013 / 16 185 изобр., 196 классов (make+model+year), 8 144 train / 8 041 test / Лицензия НЕ определена явно («license unknown», ImageNet-подобная) (комм.: UNCLEAR) / .mat devkit: bbox(4) + class_id; в TFDS — image+label+bbox / Домен: web-каталог (3/4 и боковые ракурсы, цельный ТС крупным планом)**

- **Плюсы:**
  - Канонический fine-grained бенчмарк, **рынок US** — высокое пересечение с 20 US/EU-марками NGC; в репозитории уже отмечено покрытие 20/20 (`docs/about_datasets/stanford_cars.md`).
  - Есть bbox — лёгкий crop 224×224; малый объём, чистые метки, широкая поддержка (torchvision/TFDS/HF) — быстрый sanity-check.
- **Минусы:**
  - Лицензия **«unknown»/ImageNet-подобная** → для коммерческого CARS юридически рискованно; обязателен legal review при любой внешней публикации метрик.
  - **Оригинальный хостинг ai.stanford.edu возвращает 404** (подтверждено pytorch/vision issues #7545, #7670 и `stanford_cars.md`) — только зеркала Kaggle/HF/Academic Torrents с неясной лицензией.
  - Малый объём (16k) и каталожные ракурсы — для серьёзного дообучения мало, доменный сдвиг к бортовой камере, нет night/IR.
- **Применимость для валидации:** хорошая как **быстрый US-домен sanity-check** вне VMMRdb (Top-1/Top-3 macro по 20 маркам). Лёгкий независимый срез против возможного leak VMMRdb.
- **Применимость для обучения:** малопригоден (объём + неясная лицензия). Только research.

### BoxCars116k
**2018 / 116 286 изобр., 27 496 ТС, 693 fine-grained класса (make/model/submodel/year) / Research-only, non-commercial (FIT VUT Brno) (комм.: НЕТ) / Свой JSON: метки make/model/submodel/year + 3D bbox + vehicle_id/instance_id, готовые hard/medium сплиты / Домен: камеры дорожного наблюдения (traffic surveillance), много ракурсов**

- **Плюсы:**
  - **Сильнейший domain-fit из всех кандидатов:** реальная дорожная съёмка с камер, разные ракурсы/дистанции — ближе всего к камерному сценарию CARS (хотя стационарные камеры, не движущийся борт).
  - Большой объём (116k), fine-grained иерархия, **3D bbox** (точный crop 224×224) и готовые сложные сплиты для честной оценки.
- **Минусы:**
  - **Research-only / non-commercial** — обучение продакшн-весов недопустимо без соглашения с FIT VUT Brno.
  - **Европейский рынок** (Чехия) — частичное пересечение с US/EU NGC, много локальных моделей вне 20 классов; маппинг model→make требует ручного справочника.
  - Нет night/IR.
- **Применимость для валидации:** очень хорошая — лучший доменно-близкий контр-срез против каталожного VMMRdb (cross-domain + контроль leak). 3D-bbox → crop, Top-1/Top-3 macro.
- **Применимость для обучения:** отличный материал по домену/объёму, но **research-only блокирует** коммерческое обучение. Только research/ablation.

### Frontal-103
**2020 / 65 433 изобр., 103 марки, 1 759 fine-grained моделей / Custom non-commercial research agreement (XJTU/BIT VMCLab) (комм.: НЕТ) / Метки make/model через структуру каталогов, фронтальный ракурс / Домен: web-nature фронтальные изображения**

- **Плюсы:**
  - Большой fine-grained объём (65k), 103 марки — покрывает 20 NGC-классов плюс расширение.
  - Гарантированно **фронтальный ракурс (~0°)** — попадает в угол CARS 0–30° и постановку «перёд ТС».
- **Минусы:**
  - Строго **non-commercial** («not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes»); изображения собраны из интернета и **не являются собственностью** университетов — двойной риск прав на фото.
  - Преимущественно **китайский рынок** марок — слабее пересечение с US/EU NGC.
  - Доступ через Baidu (неудобно из RU/EU); только фронт (нет 3/4 и rear), web-домен.
- **Применимость для валидации:** пригоден для среза **фронтального сценария Make** (угол ~0°), маппинг на 20 NGC, Top-1/Top-3 macro. Вспомогательный.
- **Применимость для обучения:** объём хорош, но non-commercial + затруднённый доступ. Только research.

### VeRi-776 (и родственный VERI-Wild)
**VeRi-776: 2016 / ~50 000 изобр., 776 ТС, 20 камер / VERI-Wild: 2019 / 416 314 изобр., 40 671 identity, 174 камеры / Research-only по email-соглашению (комм.: НЕТ) / VeRi-776: bbox+type+color+brand (грубый) + camera/vehicle ID; VERI-Wild: ТОЛЬКО vehicle ID + camera (make/model НЕТ) / Домен: городская CCTV-съёмка**

- **Плюсы:**
  - Очень большой объём и реальный камерный домен; VeRi-776 содержит атрибут brand + bbox (можно резать crop).
  - Разнообразие ракурсов/освещения/камер.
- **Минусы:**
  - **Это re-identification датасеты.** В **VERI-Wild make/model/brand-меток нет вообще** (подтверждено: аннотируется только vehicle ID); в VeRi-776 «brand» **грубый, не fine-grained** make/model.
  - Non-commercial; **китайский рынок** — слабое пересечение с 20 US/EU NGC; госномера замаскированы.
- **Применимость для валидации:** **маргинально** — только VeRi-776 brand, и то грубо; пересечение с 20 NGC малое. Не основной Make-benchmark, в лучшем случае доменный стресс-срез камерной съёмки.
- **Применимость для обучения:** **не рекомендуется** — меток make нет/грубые, non-commercial. Датасеты под re-ID, не под Make.

### car_models_3887 / Car Models 3778 (HF Unit293, autoevolution scrape)
**2023–2024 / ~193 000 изобр., 3 778 вариантов модели (make/model/year), 20–200 изобр./вариант, 512×512 / `license: other` на карточке HF (комм.: UNCLEAR/рискованно) / .csv с 44 колонками (make/model/year/body/engine/...) + ImageFolder, без bbox / Домен: студийно-каталожные/прессовые фото autoevolution.com**

- **Плюсы:**
  - Богатые метаданные (44 поля) — гибкая фильтрация по марке; глобальный охват, включая 20 NGC-классов.
  - Большой объём (~193k), фиксированный 512×512 — удобно ресайзить в 224×224.
- **Минусы:**
  - **`license: other` + web-scrape autoevolution.com** → права на фото у единого правообладателя-сайта, **не переданы**; для коммерческого CARS высокий юридический риск (хуже VMMRdb — единый источник). Карточка HF **не содержит** явного разрешения на commercial use.
  - **Студийно-каталожный домен** (чистый фон) — сильный сдвиг относительно бортовой камеры; нет bbox; community-чистота не верифицирована.
- **Применимость для валидации:** только вспомогательный US/EU-срез; из-за студийного домена метрики оптимистичны. Внутренний research-benchmark.
- **Применимость для обучения:** **не рекомендуется** из-за неясной лицензии и scrape-происхождения фото.

### Roboflow Universe «car brands» (community: g3/categorize-car-brands, pm-20vml/car-brands и аналоги)
**2024 / малый-средний объём (сотни–тысячи изобр. на проект) / CC BY 4.0 (подтверждено для ряда проектов) (комм.: ДА при атрибуции — с оговоркой про права на фото) / Classification или detection; экспорт YOLO/COCO/VOC / Домен: смешанный (web + driving POV), варьируется по проектам**

- **Плюсы:**
  - **ЕДИНСТВЕННЫЙ класс кандидатов с явно коммерчески-пригодной лицензией CC BY 4.0** — критично для CARS (обучение/валидация легальны при атрибуции).
  - Бренды у части проектов почти 1:1 совпадают с NGC-словарём: g3/categorize-car-brands (classification, 18 классов: Acura, Audi, BMW, Chevrolet, Ford, Honda, Hyundai, Infiniti, KIA, Lamborghini, Lexus, Mazda, MercedesBenz, Nissan, Porsche, Tesla, Toyota, Volkswagen); pm-20vml/car-brands (detection, схожий набор).
  - One-click доступ без email-соглашений; готовый экспорт под crop.
- **Минусы:**
  - **Малый объём и непроверенное качество разметки** (community) — риск шумных меток, дубликатов, train/val-утечек.
  - **CC BY 4.0 покрывает компиляцию/аннотацию датасета, но НЕ гарантирует прав на исходные фотографии**, если они scrape — для коммерции нужен контроль происхождения изображений.
  - Сильная неоднородность домена; нет night/IR; каждый проект проверять индивидуально (лицензия и состав классов варьируются от проекта к проекту).
- **Применимость для валидации:** пригоден как **юридически безопасный публичный срез** Make на US/EU-марках (отобрать проект с нужными классами и приемлемым качеством, Top-1/Top-3).
- **Применимость для обучения:** **единственный юридически безопасный (CC BY 4.0) кандидат**, НО объём/качество слабые — только как **дополнение/аугментация**, не основной корпус. Обязательны ручная чистка и проверка происхождения фото.

## Сводная таблица аналогов

| Датасет | Объём | Лицензия (комм.?) | Формат аннотаций | Домен-fit | Validation/Training | Вердикт |
|---------|-------|-------------------|------------------|-----------|---------------------|---------|
| **CompCars** | ~208k (surv. 50k) | non-commercial agreement (НЕТ) | make/model/year + bbox + viewpoint | Частичный; surv-50k близок | Validation (research) | secondary |
| **Stanford Cars** | 16 185 / 196 кл. | unknown/ImageNet-like (UNCLEAR) | bbox + class (.mat) | Слабый-средний, рынок US | Validation (sanity-check) | secondary |
| **BoxCars116k** | 116k / 693 кл. | research-only (НЕТ) | make/model/submodel + 3D bbox | **Лучший** (traffic cam) | Validation (research) | secondary |
| **Frontal-103** | 65k / 103 марки | non-commercial (НЕТ) | make/model (каталоги) | Средний (фронт ~0°) | Validation (research) | marginal |
| **VeRi-776 / VERI-Wild** | 50k / 416k | research-only (НЕТ) | re-ID; brand грубый / нет make | Домен ОК, но не Make | Маргинально / не для обуч. | not_recommended |
| **car_models_3887 (HF)** | ~193k / 3 778 | `other` + scrape (UNCLEAR) | csv make/model/year, без bbox | Слабый (студия) | Research-only | marginal |
| **Roboflow «car brands» (CC BY)** | малый-средний | **CC BY 4.0 (ДА*, см. оговорку)** | classification / bbox (YOLO/COCO) | Переменный | **Train (доп.) + Validation** | primary (по лицензии) |

> \* CC BY 4.0 разрешает коммерческое использование при атрибуции, но **не гарантирует прав на исходные фотографии**, если они scrape — требуется проверка происхождения. Confidence по пригодности корпуса — низкая (малый объём, непроверенное качество).

## Рекомендации

**Для валидации VehicleMakeNet (рекомендуется, research-режим — FAIL валиден при доменном сдвиге):**
1. **BoxCars116k** — основной доменно-близкий контр-срез против каталожного VMMRdb (реальные камеры, 3D bbox → crop 224×224). Лучший cross-domain контроль и проверка возможного leak VehicleMakeNet↔VMMRdb.
2. **CompCars surveillance-50k** — второй камерный срез (фронтальные кадры, bbox+model).
3. **Stanford Cars** — быстрый US-домен sanity-check (20/20 марок, лёгкий, есть bbox).
4. Везде: маппинг марок на 20 NGC-классов, исключение OOD (`skipped_oo_dist`), **macro-average** Top-1/Top-3 (long-tail), crop по bbox под 224×224, строгий TAO-препроцессинг (**BGR без свопа, offsets 104/117/124, без `/255`, NCHW**).

**Для обучения/дообучения коммерческих весов:**
- **Единственный юридически безопасный класс — Roboflow Universe «car brands» под CC BY 4.0** (g3/categorize-car-brands и аналоги с US/EU-брендами). Только как **дополнение/аугментация** + собственный сбор; обязательны ручная чистка и проверка происхождения исходных фото.
- Академические датасеты (CompCars, BoxCars116k, Frontal-103) держать **только как research/benchmark**; в продакшн-обучение — **не брать без отдельного лицензионного соглашения и legal review**.
- car_models_3887 и VeRi-776/VERI-Wild для обучения Make — **не использовать** (лицензия/scrape; отсутствие или грубость make-меток).

**Явный приоритет:**
- Валидация → **BoxCars116k** (домен) + **Stanford Cars** (US-домен sanity).
- Обучение → **Roboflow CC BY 4.0** (как дополнение) + собственный сбор.

> По любому кандидату до коммерческого использования — **legal review статуса прав на изображения** (актуально для всех, не только VMMRdb).

## Ссылки

- CompCars (CUHK MMLab): https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ — соглашение: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/agreement.pdf
- Stanford Cars (оригинал, 404): https://ai.stanford.edu/~jkrause/cars/car_dataset.html — зеркало: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset — статус 404: https://github.com/pytorch/vision/issues/7545
- BoxCars116k: https://github.com/JakubSochor/BoxCars — загрузка: https://medusa.fit.vutbr.cz/traffic/data/BoxCars116k.zip
- Frontal-103: https://github.com/vision-insight/Frontal-103
- VERI-Wild: https://github.com/PKU-IMRE/VERI-Wild — VeRi-776: https://github.com/JDAI-CV/VeRidataset
- car_models_3887 (HF Unit293): https://huggingface.co/datasets/Unit293/car_models_3887
- Roboflow «categorize car brands» (CC BY 4.0, classification): https://universe.roboflow.com/g3-32st4/categorize-car-brands — «car-brands» (CC BY 4.0, detection): https://universe.roboflow.com/pm-20vml/car-brands-feb0g
- Локальный контекст: `/home/vallo/CarCV-metrics/docs/about_models/vehiclemakenet.md`, `/home/vallo/CarCV-metrics/docs/about_datasets/vmmrdb.md`, `/home/vallo/CarCV-metrics/docs/about_datasets/stanford_cars.md`

## История изменений

- 2026-06-05 — создан в рамках исследования открытых датасетов-аналогов стека CARS.
