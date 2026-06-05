# Аналоги датасета «BDD100K» для задачи «Detection — детекция транспортных средств и участников движения»

> **См. также:** исходный датасет — [bdd100k.md](../about_datasets/bdd100k.md) · индекс датасетов — [datasets.md](../../datasets.md) · сводка исследования — [00_SUMMARY.md](00_SUMMARY.md)

## Контекст и требования

Эталонный валидационный датасет для Primary Detector CARS — **BDD100K** (driving POV, 100K кадров, классы `car/truck/bus/bike/motor/person/rider/traffic light/traffic sign`). Он под **UC Berkeley custom data license**: свободен для research/not-for-profit, но коммерческое использование требует соглашения с UC Berkeley OTL (`otl@berkeley.edu`). Поэтому ищем аналоги, которые (а) закрывают пробелы по домену и условиям, и (б) по возможности коммерчески пригодны.

**Целевая модель — TrafficCamNet** (NVIDIA TAO, DetectNet_v2 на backbone ResNet18; классы `car / bicycle / person / road_sign`; вход ~960×544; обучена на видео ДОРОЖНЫХ/ТРАФИК-камер — вид сверху/под углом). Целевой сценарий борта CARS — automotive POV с уровня дороги (контроль доступа, парковка, патруль, логистика), day/night/IR, дистанция ~5–50 м. Отсюда два разных «домена-fit»:

- **In-domain TrafficCamNet** (POV стационарной трафик-камеры) — даёт честную оценку PGIE в его training-домене: MIO-TCD, UA-DETRAC.
- **Бортовой/driving POV** (ближе к целевому сценарию CARS) — кросс-валидация под борт: nuImages, KITTI, Waymo, Argoverse 2.

Что требуется от аналога:
- 2D bbox-аннотации, конвертируемые в COCO/YOLO/KITTI;
- покрытие хотя бы части классов `car / person / bicycle` (маппинг GT → классы модели);
- разнообразие условий (day/night/погода) для срезовой оценки;
- по возможности — **коммерчески пригодная лицензия** (CARS — коммерческий продукт; любой NC-датасет годится только для internal/research-валидации, не для поставки/маркетинга метрик и не для дистрибутива дообученных весов без отдельной лицензии).

**Важные ограничения покрытия классов.** Класс `road_sign` (`traffic sign`) НЕ покрыт bbox-аннотациями ни в одном из найденных аналогов (в BDD100K он есть, но в аналогах либо отсутствует, либо уходит в сегментацию фона). Для валидации `road_sign` нужен отдельный traffic-sign датасет (например, Mapillary Traffic Sign — выходит за рамки данного исследования).

> Примечание про face embedding (для смежных моделей стека CARS): WIDER FACE содержит только bbox лиц **без identity-меток**, поэтому он годится для детекции лиц, но НЕ годится для обучения/валидации face-эмбеддингов (нет пар «один человек — много снимков»). К задаче Detection ТС это не относится, но фиксируем для консистентности стека.

---

## Обзор аналогов

### MIO-TCD Localization (MIOvision Traffic Camera Dataset)
**2018 (IEEE TIP, данные ~2017) / Localization: 137,743 кадра с bbox (весь MIO-TCD — свыше 500K аннотированных изображений, 11 классов) / CC BY-NC-SA 4.0 (комм.: НЕТ — подтверждено на tcd.miovision.com) / per-image CSV (класс + bbox), конвертируется в COCO/VOC / стационарные дорожные/трафик-камеры (Канада/США, day/night, разные сезоны)**

Плюсы:
- Лучший доменный матч с TrafficCamNet: вид с дорожных/трафик-камер (а не street-level photo как COCO) → честная in-domain оценка PGIE.
- Большой объём (137K кадров локализации), реальные условия (день/ночь, погода, сезоны), реальный камерный шум/компрессия/малые объекты — стресс-тест, близкий к продакшну.
- Классы покрывают `car / bicycle / pedestrian` — прямое соответствие 3 из 4 классов модели.

Минусы:
- **CC BY-NC-SA 4.0 → коммерческое использование запрещено** (подтверждено первоисточником): только internal benchmark, не для поставки/маркетинга и не для дистрибутива дообученных весов без лицензии Miovision.
- Нет класса `road_sign` — 4-й класс модели не валидируется.
- Аннотации в challenge-формате (не COCO/YOLO «из коробки») — нужна конверсия.
- Стационарная камера ≠ бортовой POV CARS (ракурс ближе к training-домену модели, чем к целевому сценарию борта).

Применимость для валидации: **сильно рекомендуется** как in-domain валидация PGIE (mAP/recall по `car/bicycle/person`). Маппинг GT → модель: `car ← {car, pickup truck, work van, single/articulated truck, bus частично}`, `person ← {pedestrian}`, `bicycle ← {bicycle}`. `road_sign` не покрыт.

Применимость для обучения: пригоден для дообучения под traffic-camera домен (объём + классы), НО только research/internal из-за NC-лицензии.

### nuImages (часть nuScenes)
**2020 / 93,000 аннотированных keyframes, ~800K 2D bbox + instance-маски, 23 класса / CC BY-NC-SA 4.0 (custom NC; комм.: НЕТ — лицензия через nuScenes@motional.com) / собственный JSON devkit (→ COCO) / driving POV (6 камер кругового обзора, Boston/Singapore, urban, day/night/rain)**

Плюсы:
- Большой driving-POV набор реальных 2D bbox (~800K) — релевантен целевому борту CARS.
- Классы vehicle/pedestrian/bicycle/motorcycle покрывают 3 из 4 классов модели.
- Сложные условия (ночь/дождь) релевантны day/night/IR-сценарию продукта; зрелый devkit, стабильный источник.

Минусы:
- **Non-commercial лицензия** → только internal validation/прототип (подтверждено: nuScenes/nuImages под CC BY-NC-SA 4.0, коммерческая лицензия — отдельно через Motional).
- Driving-POV расходится с training-доменом TrafficCamNet (вид сверху) → ожидается частичный domain gap при оценке PGIE.
- Нет `road_sign` как bbox-класса.
- 23 класса — нужна агрегация под 4 класса модели.

Применимость для валидации: рекомендуется как driving-POV кросс-валидация PGIE (дополняет MIO-TCD и BDD100K). Маппинг: `car ← {vehicle.car, vehicle.truck, vehicle.bus}`, `person ← {human.pedestrian.*}`, `bicycle ← {vehicle.bicycle, vehicle.motorcycle}`.

Применимость для обучения: технически пригоден (богатый driving-POV), но NC-лицензия → только research/internal.

### KITTI Object Detection (2D)
**2012 / 7,481 train + 7,518 test изображений, ~80K 2D bbox / CC BY-NC-SA 3.0 (комм.: НЕТ — подтверждено на cvlibs.net) / KITTI txt-формат (→ COCO/VOC/YOLO) / driving POV (стереокамера на авто, Karlsruhe, преимущественно daytime, ясная погода)**

Плюсы:
- Классический, стабильный, хорошо документированный бенчмарк — лёгкий быстрый smoke-test PGIE.
- Классы `Car / Pedestrian / Cyclist` напрямую мапятся на `car / person / bicycle`.
- Чёткие правила оценки (easy/moderate/hard); малый объём удобен для проверки пайплайна.

Минусы:
- **CC BY-NC-SA 3.0 → коммерческое использование запрещено** (подтверждено первоисточником): только internal.
- Только дневная ясная погода — нет night/rain/IR (узкое покрытие условий продукта).
- Небольшой объём (~7.5K) — слаб для обучения и ограниченная статистическая мощность валидации.
- Нет `road_sign` bbox; ограниченное географическое разнообразие (Германия).

Применимость для валидации: быстрый driving-POV sanity-check PGIE (`Car/Pedestrian/Cyclist`). Не заменяет BDD100K/MIO-TCD по объёму и разнообразию условий.

Применимость для обучения: слабо пригоден (мал объём, узкие условия) и запрещён коммерчески. Только research/аугментация.

### Roboflow Universe vehicle-detection наборы (CC BY 4.0, напр. Vehicles-COCO)
**2020–2024 (разные авторы/версии) / варьируется: Vehicles-COCO ~18,998 изображений (bus/car/motorcycle/truck); прочие наборы 1.5K–10K+ / чаще CC BY 4.0, но per-dataset (комм.: ДА — при подтверждении лицензии конкретного набора) / экспорт COCO/YOLO/VOC/TFRecord прямо из Roboflow / смешанный POV (highway/traffic-cam/dashcam/driving — зависит от набора)**

Плюсы:
- **Единственная коммерчески-пригодная категория (CC BY 4.0)** среди близких по домену — критично для продукта CARS.
- Готовый экспорт в COCO/YOLO — минимальная конверсия под эвалуатор.
- Можно подобрать набор с нужным POV (highway/traffic-cam) и классами car/bus/truck/person/bike; годится и для валидации, и для дообучения коммерческой модели (с атрибуцией).

Минусы:
- Качество/полнота аннотаций неоднородны (пользовательская разметка) — нужна ручная QA перед валидацией.
- **Лицензию надо проверять для КАЖДОГО конкретного набора** (не все CC BY 4.0; встречаются неуказанные/ограничительные). При проверке страница Roboflow Universe возвращает 403 для автоматического fetch — лицензию нужно подтверждать вручную в UI на каждый набор.
- Происхождение исходных изображений не всегда прозрачно (риск вторичных прав на фото) — нужен legal-чек источника.
- Обычно нет `road_sign`; таксономии классов разнятся между наборами.

Применимость для валидации: пригоден для коммерчески-чистой валидации PGIE при ручной курации и сверке лицензии; хорошее дополнение к BDD100K там, где нужна commercial-OK выборка.

Применимость для обучения: главный коммерчески-допустимый источник для **дообучения** детектора ТС (CC BY 4.0 + атрибуция). Рекомендуется агрегировать несколько проверенных наборов нужного POV и провести QA разметки.

### UA-DETRAC
**2015 (arXiv) / 2020 (CVIU) / 100 видео, >140K кадров, ~1.21M bbox ТС; классы car, bus, van, others / лицензия оригинала НЕ подтверждена (комм.: UNCLEAR) / 2D bbox (XML, MOT-стиль, → COCO/VOC) / traffic surveillance (стационарные дорожные камеры над магистралями/перекрёстками, Пекин/Тяньцзинь, надземный ракурс)**

Плюсы:
- Доменно близок (traffic-camera, вид под углом сверху) — релевантен training-домену TrafficCamNet.
- Большой объём bbox ТС (~1.21M) и богатые атрибуты (погода/окклюзия) для срезовой оценки.
- Прямой маппинг `car ← {car, van}`, `large ← {bus}` на класс `car`.

Минусы:
- **Доступность первоисточника деградировала:** оригинальный сайт `detrac-db.rit.albany.edu` теперь 301-редиректит на общую страницу лаборатории University at Albany, где НЕТ ни ссылок на скачивание, ни лицензии. Дата-сет распространяется только через сторонние зеркала (Roboflow/Kaggle).
- **Лицензия оригинала НЕ подтверждена** первоисточником → commercial unclear (риск для продукта). CC BY 4.0 на Roboflow-зеркалах — это лицензия редистрибуции копии, НЕ гарантия прав на оригинал.
- Только ТС: нет `person/pedestrian`, нет `bicycle`, нет `road_sign` — покрывает лишь класс `car`.
- Аннотации в MOT/XML — нужна конверсия.

Применимость для валидации: в принципе годится как in-domain car-only валидация PGIE (recall/precision по `car`), но из-за неподтверждённых лицензии и происхождения зеркала — только после legal-проверки и предпочтительно лишь internal.

Применимость для обучения: дообучение `car` под traffic-camera домен технически возможно, но из-за unclear-лицензии и шаткой доступности — не рекомендуется до legal-review.

### Waymo Open Dataset (Perception, 2D camera labels)
**2019 (v1) / расширения 2022 / ~1,950 сегментов по 20 с (~390K кадров с 5 камер), миллионы 2D bbox; 2D-классы vehicle/pedestrian/cyclist / Waymo Dataset License Agreement (proprietary NC; комм.: НЕТ — жёсткий запрет) / собственный TFRecord/proto / driving POV (мульти-камера с авто, США, day/night/dawn)**

Плюсы:
- Очень большой и качественный driving-POV набор 2D bbox в разных условиях; промышленный уровень согласованности разметки.
- Классы vehicle/pedestrian/cyclist прямо мапятся на `car/person/bicycle`.

Минусы:
- **Крайне ограничительная NC-лицензия:** подтверждён явный запрет на использование датасета и любых обученных на нём моделей/весов (i) в эксплуатации/ассисте ТС, (ii) в Production Systems, (iii) в любых primarily commercial целях → **неприменим к коммерческому продукту CARS даже косвенно**; высокий legal-риск.
- Тяжёлый proto/TFRecord-формат и большой объём — высокая стоимость конверсии/хранения.
- Driving-POV ≠ training-домен модели (вид сверху) → частичный domain gap.
- `road_sign` как 2D-bbox отсутствует.

Применимость для валидации: только академический/research-сценарий; для коммерческого продукта CARS не рекомендуется.

Применимость для обучения: **не рекомендуется** — лицензия прямо запрещает применение в контексте эксплуатации ТС. Только академические эксперименты.

### Argoverse 2 Sensor Dataset
**2022/2023 / 1,000 сенсорных сцен (~15 с), 7 ring-камер + 2 стерео, 3D-кубоиды для 26 категорий / данные CC BY-NC-SA 4.0 (комм.: НЕТ), код/API MIT / преимущественно 3D-кубоиды; 2D-боксы — проекцией 3D (не нативная 2D-разметка) / driving POV (6 городов США, day/night/погода)**

Плюсы:
- Современный, разнообразный driving-POV (6 городов, погода/ночь), богатая таксономия (26 классов, включая vehicle/pedestrian/bicyclist).
- Devkit под MIT (инструменты можно использовать свободно); хорош для кросс-доменного анализа PGIE.

Минусы:
- **Данные CC BY-NC-SA 4.0 (NC)** (подтверждено) → неприменим к коммерческому CARS без лицензии.
- Аннотации нативно 3D; 2D-боксы только проекцией → менее точный 2D-GT для mAP детектора.
- Driving-POV ≠ training-домен модели (вид сверху).
- LiDAR/мульти-камера избыточны для чисто 2D-валидации.

Применимость для валидации: только research/internal driving-POV (с проекцией 3D→2D). Менее удобен, чем nuImages (нативные 2D-боксы).

Применимость для обучения: не рекомендуется (NC-данные); 2D-обучение осложнено отсутствием нативной 2D-разметки.

---

## Сводная таблица аналогов

| Датасет | Объём | Лицензия (комм.?) | Формат аннотаций | Домен-fit | Validation / Training | Вердикт |
|---------|-------|-------------------|------------------|-----------|-----------------------|---------|
| **MIO-TCD Localization** | 137,743 кадра с bbox (весь набор >500K), 11 классов | CC BY-NC-SA 4.0 (**нет**) | per-image CSV → COCO/VOC | Очень высокий — POV трафик-камеры = training-домен модели | Validation: да (in-domain) / Training: research-only | **primary (internal)** |
| **Roboflow CC BY 4.0 vehicle-наборы** | варьируется (Vehicles-COCO ~19K), classes per-set | чаще CC BY 4.0, per-set (**да, при сверке**) | COCO/YOLO/VOC экспорт | Переменный (highway/traffic-cam/dashcam) | Validation: да (с QA) / Training: да (коммерч.) | **primary (commercial)** |
| **nuImages** | 93K keyframes, ~800K 2D bbox, 23 класса | CC BY-NC-SA 4.0 / custom NC (**нет**) | JSON devkit → COCO | Высокий для бортового POV | Validation: да / Training: research-only | **secondary (internal)** |
| **KITTI 2D Detection** | ~15K изобр., ~80K bbox | CC BY-NC-SA 3.0 (**нет**) | KITTI txt → COCO/YOLO | Средний (driving POV, только день) | Validation: smoke-test / Training: слабо | **secondary (internal)** |
| **UA-DETRAC** | >140K кадров, ~1.21M bbox ТС | оригинал не подтверждён (**unclear**) | XML/MOT → COCO | Высокий по домену, но только `car` | Validation: car-only (после legal) / Training: не рек. | **marginal (low confidence; первоисточник недоступен)** |
| **Waymo Open** | ~390K кадров, млн 2D bbox | proprietary NC, жёсткий (**нет**) | TFRecord/proto | Средний (driving POV ≠ top-view) | Validation: research-only / Training: не рек. | **not_recommended** |
| **Argoverse 2** | 1,000 сцен, 26 классов (3D) | данные CC BY-NC-SA 4.0, код MIT (**нет**) | 3D-кубоиды; 2D проекцией | Средний (driving POV, 2D-GT шумнее) | Validation: research-only / Training: не рек. | **marginal** |

---

## Рекомендации

Главный лицензионный вывод: почти все «золотые» autonomous-driving/traffic-датасеты — **NON-COMMERCIAL** и пригодны лишь для internal/research-валидации (не для поставки/маркетинга метрик и не для дистрибутива дообученных весов без отдельной лицензии): MIO-TCD, nuImages/nuScenes, KITTI, Waymo, Argoverse 2. Это подтверждено первоисточниками в ходе адверсариальной проверки.

**Для ВАЛИДАЦИИ (PGIE / TrafficCamNet):**
1. **In-domain (приоритет):** **MIO-TCD Localization** — лучший доменный матч (POV трафик-камеры = training-домен модели). Только internal benchmark (NC). Маппинг GT → `car/person/bicycle`; `road_sign` не покрыт.
2. **Бортовой/driving-POV кросс-валидация:** **nuImages** (нативные 2D-боксы, ночь/дождь) и **KITTI** (быстрый sanity-check). Оба NC → только internal.
3. **Коммерчески-чистая выборка:** курированные **Roboflow CC BY 4.0** наборы нужного POV + уже задокументированный **COCO vehicle subset** (CC BY 4.0; но street-level POV → известный большой domain gap, см. `trafficcamnet.md`/`coco.md`).
4. UA-DETRAC — только car-only и только после legal-review (лицензия unclear, первоисточник недоступен); Waymo/Argoverse 2 — не для продукта.

**Для ДООБУЧЕНИЯ коммерческой модели:**
- Единственный допустимый путь — **Roboflow Universe CC BY 4.0** vehicle-наборы (per-set сверка лицензии вручную + QA разметки + атрибуция), агрегировав несколько наборов нужного POV. Дополнительно — COCO (CC BY 4.0).
- Все NC-наборы (MIO-TCD/nuImages/KITTI/Argoverse 2/Waymo) — дообучение **только в research/internal-режиме**; коммерческий дистрибутив весов без отдельной лицензии правообладателя недопустим.

**Явный приоритет:**
- Валидация (in-domain): **MIO-TCD** (internal) → дополнить nuImages/KITTI.
- Валидация + обучение (коммерческие): **Roboflow CC BY 4.0 vehicle-наборы** (с QA и per-set legal-чеком) + COCO.
- Класс `road_sign` не покрывает ни один аналог — нужен отдельный traffic-sign датасет (Mapillary Traffic Sign — отдельное исследование).

Все NC-датасеты до любой внешней публикации/поставки требуют legal review — согласуется с действующей политикой проекта по BDD100K.

---

## Ссылки

- MIO-TCD: https://tcd.miovision.com/
- MIO-TCD challenge: https://tcd.miovision.com/challenge/tswc2017.html
- nuImages: https://www.nuscenes.org/nuimages
- nuScenes Terms of Use (Non-Commercial): https://www.nuscenes.org/terms-of-use
- KITTI Object Detection: https://www.cvlibs.net/datasets/kitti/eval_object.php
- Roboflow Universe Vehicles-COCO: https://universe.roboflow.com/vehicle-mscoco/vehicles-coco
- UA-DETRAC (первоисточник, ныне редиректит): https://detrac-db.rit.albany.edu/ → https://www.albany.edu/cnse/research/computer-vision-machine-learning-lab
- Waymo Open Dataset Terms: https://waymo.com/open/terms/
- Argoverse 2: https://www.argoverse.org/av2.html
- BDD100K (эталон): https://www.bdd100k.com/

---

## История изменений

- 2026-06-05 — создан в рамках исследования открытых датасетов-аналогов стека CARS. Факты по лицензиям и доступности перепроверены через первоисточники: MIO-TCD/KITTI/Argoverse 2 — CC BY-NC-SA (NC, подтверждено), nuImages — custom NC (подтверждено), Waymo — proprietary NC с явным запретом на эксплуатацию/production (подтверждено); UA-DETRAC — первоисточник `detrac-db.rit.albany.edu` недоступен (301 на общую страницу лаборатории), лицензия оригинала не подтверждена → confidence понижен до low.
