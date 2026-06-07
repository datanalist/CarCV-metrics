# Исследование открытых датасетов-аналогов для стека моделей CARS — сводка

> **См. также:** индекс датасетов — [datasets.md](../../datasets.md) · модели — [models.md](../../models.md) · детальные файлы по задачам — в разделе «Детальные файлы» ниже.

## Цель и метод

Для каждого исходного датасета из `datasets.md` подобраны открытые датасеты-аналоги, пригодные для **валидации** и/или **обучения (дообучения)** соответствующих моделей из `models.md`. Источники проверялись по официальным страницам проектов, академическим публикациям (paper), карточкам Hugging Face и репозиториям GitHub. Лицензии анализировались **адверсариально под коммерческий продукт CARS** (бортовая видеоаналитика ТС): любая неопределённость лицензии (`unclear`, отсутствие SPDX, доступ по email-agreement) трактуется как «по умолчанию запрещено до legal review», а пометки `NC` (non-commercial), `ND` (no-derivatives) и `research-only` считаются блокерами для продуктового использования. Объём, доменное соответствие (POV: трафик/надзорная камера, бортовая dashcam, web-каталог) и возможность применения (validation vs training) фиксировались отдельно. Итог по каждой из 8 задач вынесен в детальный файл `docs/dataset_research/*.md`.

## Краткие итоги по задачам

### 1. Detection — детекция ТС и участников движения: BDD100K → TrafficCamNet
**Топ-аналоги:** **MIO-TCD Localization** (137 743 кадра с bbox; POV трафик-камеры = training-домен модели) — лучший in-domain матч, но `CC BY-NC-SA 4.0`, только internal. Коммерчески-чистый путь — **курированные Roboflow Universe `CC BY 4.0` vehicle-наборы + COCO** (валидация И дообучение, с per-set legal-чеком и QA разметки).
**Вердикт:** для in-domain валидации годен (internal); коммерческая поставка метрик и дообучение — только через Roboflow CC BY 4.0 / COCO. Driving-POV кросс-валидация — nuImages/KITTI (internal, NC).
**Детально:** `docs/dataset_research/01_detection_bdd100k.md`

### 2. LP Detection — детекция номерных знаков: autoriaNumberplateDataset-2021 → nomeroff_lpd
**Топ-аналоги:** **CCPD** (>300k, `MIT`, 4 угла в имени файла) — единственный крупный коммерчески-чистый источник с keypoints для предобучения pose-головы; **UC3M-LP** (`ODbL-1.0`, EU/испанские пластины) — лучший EU-валидационный бенчмарк рядом с UA-доменом AUTO.RIA.
**Вердикт:** пригоден и для валидации, и для дообучения (CCPD — основа, UC3M-LP — EU fine-tune; для keypoints из UC3M-LP нужен свой конвертер полигонов). RodoSol-ALPR/UFPR-ALPR технически идеальны (4 угла, бортовой домен), но `non-commercial` → заблокированы.
**Детально:** `docs/dataset_research/02_lpd_autoria.md`

### 3. Face Detection — детекция лиц: WIDER FACE → FaceDetect/FaceNet
**Топ-аналоги:** **DARK FACE** (~50 399 bbox; реальные ночные/low-light сцены) — закрывает ключевой пробел WIDER FACE по освещённости; **Open Images V7 «Human face»** — единственный с коммерческой лицензией аннотаций `CC BY 4.0` и большим объёмом.
**Вердикт:** для NIGHT/low-light валидации — DARK FACE (при сохранении WIDER_val как эталона Easy/Medium/Hard); для дообучения — Open Images V7 при обязательном legal review лицензий самих изображений. Лицензия DARK FACE не объявлена → для обучения только после уточнения.
**Детально:** `docs/dataset_research/03_facedetect_widerface.md`

### 4. OCR — распознавание текста номеров: autoriaNumberplateOcrRu-2021 → nomeroff_ocr
**Топ-аналоги:** **Nomeroff мультирегиональные OCR-пакеты СНГ + Ru-Military-2022** (`CC BY 4.0`) — единственный коммерчески чистый и независимый от RU-обучения held-out без train-leakage; **AY000554/Car_plate_OCR_dataset** (`CC BY 4.0`, 37 775 train-кропов, чистый ГОСТ-алфавит).
**Вердикт:** пригоден для валидации и дообучения (Ru-Military как честный held-out, AY000554 — отличный train, но для валидации вторичен из-за риска leakage). CCPD — только стресс-тест геометрии/латиницы, не для RU-метрик. GLPD/UFPR — заблокированы (NC+ND / academic).
**Детально:** `docs/dataset_research/04_ocr_autoria.md`

### 5. Color — классификация цвета кузова: MAD-Cars (Yandex) → bae_model_f3
**Топ-аналоги:** **VCoR** (~10.5K, 15/15 классов CARS, готовые кропы) — единственный с полным покрытием таксономии, но лицензия `custom 'research and education'` → только internal benchmark до legal review; **DataCluster Labs Vehicle Color** (preview `CC BY 4.0` / full — платная коммерческая лицензия) — единственный коммерчески-допустимый путь к обучению.
**Вердикт:** коммерчески-чистого открытого пути к обучению по сути НЕТ — нужна платная лицензия DataCluster или собственная разметка; VCoR/UFPR-VCR/VeRi-776/MAD-Cars для обучения непригодны (NC/unclear). VCoR годен для internal-валидации.
**Детально:** `docs/dataset_research/05_color_madcars.md`

### 6. Make — классификация марки ТС (fine-grained): VMMRdb → VehicleMakeNet
**Топ-аналоги:** **BoxCars116k** (116k изобр., 693 fine-grained класса, реальные камеры наблюдения, 3D bbox под crop 224×224) — лучший доменный контр-срез для валидации, но `research-only/NC`; **Stanford Cars** (US-рынок, 20/20 пересечение с марками NGC) — быстрый sanity-check; **Roboflow Universe «car brands»** (`CC BY 4.0`) — единственный юридически пригодный для обучения, но слабый объём/качество.
**Вердикт:** валидация — BoxCars116k + Stanford Cars (internal); коммерческое дообучение — только Roboflow CC BY 4.0 как дополнение/аугментация при чистке и проверке прав на исходные фото. VeRi-776/VERI-Wild не имеют пригодных make-меток.
**Детально:** `docs/dataset_research/06_make_vmmrdb.md`

### 7. Type — классификация типа кузова/класса ТС: Stanford Cars → VehicleTypeNet
**Топ-аналоги:** **VTID2** (4 356 изобр., `CC BY 4.0`, нативная метка типа) — единственный коммерчески-чистый; **BIT-Vehicle** (~10k, таксономия почти 1:1 с VehicleTypeNet, готовый маппинг) — лучший для ВНУТРЕННЕЙ валидации, но лицензия `unclear`.
**Вердикт:** коммерчески-чистый путь к обучению по сути отсутствует, кроме VTID2 как добавки (мал); крупные доменно-близкие (MIO-TCD, BoxCars116k, CompCars) — NC/unclear → только internal non-public эксперименты. Метрики на BIT-Vehicle не публиковать до прояснения лицензии.
**Детально:** `docs/dataset_research/07_type_stanford.md`

### 8. Face embedding — эмбеддинги/распознавание лиц: WIDER FACE → ArcFace-подобная модель
**Топ-аналоги:** **LFW** (6 000 пар) — обязательный smoke-стандарт верификации 1:1, плюс срезы **AgeDB-30 / CFP-FP / CALFW / CPLFW** по возрасту и позе; **DigiFace-1M** (синтетика, фиктивные лица) — pretrain без privacy-блокера, опц. усиленный VGGFace2.
**Вердикт:** WIDER FACE идентичностей НЕ содержит → исходный датасет непригоден для этой задачи. Все train-кандидаты (DigiFace-1M, VGGFace2, CASIA-WebFace) — non-commercial → только R&D, не production без отдельного legal review. Glint360K (от отозванного MS-Celeb-1M) — legal/этический блокер.
**Детально:** `docs/dataset_research/08_faceembed_widerface.md`

## Большая сводная таблица всех аналогов

| Задача | Исходный датасет | Целевая модель | Рекомендуемый аналог(и) | Объём | Лицензия (комм.?) | Домен-fit | Validation/Training |
|---|---|---|---|---|---|---|---|
| 1. Detection | BDD100K | TrafficCamNet | MIO-TCD Localization | 137 743 кадра с bbox | CC BY-NC-SA 4.0 (нет) | Очень высокий (POV трафик-камеры) | Validation (internal) |
| 1. Detection | BDD100K | TrafficCamNet | Roboflow CC BY 4.0 vehicle (+COCO) | 1.5K–19K+ per-set | CC BY 4.0 per-set (да, при сверке) | Переменный (highway/dashcam) | Both |
| 1. Detection | BDD100K | TrafficCamNet | nuImages (nuScenes) | 93K keyframes, ~800K bbox | CC BY-NC-SA 4.0 (нет) | Высокий (driving POV, ночь/дождь) | Validation (internal) |
| 1. Detection | BDD100K | TrafficCamNet | KITTI 2D Detection | ~15K изобр., ~80K bbox | CC BY-NC-SA 3.0 (нет) | Средний (driving, день/ясно) | Validation (smoke) |
| 2. LP Detection | autoriaNumberplateDataset-2021 | nomeroff_lpd | CCPD | >300k (+~12k Green) | MIT (да) | Средний (CN, парковки; Rotate/Tilt/Blur) | Both |
| 2. LP Detection | autoriaNumberplateDataset-2021 | nomeroff_lpd | UC3M-LP | ~1975 изобр. (2547 пластин) | ODbL-1.0 (да, share-alike) | Высокий (EU/испанские, дорожные сцены) | Validation |
| 2. LP Detection | autoriaNumberplateDataset-2021 | nomeroff_lpd | Roboflow rxg4e | ~10 125 изобр. | CC BY 4.0 (да, атрибуция) | Низкий-средний (микс стран, bbox) | Training |
| 3. Face Detection | WIDER FACE | FaceDetect/FaceNet | DARK FACE (UG2+) | 6 000 изобр., ~50 399 bbox | Не объявлена (unclear) | Высокий по свету (ночь/low-light) | Validation |
| 3. Face Detection | WIDER FACE | FaceDetect/FaceNet | Open Images V7 «Human face» | ~9M изобр., ~16M bbox (face — не подтв.) | Аннотации CC BY 4.0 (да); фото unclear | Средний (web, нет automotive/IR) | Training |
| 3. Face Detection | WIDER FACE | FaceDetect/FaceNet | UFDD | 6 424 изобр., 10 895 лиц | Не объявлена (unclear) | Средний-высокий (погод./оптич. деградации) | Validation |
| 3. Face Detection | WIDER FACE | FaceDetect/FaceNet | Face Mask Detection (andrewmvd) | 853 изобр., ~4 072 объекта | CC0 1.0 (да) | Низкий-средний (маски, окклюзия) | Training |
| 4. OCR | autoriaNumberplateOcrRu-2021 | nomeroff_ocr | Nomeroff мультирегион. OCR + Ru-Military | RU-эталон 57 120 кропов; регионы тыс.–десятки тыс. | CC BY 4.0 (да) | Средний (формат=RU, кропы; СНГ-кириллица; Ru-Military held-out) | Both |
| 4. OCR | autoriaNumberplateOcrRu-2021 | nomeroff_ocr | AY000554/Car_plate_OCR_dataset (HF) | 45 514 кропов (train 37 775) | CC BY 4.0 (да) | Средний (ГОСТ-алфавит 22/22, кропы) | Training |
| 4. OCR | autoriaNumberplateOcrRu-2021 | nomeroff_ocr | CCPD / CCPD2020 | 300 000+ кадров | MIT (да) | Низкий-средний (CN+латиница, геометрия) | Validation (стресс) |
| 5. Color | MAD-Cars (Yandex) | bae_model_f3 | VCoR | ~10.5K, 15 классов | custom 'research/education' (unclear → нет) | Средний (web/mixed), но 15/15 классов | Validation (internal) |
| 5. Color | MAD-Cars (Yandex) | bae_model_f3 | DataCluster Labs Vehicle Color | preview ~6K; full больше | preview CC BY 4.0 / full платная (да) | Средний (real-world bbox; покрытие классов не подтв.) | Both (full — платно) |
| 5. Color | MAD-Cars (Yandex) | bae_model_f3 | UFPR-VCR | 10 039 изобр., 11 классов | academic NC (нет) | Высокий (surveillance, ночь/окклюзии; 11/15) | Validation (academic) |
| 6. Make | VMMRdb | VehicleMakeNet | BoxCars116k | 116 286 изобр., 693 класса | Research-only/NC (нет) | Лучший (камеры наблюдения, 3D bbox, EU) | Validation |
| 6. Make | VMMRdb | VehicleMakeNet | Stanford Cars (Cars196) | 16 185 изобр., 196 классов | Unknown/ImageNet-like (unclear) | Слабый-средний (web-каталог), US 20/20 | Validation (sanity) |
| 6. Make | VMMRdb | VehicleMakeNet | Roboflow Universe «car brands» | сотни–тысячи на проект, ~18 классов | CC BY 4.0 (да, атрибуция) | Переменный (web+driving), бренды ~1:1 NGC | Both |
| 6. Make | VMMRdb | VehicleMakeNet | CompCars (surveillance-50k) | ~208k всего; 50K surveillance | Custom NC research (нет) | Частичный (камерный срез; рынок CN) | Validation |
| 7. Type | Stanford Cars | VehicleTypeNet | VTID2 | 4 356 изобр., 5 классов | CC BY 4.0 (да) | Средний-высокий (дорожные, нативный тип; pickup→truck) | Validation |
| 7. Type | Stanford Cars | VehicleTypeNet | BIT-Vehicle | 9 850 изобр., 6 классов | Не указана (unclear) | Высокий (таксономия ~1:1, есть largevehicle) | Validation (internal) |
| 7. Type | Stanford Cars | VehicleTypeNet | MIO-TCD Classification | 648 959 кропов, 11 классов | CC BY-NC-SA 4.0 (нет) | Очень высокий (надзор, день/ночь; car свёрнут) | Validation (internal) |
| 7. Type | Stanford Cars | VehicleTypeNet | BoxCars116k (body-split) | 116 286 изобр., body-type | research-only (unclear) | Высокий (надзор, нативный body-split; парк CZ) | Both (после legal) |
| 8. Face embedding | WIDER FACE (без identity) | ArcFace-подобная | LFW | 13 233 img / 5 749 id; 6 000 пар | research (нет) | Низкий (web, фронт., нет IR/night) | Validation |
| 8. Face embedding | WIDER FACE (без identity) | ArcFace-подобная | AgeDB-30 / CFP-FP / CALFW / CPLFW | по 3–7 тыс. пар | research/academic (unclear) | Низкий-средний (срезы возраст/поза) | Validation |
| 8. Face embedding | WIDER FACE (без identity) | ArcFace-подобная | DigiFace-1M (synthetic) | 1.22M img / 110k id | R-UDA NC (нет); без реальных персон | Низкий, но без privacy-блокера | Training (R&D) |
| 8. Face embedding | WIDER FACE (без identity) | ArcFace-подобная | VGGFace2 | ~3.31M img / 9 131 id | CC BY-NC 4.0 (нет) | Низкий-средний (web-селебрити) | Training (R&D) |

## Кросс-выводы

**Аналоги/семейства, покрывающие несколько задач:**
- **CCPD** (`MIT`) работает сразу на LP Detection (предобучение keypoint-головы, 4 угла) и на OCR (стресс-тест геометрии/латиницы) — единственное крупное коммерчески-чистое LP-семейство.
- **BoxCars116k** покрывает Make (валидация, контр-срез) и Type (нативный body-split) — лучший доменный матч в обеих, но `non-commercial/unclear` → только internal.
- **MIO-TCD** даёт два сабсета: Localization для Detection и Classification для Type — оба `CC BY-NC-SA 4.0`, in-domain, но NC.
- **Roboflow Universe `CC BY 4.0`** — сквозной коммерчески-пригодный путь для Detection, Make (и потенциально других классификаторов): единственное семейство, где возможно и обучение, но ценой ручной per-set сверки лицензий и QA разметки.
- **WIDER FACE** формально исходник для двух задач (Face Detection и Face embedding), но для эмбеддингов непригоден (нет идентичностей) — это разные семейства аналогов.

**Где открытых данных для домена CARS (бортовой/трафик-камера POV) мало:**
- **Color** и **Type** — почти нет открытых наборов одновременно с (а) бортовым/надзорным POV, (б) полной таксономией CARS и (в) коммерческой лицензией; коммерчески-чистые VTID2/DataCluster-preview малы.
- **Make** — fine-grained марки в камерном POV почти все NC (BoxCars116k, CompCars, VeRi).
- **Face embedding** — открытых коммерческих train-наборов нет в принципе; добавить IR/night-домен бортовой камеры нечем.

**Лицензионные риски для коммерческого продукта:**
- Большинство лучших по домену наборов — `CC BY-NC-SA`, `non-commercial`, `academic email-agreement` или `unclear` → блокируют продуктовое использование и часто даже публикацию метрик.
- `CC BY 4.0` на **аннотации** (Open Images V7, Roboflow) НЕ гарантирует прав на исходные **изображения** (web-scrape) — нужен per-image/per-set legal review.
- `ODbL` (UC3M-LP) добавляет share-alike на производную базу данных; `ND` (GLPD) запрещает дообучение и распространение производной модели.

**Что годится для дообучения vs только для валидации:**
- **Дообучение (коммерчески-чистое):** CCPD (`MIT`), Nomeroff CC BY 4.0 + AY000554, Open Images V7 «Human face» (с legal review фото), Roboflow CC BY 4.0 (Detection/Make), Face Mask CC0, DataCluster full (платно). DigiFace-1M/VGGFace2 — только R&D-дообучение.
- **Только валидация (internal, NC/unclear):** MIO-TCD, nuImages, KITTI, BoxCars116k, CompCars, VCoR, UFPR-VCR, BIT-Vehicle, DARK FACE, LFW и срезы (AgeDB/CFP/CALFW/CPLFW).

## Пробелы и риски

- **Color / Type:** нет открытого набора, сочетающего коммерческую лицензию + бортовой POV + полную таксономию CARS; коммерческий путь к обучению — только платный DataCluster или собственная разметка.
- **Make (fine-grained, камерный POV):** лучшие наборы (BoxCars116k, CompCars) `non-commercial`; коммерчески-чистый Roboflow слаб по объёму и качеству.
- **Face embedding:** открытых **коммерческих** train-наборов нет; WIDER FACE не содержит идентичностей и для этой задачи непригоден.
- **Домен-сдвиг web/объявления/каталог → бортовой POV:** Stanford Cars, car_models_3887, Open Images, VCoR — студийные/web-фото, отличаются ракурсом и условиями от бортовой/трафик-камеры → риск переоценки качества на валидации.
- **Identity-датасеты лиц и приватность:** все крупные train-наборы лиц несут privacy-риск; синтетический DigiFace-1M снимает privacy, но добавляет sim-to-real gap.
- **Устаревшие/отозванные:** Glint360K производен от **отозванного MS-Celeb-1M** (legal/этический блокер); официальные ссылки VGGFace2 удалены; оригиналы Stanford Cars (ai.stanford.edu) и UA-DETRAC недоступны (404/301) → проблемы воспроизводимости и подтверждения лицензии.
- **Неподтверждённые/`unclear` лицензии:** DARK FACE, UFDD, MAFA, BIT-Vehicle, VCoR, VehicleX, Stanford Cars, UA-DETRAC, Kaggle nomeroff-russian — нельзя использовать (и часто публиковать метрики) до legal review.
- **Train-leakage в OCR:** AY000554 и Kaggle nomeroff пересекаются с источником обучения RU-модели → как валидация дают завышенные метрики; честный held-out — Nomeroff Ru-Military.
- **NIGHT/IR-домен:** для Face Detection есть только DARK FACE (low-light, НЕ истинный IR); истинного IR-набора лиц под бортовую камеру среди открытых нет.

## Детальные файлы

- `docs/dataset_research/01_detection_bdd100k.md` — Detection: BDD100K → TrafficCamNet (MIO-TCD, Roboflow CC BY 4.0, nuImages, KITTI, UA-DETRAC, Waymo).
- `docs/dataset_research/02_lpd_autoria.md` — LP Detection: AUTO.RIA → nomeroff_lpd (CCPD, UC3M-LP, Roboflow rxg4e, Open Images, RodoSol, UFPR).
- `docs/dataset_research/03_facedetect_widerface.md` — Face Detection: WIDER FACE → FaceDetect (DARK FACE, Open Images «Human face», UFDD, Face Mask CC0, FDDB, MAFA).
- `docs/dataset_research/04_ocr_autoria.md` — OCR: AUTO.RIA OcrRu → nomeroff_ocr (Nomeroff мультирегион + Ru-Military, AY000554, CCPD, Kaggle, GLPD, UFPR).
- `docs/dataset_research/05_color_madcars.md` — Color: MAD-Cars → bae_model_f3 (VCoR, UFPR-VCR, DataCluster, Chen 2014, VeRi-776, VehicleX).
- `docs/dataset_research/06_make_vmmrdb.md` — Make: VMMRdb → VehicleMakeNet (BoxCars116k, Stanford Cars, CompCars, Roboflow «car brands», VeRi, car_models_3887).
- `docs/dataset_research/07_type_stanford.md` — Type: Stanford Cars → VehicleTypeNet (VTID2, BIT-Vehicle, MIO-TCD Classification, CompCars, BoxCars116k, UA-DETRAC).
- `docs/dataset_research/08_faceembed_widerface.md` — Face embedding: WIDER FACE (без identity) → ArcFace-подобная (LFW, AgeDB/CFP/CALFW/CPLFW, VGGFace2, DigiFace-1M, CASIA-WebFace, Glint360K).

## История изменений

- 2026-06-05 — сводка по исследованию аналогов
