# Аналоги модели «TrafficCamNet» для задачи «Detection — детекция транспортных средств и участников движения»

> **См. также:** спецификация исходной модели — [trafficcamnet.md](../about_models/trafficcamnet.md) · реестр моделей — [models.md](../../models.md) · исследование датасетов-аналогов (та же задача) — [01_detection_bdd100k.md](../dataset_research/01_detection_bdd100k.md) · сводка исследования моделей — [00_SUMMARY.md](00_SUMMARY.md)

> Документ подготовлен в рамках исследования открытых моделей-аналогов стека CARS.
> **CARS — коммерческий продукт.** Лицензионный статус каждого кандидата трактуется консервативно: пока коммерческое использование не подтверждено первоисточником явно, оно считается **запрещённым/неясным (restricted/unclear)** и помечается как блокер. Любое включение модели/весов в прод-сборку или дистрибуцию TensorRT-engine требует отдельного legal review.

---

## Контекст и требования

**Целевая модель — TrafficCamNet** (NVIDIA TAO, DetectNet_v2 с backbone ResNet-18 pruned). В пайплайне CARS она работает как **PGIE — первичный детектор** в DeepStream-конвейере: на вход поступает кадр **960×544×3**, на выходе — боксы по **4 классам** `{car, bicycle, person, road_sign}`, которые далее режутся на кропы и уходят во вторичные классификаторы (SGIE) и в ветки распознавания номеров/лиц. Модель компактна (ONNX ~5.36 MB), поставляется с INT8-калибровочным кэшем и нативно ложится в путь ONNX→TensorRT/DeepStream. Препроцессинг специфичен: вход 960×544 RGB, NCHW, нормализация `/255`, offsets 0; декодер DetectNet_v2 — grid-based, anchor-free (coverage 60×34 + bbox 60×34×16).

Рантайм — бортовой: **Jetson Orin Nano 8GB**, исполнение через **DeepStream/TensorRT (FP16/INT8)** либо ONNX Runtime. PGIE запускается на **каждом кадре**, поэтому его латентность и объём памяти задают потолок для всего пайплайна (после PGIE на том же чипе работают SGIE-классификаторы и ветки LPD/OCR/Face). Это жёстко ограничивает бюджет детектора: ResNet-18-класс DetectNet_v2 (единицы M параметров) — практический ориентир, а тяжёлые DETR-бэкбоны и x-варианты YOLO требуют обязательного INT8 и реального замера FPS на устройстве.

Целевой сценарий — **бортовой POV** (камера в движущемся авто, вид с уровня дороги, дистанции 5–50 м: контроль доступа, парковка, патруль, логистика) с режимами **day/night/IR** и регионом **RU/UA**. Ключевая проблема инкумбента зафиксирована замером: TrafficCamNet обучен на **виде сверху с трафик-камер**, а не на бортовом POV, из-за чего на COCO-валидации он даёт **FAIL** — это артефакт **доменного разрыва (вид-сверху vs бортовой POV)**, а не дефект архитектуры. Класс детекции и боксы к региону почти не привязаны, но ракурс, ночь/ИК-подсветка и СНГ-специфика сцен формируют domain gap ко всем обучающим доменам аналогов. Наконец, **коммерческая лицензия — жёсткое требование**: EULA без явного разрешения, AGPL/copyleft, non-commercial веса — всё это блокеры.

**Что требуется от аналога:**
1. **Детекция участников движения** — минимум классы ТС + person (+ велосипед/мотоцикл, road_sign по возможности), чтобы кормить PGIE→SGIE-кропы без потери покрытия.
2. **Коммерчески-чистая лицензия на код И веса** (Apache-2.0/MIT/BSD); NC/AGPL/EULA/отсутствие лицензии — блокеры.
3. **Edge-бюджет под Orin Nano 8GB** — лёгкий вариант (Nano/S/Tiny/R18-класс) с реальным путём ONNX→TensorRT (FP16/INT8) и замеренным FPS.
4. **Нативный ONNX или штатный экспорт в ONNX** — для интеграции в DeepStream/TensorRT.
5. Доменная близость к бортовому/dashcam POV (day/night/IR, RU/UA) — либо пригодность как база для дообучения на собственных данных.

> **Важно по интеграции и измерениям.** Любая не-DetectNet_v2 модель (DETR-семейство, YOLO, SSD, NanoDet) приносит **другой постпроцессинг** — DeepStream-парсер и декодер придётся переписывать, нельзя просто подменить `.onnx`, сохранив grid-декодер TrafficCamNet. Точность всех аналогов ниже приведена **по заявлению авторов (COCO mAP) и нами не измерена**; латентность DETR-семейства на Orin Nano 8GB **требует реального замера** (цифры авторов получены на T4/4090, которые мощнее Orin Nano).

---

## Обзор аналогов

### DashCamNet (NVIDIA TAO)
**актуальный TAO / DetectNet_v2 (grid-based, anchor-free), backbone ResNet-18 / единицы-десятки MB (сопоставимо с TrafficCamNet ~5.4 MB ONNX; точное число параметров на карточке не приведено) / Лицензия: КОД Apache-2.0, ВЕСА — NVIDIA Model EULA (комм.: UNCLEAR — блокер до legal review) / Форматы: .etlt→ONNX (TAO export)→TensorRT FP16/INT8 / Лучший domain-fit подборки (dashcam POV)**

Та же архитектура, что у инкумбента — DetectNet_v2 ResNet-18 (подтверждено официальной TAO-документацией: «based on NVIDIA DetectNet_v2 detector with ResNet 18 as a feature extractor»; спор ResNet-18 vs ResNet-34 разрешён в пользу **ResNet-18**). Интерфейс совпадает с TrafficCamNet: вход 960×544×3 RGB, NCHW, тот же препроцессинг (`/255`, offsets 0); выход DetectNet_v2 (coverage 60×34×4 + bbox 60×34×16). Классы (подтверждено TAO-докой): `car, person, road_sign, two-wheeler` (4).

Плюсы:
- **Идеальный viewpoint-fit:** dashcam POV совпадает с бортовым сценарием CARS (5–50 м, уровень дороги) — закрывает главный провал TrafficCamNet (вид-сверху).
- **Drop-in замена в текущем стеке:** тот же DetectNet_v2-декодер, тот же вход 960×544 и препроцессинг, тот же путь ONNX→TensorRT INT8 — нулевая стоимость переписывания парсера.
- Те же 4 класса (`car/person/road_sign/two-wheeler`) обслуживают PGIE→SGIE-кропы напрямую.

Минусы:
- **ЛИЦЕНЗИОННЫЙ БЛОКЕР (подтверждён адверсариально по форуму NVIDIA, модератор Morganh, тред 281977):** исходные предобученные веса под **NVIDIA Model EULA** **нельзя редистрибутировать** в составе продукта — запрещено до legal review (та же проблема, что у TrafficCamNet). Точный текст Model EULA с NGC-карточки **не вычитан** (JS-рендеринг) → legal review.
- Коммерчески оправдан **только как СТАРТ для дообучения**: производный (дообученный) деплой по EULA коммерчески чист, но это требует прямой вычитки текста legal-специалистом.
- Точность на бортовом POV **нами не измерена**; число параметров на карточке не приведено. RU/UA-специфика не покрыта (детектор общего класса) — дообучение под СНГ всё равно желательно.

Применимость в пайплайне CARS: PGIE — прямая замена/дополнение TrafficCamNet с лучшим domain-fit, **либо** база для fine-tune под RU/UA бортовой POV (после чего производный деплой коммерчески чист по EULA).
Edge / Jetson Orin Nano: отличная — DetectNet_v2 ResNet-18 целевым образом оптимизирован под Jetson, INT8-калибровка штатная, латентность единицы мс (**не измерено нами на устройстве**). Прямой drop-in в существующий DeepStream-пайплайн без переписывания декодера.

### D-FINE (N/S/M/L/X)
**2024 / DETR-подобный real-time детектор (развитие RT-DETR, FDR + GO-LSD, NMS-free) / N:4M, S:10M, M:19M, L:31M, X:62M / Лицензия: Apache-2.0 на КОД И ВЕСА (комм.: ДА — проверено вычиткой LICENSE) / Форматы: PyTorch, штатный ONNX (NMS-free) + TensorRT / Edge-fit N/S реалистичен, нужен замер FPS на Orin Nano**

DETR-подобный детектор, переопределяющий регрессию боксов как fine-grained distribution refinement; NMS-free transformer-decoder. Вход обычно 640×640 RGB (DETR-препроцессинг: `/255` + ImageNet-нормализация); выход — набор query-боксов (logits + boxes) без NMS; классы — 80 COCO (нужен реткласс/fine-tune под 4 класса CARS). Предобученные веса — COCO2017 и Objects365+COCO.

Плюсы:
- **Чистая Apache-2.0 на КОД И ВЕСА** (подтверждено прямой вычиткой LICENSE-файла «Apache License Version 2.0» и README: «all code, weights released under Apache 2.0, suitable for commercial use») — коммерчески безопасно без оговорок (в отличие от TrafficCamNet EULA, Ultralytics AGPL, YOLO-NAS NC).
- SOTA точность/скорость среди real-time DETR (по заявлению авторов), штатный ONNX→TensorRT, NMS-free упрощает граф и постпроцессинг.
- Линейка N/S под edge + Objects365-претрейн как хорошая отправная точка для дообучения под бортовой POV.

Минусы:
- **Доменный разрыв:** COCO/Objects365 (web/смешанный POV) ≠ бортовой POV CARS — без fine-tune точность на сцене CARS низкая; RU/UA не покрыт.
- **DETR-декодер заметно тяжелее DetectNet_v2** по латентности на Orin Nano; нужен реальный замер FPS/памяти в INT8 (**цифры авторов на T4/4090, нами не измерено**).
- Нужен реткласс/дообучение под 4 класса CARS и интеграция нового (не-DetectNet_v2) постпроцессинга в DeepStream.

Применимость в пайплайне CARS: PGIE-кандидат №1 среди open-weights — дообучить S/N под бортовой POV + RU/UA и 4 класса, экспортировать ONNX→TensorRT INT8; коммерчески чистая альтернатива TrafficCamNet.
Edge / Jetson Orin Nano: N (4M) и S (10M) реалистичны в FP16/INT8; DETR-декодер тяжелее CNN, но N/S — целевые real-time варианты. **FPS/память на Orin Nano не измерены.**

### RF-DETR (Nano / Small / Medium / Large)
**2025 / Real-time DETR на базе DINOv2-backbone (детекция + сегментация, NMS-free) / Core Nano→Large (Large ~129M по блогу Roboflow); Nano целевой под edge / Лицензия: Core Nano→Large + весь КОД — Apache-2.0 (комм.: ДА); XL/2XL (rfdetr[plus]) — PML 1.0 проприетарно (комм.: НЕТ) / Форматы: PyTorch, model.export('onnx')→trtexec TensorRT / Edge-fit Nano/Small реалистичен, DINOv2 проверять по памяти**

Real-time DETR, спроектированный под fine-tuning. Вход квадратный (например 560×560/640×640) RGB, DETR-препроцессинг; выход — query-боксы без NMS; классы COCO по умолчанию (нужен fine-tune под CARS).

Плюсы:
- **Core Nano→Large под чистый Apache-2.0** (код И веса; LICENSE репо + блог Roboflow: «core models (Nano through Large) and all code under Apache 2.0»; COCO-pretrained чекпойнты тоже Apache-2.0) — коммерчески безопасно.
- Спроектирован под fine-tuning + удобный экспорт ONNX→TensorRT; быстрый real-time (по заявлению авторов ~100 FPS на T4 FP16 для Nano).
- Активно поддерживается Roboflow, хорошая документация и tooling для дообучения.

Минусы:
- **ЛОВУШКА ЛИЦЕНЗИИ (подтверждена адверсариально):** НЕ путать Core (Apache-2.0) с **XLarge/2XLarge (пакет rfdetr[plus], лицензия PML 1.0, проприетарно, требует Roboflow-аккаунта)** — для продукта брать **ТОЛЬКО Nano→Large**.
- **Доменный разрыв** COCO→бортовой POV: без fine-tune точность на сцене CARS низкая; RU/UA не покрыт.
- DINOv2/DETR-стек тяжелее DetectNet_v2; нужен замер FPS/памяти на Orin Nano (**не измерено**) и интеграция нового постпроцессинга.

Применимость в пайплайне CARS: PGIE-кандидат — fine-tune RF-DETR-Nano/Small под бортовой POV + 4 класса, ONNX→TensorRT; коммерчески чистая альтернатива (только Core-варианты).
Edge / Jetson Orin Nano: Nano целевым образом под edge/real-time; на Orin Nano 8GB реалистичны Nano/Small в FP16/INT8, но DINOv2-backbone может быть тяжеловат — **проверять память на устройстве, не измерено**.

### PP-YOLOE+ (s / m / l / x)
**2022 / Anchor-free YOLO (backbone CSPRepResNet, neck PAN, голова ET-head, label assignment TAL) / s/m/l/x; l ~51.4 mAP COCO@640 (по авторам); x ~100M (тяжёлый); s/m — edge / Лицензия: Apache-2.0 на КОД И ВЕСА (комм.: ДА) / Форматы: Paddle→ONNX (paddle2onnx)→TensorRT FP16 / Edge-fit s/m реалистичен; экосистема Paddle — лишний шаг конвертации**

Зрелое индустриальное anchor-free YOLO-семейство (PaddleDetection), претрейн на Objects365. Вход 640×640 RGB; выход — боксы + class scores (anchor-free), декодирование штатное в PaddleDetection; классы COCO по умолчанию (нужен fine-tune под 4 класса CARS).

Плюсы:
- **Чистая Apache-2.0 на КОД И ВЕСА** (LICENSE репо PaddleDetection Apache-2.0; веса в релизах того же репо) — коммерчески безопасно.
- Зрелая индустриальная линейка со штатным ONNX→TensorRT и хорошей точностью/скоростью (по заявлению авторов; PP-YOLOE+_l ~149 FPS на спец. железе).
- Objects365-претрейн как сильная отправная точка для дообучения под сцену CARS.

Минусы:
- **Экосистема Paddle (не PyTorch)** — дополнительный шаг конвертации paddle2onnx и менее привычный tooling для команды на PyTorch.
- **Доменный разрыв** COCO→бортовой POV, RU/UA не покрыт; без fine-tune точность на сцене CARS низкая.
- Вариант x (~100M) слишком тяжёл для Orin Nano; для edge только s/m.

Применимость в пайплайне CARS: PGIE-кандидат — fine-tune PP-YOLOE+_s/m под бортовой POV + 4 класса, paddle2onnx→TensorRT; коммерчески чистая альтернатива TrafficCamNet.
Edge / Jetson Orin Nano: s/m реалистичны в FP16/INT8; x — слишком тяжёлый. Экспорт Paddle→ONNX добавляет шаг конвертации vs нативный PyTorch, но рабочий. **FPS на устройстве не измерены.**

### YOLOX (Nano / Tiny / S / M / L / X)
**2021 / Anchor-free YOLO (decoupled head, SimOTA, backbone CSPDarknet) / Nano:0.91M, Tiny ~5M, S ~9M, M ~25M, L ~54M, X ~99M / Лицензия: Apache-2.0 на КОД И ВЕСА (комм.: ДА — LICENSE вычитан) / Форматы: PyTorch, штатный ONNX/TensorRT/ncnn/OpenVINO/MegEngine / Edge-fit Nano/Tiny отличный — один из самых лёгких open-детекторов**

Зрелый anchor-free YOLO. Вход 416×416 (Nano/Tiny) или 640×640 (S+) RGB; выход — боксы + class scores (anchor-free), нужен NMS; классы COCO по умолчанию (нужен fine-tune под CARS).

Плюсы:
- **Чистая Apache-2.0** (код И веса; LICENSE вычитан «Apache License Version 2.0»; веса в релизах того же репо, отдельной NC-лицензии на веса нет), очень зрелый и широко развёрнутый стек экспорта (ONNX/TensorRT/ncnn/OpenVINO).
- **YOLOX-Nano (0.91M)/Tiny — крайне лёгкие, идеальны под бюджет Orin Nano 8GB**, оставляют ресурс под SGIE/LPD/OCR.
- Простая интеграция, много готовых примеров деплоя на Jetson.

Минусы:
- Архитектура **2021 года** — точность/скорость ниже современных D-FINE/RF-DETR/RT-DETRv2 при сопоставимых параметрах.
- **Доменный разрыв** COCO→бортовой POV, RU/UA не покрыт; нужен fine-tune под 4 класса CARS.
- Anchor-free, но **с NMS** — лишний постпроцессинг vs NMS-free DETR-семейство.

> Примечание по лицензии: открытый issue #1865 — лишь **неотвеченный пользовательский вопрос** о бандлинге весов, **НЕ ограничение** (Apache-2.0 явно разрешает коммерцию).

Применимость в пайплайне CARS: PGIE-кандидат для самого жёсткого edge-бюджета — fine-tune YOLOX-Nano/Tiny под бортовой POV CARS, ONNX→TensorRT INT8; коммерчески чистый.
Edge / Jetson Orin Nano: Nano (0.91M) и Tiny — отлично (очень малый бюджет памяти/латентности), INT8/FP16. **Конкретный FPS на устройстве не измерен.**

### RT-DETRv2 / RT-DETR (R18 / R34 / R50) + HuggingFace PekingU
**2023–2024 / Real-time DETR (NMS-free transformer-decoder, backbone ResNet R18/R34/R50/R101 + hybrid encoder) / R18:20M, R34:31M, R50:42M, R101:76M; v2-S:20M (~48.1 mAP), v2-L:42M (~53.4 mAP) по авторам / Лицензия: Apache-2.0 на КОД И ВЕСА (комм.: ДА) / Форматы: PyTorch + HF transformers, готовый ONNX (onnx-community)→TensorRT/OpenVINO / Edge-fit R18 пограничен на Orin Nano — нужен замер латентности**

Real-time DETR с ResNet-бэкбоном, интегрирован в HF transformers (`RTDetrForObjectDetection`). Вход 640×640 RGB (DETR-препроцессинг); выход — query-боксы без NMS; классы COCO (нужен fine-tune под 4 класса CARS). Веса — COCO и Objects365+COCO.

Плюсы:
- **Чистая Apache-2.0** (код И веса; lyuwenyu/RT-DETR LICENSE Apache-2.0; HF PekingU/rtdetr_r18vd|r50vd тоже Apache-2.0), доступен и в HF transformers (простой fine-tune/экспорт), и в оригинальном репо.
- NMS-free → упрощённый ONNX-граф и постпроцессинг; SOTA real-time точность (по заявлению авторов).
- Готовые ONNX-веса (onnx-community) и зрелый путь ONNX→TensorRT/OpenVINO.

Минусы:
- **Минимальный R18 = 20M params, DETR-декодер тяжелее CNN** — на Orin Nano 8GB **пограничен по латентности/памяти**, нужен реальный замер (на T4 R18 ~217 FPS по авторам, но T4 мощнее Orin Nano — **нами не измерено**).
- **Доменный разрыв** COCO→бортовой POV, RU/UA не покрыт; нужен fine-tune под 4 класса CARS.
- DETR-семейство чувствительнее к гиперпараметрам/объёму данных при fine-tune, чем CNN-детекторы.

Применимость в пайплайне CARS: PGIE-кандидат — fine-tune RT-DETRv2-S/R18 под бортовой POV + 4 класса через HF, ONNX→TensorRT; коммерчески чистая альтернатива (но проверить edge-латентность).
Edge / Jetson Orin Nano: R18 (20M) — наименьший вариант, пограничный в FP16/INT8; реальная латентность DETR-декодера на устройстве **требует замера**.

### NanoDet-Plus (m, 320 / 416)
**2021 / FCOS-style one-stage anchor-free (Generalized Focal Loss, Ghost-PAN, AGM+DSLA, backbone ShuffleNetV2) / m@320 ~1.17M params, 0.9 GFLOPs; веса ~980 KB int8 / 1.8 MB fp16 / Лицензия: Apache-2.0 на КОД И ВЕСА (комм.: ДА) / Форматы: PyTorch, export_onnx.py→ONNX→TensorRT + ncnn/MNN/OpenVINO / Edge-fit — самый лёгкий из всех, с огромным запасом**

Ультра-лёгкий мобильный детектор. Вход 320×320 или 416×416 RGB; выход — боксы + class scores (anchor-free), декодирование штатное; классы COCO (нужен fine-tune под CARS).

Плюсы:
- **Чистая Apache-2.0** (код И веса; LICENSE репо «Apache License, Version 2.0, copyright 2020-2021 RangiLyu»), предельно лёгкий — идеален для жёсткого edge-бюджета и низкой латентности.
- Зрелый экспорт ONNX/ncnn/MNN/OpenVINO, простой fine-tune.
- **Высокий FPS-запас оставляет ресурс Orin Nano** под SGIE-классификаторы и LP/OCR в пайплайне CARS.

Минусы:
- **Низкая абсолютная точность (мобильный класс)** — для контроля доступа/патруля с требованием precision≥0.90 может не дотянуть даже после fine-tune.
- **Доменный разрыв** COCO→бортовой POV, RU/UA не покрыт; **мелкие/дальние объекты (до 50 м) — слабое место** (recall может проседать).
- Anchor-free FCOS-голова с собственным постпроцессингом — интеграция в DeepStream требует кастомного парсера.

Применимость в пайплайне CARS: PGIE для ультра-лёгкого профиля — fine-tune NanoDet-Plus-m под бортовой POV если приоритет латентность/память над точностью; коммерчески чистый.
Edge / Jetson Orin Nano: самый лёгкий из всех (1.17M, <2MB) — с огромным запасом влезает в 8GB, минимальная латентность/память. **Конкретный FPS не измерен.**

### OpenVINO vehicle-detection-adas-0002 (Intel OMZ)
**SSD-фреймворк с tuned MobileNet v1 (подтверждено README/докой OMZ) / лёгкая SSD-MobileNetV1, точное число параметров не приведено (единицы M, mobile-класс) / Лицензия: Apache-2.0 на код И веса (Intel OMZ) (комм.: ДА) / Форматы: OpenVINO IR нативно; путь IR→ONNX→TensorRT нестандартный / Edge-fit лёгкий, но рантайм заточен под Intel — на Jetson трение**

SSD-детектор. Вход `1×3×384×672` (B,C,H,W), **BGR** (подтверждено); выход `1×1×200×7` (image_id, label, conf, bbox, до 200 детекций). Детектирует **ТОЛЬКО vehicle (1 класс, label 1)**.

Плюсы:
- **Чистая Apache-2.0** (Intel OMZ; LICENSE репозитория OMZ Apache-2.0) — коммерчески безопасно без оговорок.
- ADAS/front-facing viewpoint **близок к бортовому POV CARS** (лучше TrafficCamNet по домену, как у DashCamNet).
- Очень лёгкая SSD-MobileNet под edge.

Минусы:
- **Только 1 класс (vehicle)** — нет person/bicycle/road_sign; **не покрывает участников движения**, недостаточно для PGIE CARS.
- **Заточена под OpenVINO/Intel-рантайм;** путь на Jetson TensorRT (IR→ONNX→TensorRT) **нестандартный и рискованный** (на Jetson целевой рантайм — TensorRT, не OpenVINO).
- Устаревшая SSD-MobileNetV1 (низкая точность по совр. меркам), без RU/UA-специфики; обучающие данные неизвестны.

Применимость в пайплайне CARS: **не как самостоятельный PGIE** (мало классов), а как вспомогательный/референсный vehicle-детектор или быстрый baseline; для CARS малопригоден из-за рантайм-несовместимости с Jetson и 1 класса.
Edge / Jetson Orin Nano: очень лёгкая (влезет с запасом), но рантайм OpenVINO заточен под Intel — для Jetson/TensorRT нужен нестандартный экспортный путь, что снижает edge-практичность именно на целевом устройстве.

### Ultralytics YOLO (YOLOv8 / YOLO11 / YOLO12)
**2023–2025 / Anchor-free YOLO (CSP-backbone, C2f/C3k2, decoupled head) / YOLOv8n ~3.2M, YOLO11n ~2.6M (целевые для edge) / Лицензия: AGPL-3.0 на КОД И ВЕСА (комм.: НЕТ — БЛОКЕР, подтверждён) / Форматы: PyTorch, лучший в классе экспорт ONNX/TensorRT/OpenVINO/TFLite/CoreML / Edge-fit технически отличный — но юридически неприменим**

Новейшие итерации YOLO. Вход 640×640 RGB (`/255`); выход — боксы + class scores, нужен NMS (или end2end-вариант); классы COCO (нужен fine-tune под CARS).

Плюсы:
- **Технически лучший tooling:** простейший fine-tune, экспорт ONNX→TensorRT INT8, отличный edge-fit (YOLO11n).
- Высокая точность/скорость, огромное сообщество и готовые рецепты под Jetson.

Минусы:
- **БЛОКЕР (подтверждён по ultralytics.com/license + docs):** **AGPL-3.0 (copyleft) распространяется на код И на дообученные веса** («all Ultralytics YOLO trained models fall under AGPL-3.0»; compliance требует публикации исходников всего derivative work, включая веса) — для проприетарного CARS нужна платная **Ultralytics Enterprise License**.
- Без покупки Enterprise — обязательство **открыть исходники всего продукта** (включая веса), что неприемлемо для коммерческого CARS.
- Доменный разрыв COCO→бортовой POV, RU/UA не покрыт (общий минус всех COCO-детекторов).

> Примечание: **YOLO12 не имеет отдельной чистой лицензии** — наследует Ultralytics AGPL.

Применимость в пайплайне CARS: технически — отличный PGIE, но **юридически НЕ применим в проприетарном CARS** без платной Enterprise License. Допустимо только для внутренних R&D-экспериментов/бенчмарка, не для поставки.
Edge / Jetson Orin Nano: YOLOv8n/YOLO11n — технически один из лучших вариантов под Jetson (легчайшие, высокий FPS, INT8). Лицензия блокирует прод-использование.

### YOLO-NAS (Deci / SuperGradients)
**2023 / NAS-сгенерированная YOLO-архитектура (quantization-friendly блоки) / S/M/L; INT8-friendly / Лицензия: КОД Apache-2.0, но ВЕСА — non-commercial Deci (комм.: НЕТ — БЛОКЕР, подтверждён вычиткой) / Форматы: PyTorch (SuperGradients), экспорт ONNX/TensorRT / Edge-fit S технически реалистичен — но веса NC**

NAS-детектор. Вход 640×640 RGB; выход — боксы + class scores; классы COCO (нужен fine-tune под CARS).

Плюсы:
- Хороший точность/латентность-баланс, INT8-friendly дизайн под edge.
- Код Apache-2.0, экспорт ONNX/TensorRT доступен.

Минусы:
- **БЛОКЕР (подтверждён вычиткой LICENSE.YOLONAS.md):** код Apache-2.0, но **предобученные ВЕСА — под отдельной НЕ-КОММЕРЧЕСКОЙ лицензией Deci**. Прямые запреты: «shall not resell, lease, sublicense or distribute the Software to any person»; нельзя «for any commercial use, including in connection with any models used in a production environment» без соглашения с Deci; нельзя реверсить/модифицировать.
- Коммерчески чистый путь только: обучить с нуля самим (дорого) или купить лицензию у Deci — **нецелесообразно vs Apache-альтернатив**.
- Доменный разрыв COCO→бортовой POV, RU/UA не покрыт; **Deci поглощена NVIDIA — риск поддержки/доступности весов**.

Применимость в пайплайне CARS: **НЕ применим как готовые веса** в коммерческом CARS (NC-веса). Только обучение с нуля или платная лицензия — нецелесообразно при наличии D-FINE/RF-DETR/PP-YOLOE+ (Apache).
Edge / Jetson Orin Nano: S-вариант технически реалистичен, дизайн INT8-friendly. Лицензия весов блокирует прод-использование независимо от edge-fit.

---

## Сводная таблица аналогов
| Модель | Год | Архитектура | Параметры/размер | Лицензия (комм.?) | Форматы (ONNX/TRT) | Edge-fit (Orin Nano) | Домен-fit | Вердикт |
|---|---|---|---|---|---|---|---|---|
| DashCamNet (NVIDIA TAO) | актуальный TAO | DetectNet_v2 ResNet-18 | единицы-десятки MB (~5.4MB класс) | ВЕСА: NVIDIA Model EULA (UNCLEAR — блокер) | .etlt→ONNX→TRT FP16/INT8 (нативно) | отличный (drop-in DeepStream) | лучший — dashcam POV ≈ бортовой | primary |
| D-FINE (N/S/M/L/X) | 2024 | DETR real-time, NMS-free | N:4M / S:10M / M:19M / L:31M / X:62M | Apache-2.0 код+веса (ДА) | штатный ONNX (NMS-free)→TRT | N/S ОК, DETR тяжелее CNN — замер | COCO/O365 web POV — gap, нет RU/UA | primary |
| RF-DETR (Nano→Large) | 2025 | Real-time DETR (DINOv2) | Core Nano→Large (L ~129M) | Core Apache-2.0 (ДА); XL/2XL PML 1.0 (НЕТ) | export('onnx')→trtexec TRT | Nano/Small ОК, DINOv2 проверять память | COCO web POV — gap, нет RU/UA | primary |
| PP-YOLOE+ (s/m/l/x) | 2022 | Anchor-free YOLO (CSPRepResNet+ET-head) | s/m edge; l ~51.4 mAP; x ~100M | Apache-2.0 код+веса (ДА) | paddle2onnx→TRT FP16 | s/m ОК; x тяжёлый; +шаг Paddle→ONNX | COCO/O365 web POV — gap, нет RU/UA | secondary |
| YOLOX (Nano/Tiny/S/M/L/X) | 2021 | Anchor-free YOLO (CSPDarknet, SimOTA) | Nano:0.91M / Tiny ~5M / S ~9M … X ~99M | Apache-2.0 код+веса (ДА) | ONNX/TRT/ncnn/OpenVINO (зрелый) | Nano/Tiny отличный (легчайшие) | COCO web POV — gap, нет RU/UA; стар (2021) | secondary |
| RT-DETRv2/RT-DETR (R18/R34/R50) | 2023–2024 | Real-time DETR (ResNet+hybrid enc, NMS-free) | R18:20M / R34:31M / R50:42M | Apache-2.0 код+веса (ДА) | готовый ONNX→TRT/OpenVINO | R18 пограничен — замер латентности | COCO/O365 web POV — gap, нет RU/UA | secondary |
| NanoDet-Plus (m, 320/416) | 2021 | FCOS anchor-free (ShuffleNetV2, GFL, Ghost-PAN) | m@320 ~1.17M, <2MB | Apache-2.0 код+веса (ДА) | export_onnx→TRT + ncnn/MNN | самый лёгкий, огромный запас | COCO web POV — gap; мелкие/дальние слабо | marginal |
| OpenVINO vehicle-detection-adas-0002 | — | SSD + MobileNet v1 | лёгкая (единицы M, не приведено) | Apache-2.0 код+веса (ДА) | IR нативно; IR→ONNX→TRT нестандартный | лёгкий, но рантайм Intel — трение на Jetson | ADAS POV близок, НО 1 класс (vehicle) | marginal |
| Ultralytics YOLO (v8/11/12) | 2023–2025 | Anchor-free YOLO (CSP, C2f/C3k2) | YOLOv8n ~3.2M / YOLO11n ~2.6M | AGPL-3.0 код+веса (НЕТ — блокер) | лучший в классе ONNX/TRT/… | технически отличный (YOLO11n) | COCO web POV — gap, нет RU/UA | not_recommended |
| YOLO-NAS (Deci/SuperGradients) | 2023 | NAS YOLO (INT8-friendly блоки) | S/M/L | КОД Apache-2.0; ВЕСА NC Deci (НЕТ — блокер) | ONNX/TRT (INT8-friendly) | S технически ОК | COCO web POV — gap, нет RU/UA | not_recommended |

---

## Рекомендации

**Primary (брать в первую очередь):**

- **DashCamNet (NVIDIA TAO) — лучший domain-fit и drop-in в текущий стек.** Единственный аналог, обученный на dashcam POV (совпадает с бортовым сценарием CARS), с тем же DetectNet_v2-декодером, входом 960×544 и путём ONNX→TensorRT — закрывает главный провал TrafficCamNet без переписывания парсера. **Лицензионное предостережение — критично:** исходные предобученные веса под **NVIDIA Model EULA** редистрибутировать в продукте **запрещено** (подтверждено форумом NVIDIA, тред 281977) → unclear/блокер до legal review. Коммерчески чист **только производный (дообученный) деплой**. Роль: либо domain-релевантная **инициализация для fine-tune** под RU/UA (после чего производный деплой ОК), либо референс домена.

- **D-FINE-S/N — №1 среди open-weights с чистой лицензией.** **Apache-2.0 на код И веса** (вычитано), SOTA real-time DETR (по авторам), NMS-free упрощает деплой, Objects365-претрейн — хорошая база для fine-tune. **Edge-предостережение:** DETR-декодер тяжелее DetectNet_v2 — обязателен **реальный замер FPS/памяти на Orin Nano в INT8** (цифры авторов на T4/4090, нами не измерено).

- **RF-DETR-Nano/Small — чистая альтернатива, заточенная под fine-tuning.** **Core Nano→Large под Apache-2.0** (код И веса). **Ловушка лицензии:** XL/2XL (rfdetr[plus]) — PML 1.0 проприетарно; брать только Core Nano→Large. **Edge-предостережение:** DINOv2-backbone проверять по памяти на Orin Nano (не измерено).

**Secondary:**

- **PP-YOLOE+_s/m** (Apache-2.0 код И веса, Objects365-претрейн) — зрелая альтернатива; минус — экосистема Paddle (шаг paddle2onnx, непривычный для PyTorch-команды).
- **YOLOX-Nano/Tiny** (Apache-2.0 код И веса) — **самый надёжный выбор для жёсткого edge-бюджета**: легчайшие, зрелый экспорт, много Jetson-примеров; минус — архитектура 2021 года (точность ниже DETR) и NMS.
- **RT-DETRv2-S/R18** (Apache-2.0, HF transformers) — удобный fine-tune/экспорт; минус — R18=20M, DETR-декодер **пограничен по латентности на Orin Nano**, нужен замер.

**Marginal:**

- **NanoDet-Plus-m** (Apache-2.0) — ультра-лёгкий запасной при приоритете латентности над точностью; минус — низкая абсолютная точность и слабость на мелких/дальних объектах (до 50 м), требование precision≥0.90 под вопросом.
- **OpenVINO vehicle-detection-adas-0002** (Apache-2.0) — не самостоятельный PGIE (только 1 класс vehicle, нет участников движения), рантайм заточен под Intel (на Jetson IR→ONNX→TensorRT нестандартен). Только как референс/baseline.

**Not recommended (лицензионные блокеры):**

- **Ultralytics YOLO (v8/11/12)** — технически отличный, но **AGPL-3.0 на код И обученные веса**: для проприетарного CARS нужна платная Enterprise License, иначе обязательство открыть весь продукт. Только внутренний R&D, не поставка. YOLO12 наследует ту же AGPL.
- **YOLO-NAS (Deci)** — код Apache-2.0, но **веса под non-commercial Deci** (вычитано LICENSE.YOLONAS.md): production-использование весов запрещено. Нецелесообразно при наличии Apache-альтернатив; плюс риск доступности (Deci поглощена NVIDIA).

**Честный вывод по домену.** Нет открытого аналога, одновременно (а) обученного на бортовом/dashcam POV, (б) с коммерчески-чистой лицензией на **веса**, (в) с RU/UA-спецификой. Лучший domain-fit (DashCamNet) — под EULA (только производный деплой); чистые Apache-веса (D-FINE/RF-DETR/PP-YOLOE+/YOLOX/RT-DETRv2/NanoDet/OMZ-ADAS) обучены на COCO/Objects365 (web POV) → **доменный разрыв и нет RU/UA**. Готового drop-in коммерчески-чистого аналога **не существует**.

**Рекомендуемый путь для PGIE CARS (раз чистого дроп-ина нет):** взять **Apache-2.0 модель под edge** (приоритет D-FINE-S/N или RF-DETR-Nano как SOTA; либо YOLOX-Nano/Tiny / PP-YOLOE+_s как проверенные лёгкие; RT-DETRv2-R18 пограничен по латентности; NanoDet-Plus — ультра-лёгкий запасной) → **fine-tune на собственном датасете бортового POV RU/UA** (day/night/IR, 5–50 м, 4 класса) → экспорт ONNX→TensorRT FP16/INT8. Это снимает И domain gap, И лицензионный риск. Параллельно **DashCamNet** — domain-релевантная инициализация для fine-tune (производный деплой коммерчески ОК по EULA, требует legal review). Вопрос обучающих данных под дообучение разобран в [исследовании датасетов-аналогов](../dataset_research/01_detection_bdd100k.md).

> **Что искали и не нашли:** открытого детектора, предобученного именно на бортовом POV с RU/UA-номерами/сценами — таких готовых весов в открытом доступе (HF/GitHub/NGC) **нет**. PeopleNet (NVIDIA) — person-детектор (не ТС), под тем же EULA. EfficientDet/SSD-MobileNet — устаревшие, без выигрыша vs YOLOX/NanoDet, опущены.

---

## Ссылки

- DashCamNet — NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet
- DashCamNet — TAO docs: https://docs.nvidia.com/tao/tao-toolkit-archive/tao-30-2108/text/purpose_built_models/dashcamnet.html
- NVIDIA TAO commercial licensing — форум (тред 281977): https://forums.developer.nvidia.com/t/understanding-tao-models-commercial-licensing/281977
- DashCamNet on Jetson — Seeed wiki: https://wiki.seeedstudio.com/DashCamNet-with-Jetson-Xavier-NX-Multicamera/
- D-FINE — GitHub: https://github.com/Peterande/D-FINE
- D-FINE — LICENSE (Apache-2.0): https://github.com/Peterande/D-FINE/blob/master/LICENSE
- D-FINE — HuggingFace: https://huggingface.co/Peterande/D-FINE
- D-FINE — paper (arXiv): https://arxiv.org/pdf/2410.13842
- RF-DETR — GitHub: https://github.com/roboflow/rf-detr
- RF-DETR — docs: https://rfdetr.roboflow.com/latest/
- RF-DETR — LICENSE: https://github.com/roboflow/rf-detr/blob/develop/LICENSE
- RF-DETR Nano/Small/Medium — блог Roboflow: https://blog.roboflow.com/rf-detr-nano-small-medium/
- PP-YOLOE+ — PaddleDetection config: https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe
- PaddleDetection — LICENSE (Apache-2.0): https://github.com/PaddlePaddle/PaddleDetection/blob/develop/LICENSE
- PP-YOLOE — paper (arXiv): https://arxiv.org/pdf/2203.16250
- YOLOX — GitHub: https://github.com/Megvii-BaseDetection/YOLOX
- YOLOX — LICENSE (Apache-2.0): https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE
- YOLOX — paper (arXiv): https://arxiv.org/pdf/2107.08430
- RT-DETR — GitHub (lyuwenyu): https://github.com/lyuwenyu/RT-DETR
- RT-DETR — LICENSE (Apache-2.0): https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE
- RT-DETR R18 — HuggingFace (PekingU): https://huggingface.co/PekingU/rtdetr_r18vd
- RT-DETR R50 ONNX — HuggingFace (onnx-community): https://huggingface.co/onnx-community/rtdetr_r50vd
- NanoDet — GitHub: https://github.com/RangiLyu/nanodet
- NanoDet — LICENSE (Apache-2.0): https://github.com/RangiLyu/nanodet/blob/main/LICENSE
- OpenVINO vehicle-detection-adas-0002 — docs: https://docs.openvino.ai/2023.3/omz_models_model_vehicle_detection_adas_0002.html
- OpenVINO vehicle-detection-adas-0002 — README: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-detection-adas-0002/README.md
- OpenVINO OMZ — LICENSE (Apache-2.0): https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE
- Ultralytics — GitHub: https://github.com/ultralytics/ultralytics
- Ultralytics — лицензия (AGPL-3.0 / Enterprise): https://www.ultralytics.com/license
- Ultralytics — docs: https://docs.ultralytics.com/
- Ultralytics YOLO11 — HuggingFace: https://huggingface.co/Ultralytics/YOLO11
- YOLO-NAS — LICENSE.YOLONAS.md (веса NC): https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md
- YOLO-NAS — YOLONAS.md: https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md
- YOLO-NAS — issue #2034 (лицензия весов): https://github.com/Deci-AI/super-gradients/issues/2034

---

## История изменений
- 2026-06-05 — создан в рамках исследования открытых моделей-аналогов стека CARS (многоагентный workflow, веб-поиск по первоисточникам, лицензии перепроверены адверсариально).
