# Аналоги модели «FaceDetect (NVIDIA FaceNet)» для задачи «Face Detection — детекция лиц»

> **См. также:** спецификация исходной модели — [facedetect.md](../about_models/facedetect.md) · реестр моделей — [models.md](../../models.md) · исследование датасетов-аналогов (та же задача) — [03_facedetect_widerface.md](../dataset_research/03_facedetect_widerface.md) · сводка исследования моделей — [00_SUMMARY.md](00_SUMMARY.md)

## Контекст и требования

**Целевая модель — FaceDetect (NVIDIA TAO FaceNet).** Это детектор лиц семейства **DetectNet_v2** (backbone **ResNet-18**, вариант `pruned_quantized_v2.0.1`), один класс `face`, фиксированный вход **736×416**, выход — axis-aligned bbox. В конвейере CARS FaceDetect играет роль **SGIE4** (вторичный детектор в DeepStream): он находит лица водителя и пассажиров, чтобы система могла сохранить кропы и зафиксировать присутствие людей. Модель питает требование **FR-06** (сохранение кропов лиц), поля БД `face_count` / `face_coords` и артефакты `face_images/{track_id}.bmp`.

**Технические ограничения цели.** Развёртывание идёт на **NVIDIA Jetson Orin Nano 8GB** в реальном времени; рантайм — **ONNX → TensorRT / DeepStream (nvinfer)**, с приоритетом FP16/INT8. Бюджет SGIE-детектора жёсткий: лицо детектируется уже внутри кропа/кадра поверх первичных детекторов, поэтому вес engine и latency должны быть минимальны. Любой аналог должен либо иметь готовый ONNX, либо штатно экспортироваться в ONNX и далее в TensorRT.

**Доменные требования.** Бортовой POV — лица **через лобовое стекло на дистанции 5–50 м**, под углом 0–30°, в режимах **day / night / IR**. RU/UA-специфика к лицам **нерелевантна** (это касается номеров и OCR, не лиц). Главный доменный пробел текущего FaceNet — обучение на веб-фото (WIDER FACE-подобный домен) и отсутствие явного покрытия IR/ночи. Дистанция 50 м означает мелкие лица в кадре — критичный для recall сценарий.

**Коммерческая лицензия — жёсткое требование.** CARS — коммерческий продукт, поэтому non-commercial веса (research-only), copyleft (GPL/AGPL) и невыясненное происхождение весов трактуются адверсариально как **блокеры до legal review или переобучения**. Текущий FaceNet (Model EULA, `.etlt` зашифрован) сам по себе несвободен — это дополнительная причина искать коммерчески-чистые аналоги.

**Что требуется от аналога:**
- 1 класс `face`, bbox-выход, желательно с 5 keypoints (на будущий face-alignment/embedding);
- **коммерчески разрешённая лицензия и на код, и на веса** (приоритет — permissive SPDX: MIT/Apache-2.0);
- готовый ONNX или штатный экспорт ONNX → TensorRT, малый вес engine под бюджет SGIE4 на Orin Nano;
- по возможности — покрытие day/night/**IR** бортового сценария и устойчивость на мелких/дальних лицах;
- совместимость постпроцессинга (decode + NMS) с интеграцией в DeepStream nvinfer.

> **Дисклеймер.** Все цифры точности ниже — **по заявлению авторов** на WIDER FACE, **не измерено нами**: эвалуатор FaceDetect в проекте пока **не реализован**. Метрики на WIDER FACE дополнительно ограничены лицензией датасета **CC BY-NC-ND 4.0** — годятся только как внутренний research-бенчмарк до legal review (см. [03_facedetect_widerface.md](../dataset_research/03_facedetect_widerface.md)). Лицензии всех кандидатов перепроверены адверсариально по первоисточникам (июнь 2026).

---

## Обзор аналогов

### YuNet (face_detection_yunet)
**2023–2026 / anchor-free tiny-детектор (depthwise-separable backbone в духе MobileNet + упрощённый FPN-neck, выход bbox + 5 keypoints + score) / ~0.083M параметров (~75 856), ONNX-файл ~340 KB FP32 + INT8 block-quantized / MIT (комм.: ДА) / нативно ONNX (вкл. int8), путь ONNX→TensorRT подтверждён / отличный edge-fit, domain gap к бортовому POV сохраняется**

Плюсы:
- **ПОДТВЕРЖДЕНА чистая MIT и на код, и на веса** — README OpenCV Zoo дословно «All files in this directory are licensed under MIT License», веса `.onnx` лежат в том же каталоге. Это снимает legal-блокеры коммерции (главное преимущество против FaceNet Model EULA и InsightFace NC).
- Готовые ONNX (включая int8) + **подтверждённый путь ONNX→TensorRT на Jetson** (публичный пример NobuoTsukamoto/tensorrt-examples/python/yunet); крайне малый размер идеален для SGIE-бюджета Orin Nano.
- WIDER FACE AP конкурентна для размера: цитируемые авторами 0.834/0.824/0.708 (другой вариант/версия в README показывает 0.8844/0.8656/0.7503; Easy/Medium/Hard) — **по заявлению авторов, не измерено нами**.
- Даёт 5 keypoints лица бесплатно — полезно для будущего face-alignment/embedding.

Минусы:
- **Domain gap к бортовому POV сохраняется** (обучение на веб-фото WIDER FACE), как и у FaceNet — нужна валидация на automotive-срезе.
- Нет IR/ночи в обучении — для ночных/ИК-сценариев CARS нужна отдельная модель/выборка.
- Слабее на очень мелких/дальних лицах (низкий Hard-AP), что критично на дистанции 50 м.
- Выходной формат отличается от DetectNet_v2 (нужен свой decode + NMS, не переиспользуется `detectnet_v2_decode`).
- Метрики валидации на WIDER FACE ограничены NC-лицензией датасета (CC BY-NC-ND) — только внутренний research-бенчмарк (касается процесса оценки, не самой MIT-модели).
- *Оговорка по данным:* точное число параметров/GFLOPs в README OpenCV Zoo **НЕ указано** — взято из публикаций авторов, не из карточки.

**Применимость в пайплайне CARS:** primary-замена для FaceDetect в Python/ONNX-сервисе (ONNX Runtime) и в DeepStream (ONNX→TRT через nvinfer). Подтверждённая чистая MIT снимает legal-риск FaceNet Model EULA, размер вписывается в SGIE-бюджет. Маппинг класса тривиален (оба — 1 класс `face`), но требуется написать собственный парсер выхода (decode + NMS вместо `detectnet_v2_decode`) и валидацию на automotive-срезе.
**Edge / Jetson Orin Nano:** отлично — ~0.083M параметров, ~0.31 GFLOPs при 360×640, заявлено ~1.6 ms/кадр на 320×320 (**по заявлению авторов, не измерено нами**). Доступен INT8 block-quantized вариант (block_size=64). Минимальный вес engine. Экспорт ONNX→TensorRT подтверждён публичным примером на Jetson.

### NVIDIA FaceDetectIR (faceirnet)
**~2021+ / DetectNet_v2 (GridBox), backbone ResNet-18 / порядок единиц-десятка MB `.etlt` (как у FaceNet ~5.8 MB), есть pruned-варианты / NVIDIA AI/TAO Model EULA (комм.: ДА — деплой, но не permissive) / `.etlt`→ONNX/TensorRT через TAO Toolkit / лучший domain-fit (единственный с IR/ночь), edge-пригоден**

Плюсы:
- **Единственный аналог, штатно покрывающий IR/ночь** — по документации TAO «Primary use case for this model is to detect faces from an IR (infrared) camera» (ПОДТВЕРЖДЕНО). Прямо закрывает доменный пробел CARS (через лобовое стекло, ночь, ИК-подсветка), которого нет ни у WIDER FACE-валидации, ни у текущего FaceNet, ни у YuNet/ULFG.
- **Та же архитектура DetectNet_v2 + ResNet-18 + то же декодирование**, что у текущего FaceNet — реализация эвалуатора и nvinfer-конфиг переиспользуются почти 1:1, минимальная интеграционная стоимость.
- Коммерческий деплой разрешён по Model EULA — **ПОДТВЕРЖДЕНО офиц. ответом staff NVIDIA (Morganh) на форуме**: «they may deploy it in their own commercial applications», без AI Enterprise лицензии.
- Родной путь `.etlt`→ONNX→TensorRT в DeepStream, как у остального стека CARS.

Минусы:
- **Лицензия НЕ permissive** (Model EULA, не SPDX) — нельзя редистрибутировать веса как продукт. Коммерческое разрешение основано на **ОТВЕТЕ STAFF НА ФОРУМЕ, а не на дословном тексте EULA** — для внешней поставки артефактов/engine **ОБЯЗАТЕЛЕН legal review текста Model EULA с карточки NGC** (NVIDIA-доки/FAQ системно отсылают к «read the Model EULA on the model card»).
- **Плохо детектирует мелкие/дальние лица** (<10% площади кадра) — риск на дистанции 50 м.
- Веса `.etlt` зашифрованы, нужен `tlt-model-key` (`tlt_encode`/`nvidia_tlt`) для экспорта ONNX (та же проблема, что у текущего FaceNet).
- Малый вход 384×240 ограничивает разрешение мелких лиц; нет публичных независимых метрик на automotive.

**Применимость в пайплайне CARS:** primary-дополнение/замена для **ночных-ИК сценариев**: ставить рядом с дневным детектором (YuNet/FaceNet) или как основной SGIE4 при наличии ИК-подсветки. Нативно встраивается в текущий DeepStream-пайплайн с минимальной доработкой декодера (тот же `cov+bbox` формат). **ТРЕБУЕТ подтверждения дословного текста Model EULA legal-командой перед внешней поставкой** — форумный ответ не является юридической гарантией.
**Edge / Jetson Orin Nano:** отлично — то же семейство и backbone (ResNet-18 DetectNet_v2), что текущий FaceNet (уже считается edge-пригодным для Orin Nano как SGIE). INT8-калибровка штатна для TAO. Малый вход 384×240 ускоряет инференс. Конкретные тайминги на Orin Nano **не измерены нами**.

### Ultra-Light-Fast-Generic-Face-Detector-1MB (ULFG / RFB / slim)
**2019 / лёгкий SSD-подобный детектор (version-slim и version-RFB с Receptive Field Block) / файл ~1.04–1.1 MB FP32, ~300 KB INT8; 90–109 MFLOPs @320×240 / MIT (комм.: ДА) / готовый ONNX в репозитории, ONNX→TensorRT штатный / минимальный размер, domain gap как у YuNet**

Плюсы:
- **ПОДТВЕРЖДЕНА чистая MIT (код+веса, Copyright (c) 2019 linzai)** — нулевой лицензионный риск для коммерции; веса лежат в репозитории под той же лицензией.
- **Минимальный размер из всех кандидатов** (~1 MB FP32 / ~300 KB INT8) — максимальный запас по бюджету Orin Nano.
- Готовые ONNX в репозитории (slim/RFB 320 и 640, с/без постпроцессинга) + штатный путь в TensorRT (trtexec / ORT-TRT EP); широкая поддержка edge-рантаймов (Caffe/MNN/NCNN/OpenCV DNN).

Минусы:
- **Заметно ниже точность, особенно Hard:** slim 0.77/0.671/0.395, RFB 0.787/0.698/0.438 @320×240 (**по заявлению авторов**) — слабо на мелких/дальних лицах. Авторы прямо предупреждают: при мелком входе резко падает recall мелких лиц, для дальних лиц нужен вход 640×480.
- **Нет keypoints** (только bbox) — нет данных для будущего alignment/embedding.
- Domain gap и отсутствие IR/ночи — как у YuNet (обучен на отфильтрованном WIDER FACE, лица <10 px удалены). Репозиторий старый (2019), низкая активность/maintenance.

**Применимость в пайплайне CARS:** secondary — fallback-кандидат для максимально жёсткого latency-бюджета на Orin Nano, когда лица **крупные и близкие** (контроль доступа, лицо в кадре). При дистанции 50 м и мелких лицах уступает YuNet. Подтверждённая MIT делает его безопасной заменой при достаточной точности на близких сценах.
**Edge / Jetson Orin Nano:** идеален — ~1 MB FP32 / ~300 KB INT8, 90–109 MFLOPs @320×240, один из самых лёгких детекторов лиц, минимальная latency как SGIE. INT8 доступен. Тайминги на Orin Nano **не измерены нами**.

### MediaPipe Face Detection (BlazeFace, short/full range)
**~2019+ / BlazeFace — SSD-подобный детектор (лёгкий feature-extractor в духе MobileNetV1/V2, GPU-friendly anchors, tie-resolution вместо классического NMS, выход bbox + 6 keypoints) / очень малый (доли MB, мобильный TFLite), варианты short-range 128×128 и full-range / Apache-2.0 на апстриме (комм.: ДА — апстрим) / нативно TFLite, ONNX-конвертации в PINTO_model_zoo / отличный edge-fit, слабый domain-fit к бортовому POV**

Плюсы:
- **ПОДТВЕРЖДЁН чистый Apache-2.0 (код + официальные веса Google AI Edge)** — нет лицензионных блокеров на апстриме.
- Готовые ONNX-конвертации (PINTO_model_zoo/030_BlazeFace, вкл. ONNX/OpenVINO/TFTRT/TFLite FP32/16/INT8) с путём в TensorRT FP16/INT8 (ORT-TRT EP, `trt_fp16_enable`); крайне малый размер для edge.
- Даёт **6 keypoints** — полезно для будущего alignment/face mesh.

Минусы:
- **Профиль использования — фронтальные близкие лица (селфи)**, плохой fit к бортовому POV под углом и на дистанции 5–50 м; full-range шире, но всё равно не automotive.
- **Официальные веса — в TFLite; для ONNX нужны сторонние конверсии** (зависимость от PINTO/Qualcomm/STMicro-портов). Апстрим-лицензия Apache-2.0 **не переносится автоматически** на сторонний репозиторий — лицензия КОНКРЕТНОГО порта проверяется отдельно (PINTO обычно permissive — проверять LICENSE; STMicro/Qualcomm — свои условия).
- Нет IR/ночи; ограниченная точность на мелких/повёрнутых лицах vs YuNet/SCRFD.

**Применимость в пайплайне CARS:** secondary — жизнеспособен для сцен **контроля доступа с близким фронтальным лицом** (водитель смотрит в камеру), как сверхлёгкий SGIE на Orin Nano. Для патруля/парковки на дистанции и под углом уступает YuNet. ONNX-путь требует доверенного порта и **отдельной проверки его лицензии**.
**Edge / Jetson Orin Nano:** отлично — спроектирован под мобильный GPU, доли MB, очень низкая latency, подходит под SGIE-бюджет. INT8-варианты доступны через PINTO zoo. Тайминги на Orin Nano **не измерены нами**.

### SCRFD (InsightFace)
**2021 / Sample and Computation Redistribution, anchor-based (варианты 0.5GF/2.5GF/10GF/34GF, есть _KPS с 5 keypoints) / 0.5GF: 0.57M/500 MFLOPs … 34GF: 9.80M/34G / КОД MIT, но ВЕСА non-commercial research only (комм.: НЕТ) / штатный ONNX-экспорт, ONNX→TensorRT возможен / технически идеальный edge-fit, но веса заблокированы лицензией**

Плюсы:
- **Лучший баланс точность/вычисления среди лёгких детекторов** (по WIDER FACE, **по заявлению авторов**): SCRFD-2.5GF 93.8/92.2/77.9, 10GF 95.2/93.9/83.1 (Easy/Medium/Hard); широкий ряд от 0.5GF до 34GF.
- Штатный ONNX-экспорт с динамическим входом (`tools/scrfd2onnx.py`); технически идеально ложится на Orin Nano (0.5GF/2.5GF крошечные — 0.5–0.67M params).
- Есть keypoints-варианты (_KPS) для будущего alignment.

Минусы:
- **БЛОКЕР ДЛЯ КОММЕРЦИИ (ПОДТВЕРЖДЕНО первоисточником — README InsightFace + issue #2022):** код MIT, но **данные с аннотацией и модели, обученные на них (включая SCRFD), дословно «available for non-commercial research purposes only»**. По адверсариальному правилу = запрещено до legal review / переобучения.
- Ограничение распространяется и на **авто-загрузку весов через pip-библиотеку** — нельзя обойти «случайной» загрузкой.
- Domain gap к бортовому POV сохраняется (WIDER FACE, web); нет IR/ночи.
- Для коммерции потребуется полное переобучение SCRFD-кода (MIT) на коммерчески чистом датасете (затратно) либо платная лицензия от InsightFace (`recognition-oss-pack@insightface.ai`).

**Применимость в пайплайне CARS:** not_recommended для прямого коммерческого использования из-за NC-весов. Возможный путь: использовать **только MIT-КОД** SCRFD и **переобучить на коммерчески чистом/собственном automotive-датасете** (тогда веса свои), либо платная лицензия InsightFace. До этого — внутренний research-бенчмарк максимум (планка точности для сравнения чистых аналогов).
**Edge / Jetson Orin Nano:** технически отлично — 0.5GF/2.5GF легко влезают в Orin Nano с низкой latency, INT8/FP16 реализуемы. **Но коммерческий деплой запрещён лицензией весов** — edge-пригодность не реализуема в продукте без переобучения.

### RetinaFace (InsightFace)
**~2019 / single-stage dense localisation (FPN + context modules, multi-task: bbox + 5 keypoints + опц. 3D mesh), backbone ResNet-50 / MobileNet-0.25 / MobileNet0.25 ~0.4–1.7M params, ResNet50 — десятки M / КОД MIT, но ВЕСА non-commercial research only (комм.: НЕТ) / ONNX-экспорт доступен (в т.ч. det_10g внутри пакетов), ONNX→TensorRT возможен / MobileNet-вариант edge-пригоден, но веса заблокированы**

Плюсы:
- **Эталонная точность и устойчивость на WIDER FACE** (исторический бенчмарк, **по заявлению авторов**); MobileNet0.25-вариант edge-пригоден.
- 5 keypoints для alignment; зрелый, широко используемый код (MIT).
- Доступны готовые ONNX (в т.ч. внутри insightface-пакетов как `det_10g.onnx`, `buffalo_l`).

Минусы:
- **БЛОКЕР (ПОДТВЕРЖДЕНО первоисточником — README InsightFace):** предобученные веса RetinaFace (и пакеты `buffalo_l`/`antelopev2`, включая `det_10g`) — **non-commercial research only** (та же оговорка, что у SCRFD). README «MIT» относится к КОДУ, не к весам. Касается и **авто-загрузки через pip**.
- ResNet50-вариант тяжёл для Orin Nano; лёгкий MobileNet0.25 уступает SCRFD по балансу.
- Domain gap, нет IR/ночи.
- **Legacy-упоминание RetinaFace в доках CARS вводит в заблуждение:** `architecture.md` называет FaceDetect как «NVIDIA FaceDetect / RetinaFace», но фактически в стеке стоит **FaceNet (DetectNet_v2), не RetinaFace** — это упоминание НЕ источник истины.

**Применимость в пайплайне CARS:** not_recommended для прямого коммерческого деплоя из-за NC-весов. Тот же путь, что у SCRFD: использовать MIT-код и переобучить на чистых данных, либо платная лицензия InsightFace. Иначе — только внутренний бенчмарк. **Проясняет legacy-противоречие:** «RetinaFace» из `architecture.md` не следует тащить в продакшн как готовые insightface-веса (NC-блокер).
**Edge / Jetson Orin Nano:** MobileNet0.25-вариант лёгкий и edge-пригоден; ResNet50 — слишком тяжёл для SGIE. **Коммерческий деплой блокирован лицензией весов** — edge-пригодность не применима в продукте без переобучения.

### YOLO5Face (deepcam-cn/yolov5-face)
**2022 (ECCV Workshops) / YOLOv5 + Landmarks-head (детекция лиц + 5 keypoints), варианты yolov5n-0.5 … yolov5l / yolov5n-0.5: 0.447M/0.571 GFLOPs … yolov5l: 46.6M / GPL-3.0 copyleft (комм.: НЕТ) / TensorRT-экспорт документирован, ONNX через сторонние реализации / лёгкие варианты edge-пригодны, но лицензия блокирует коммерцию**

Плюсы:
- **Высокая WIDER FACE AP, особенно Hard** (**по заявлению авторов**): yolov5s 94.3/92.6/83.2, yolov5m 95.3/93.8/85.3 — лучше большинства лёгких аналогов.
- Лёгкие варианты (n-0.5/n: 0.45–1.7M params) edge-пригодны; есть keypoints; документирован TensorRT FP16.
- Зрелая YOLO-экосистема, знакомый стек экспорта.

Минусы:
- **БЛОКЕР (ПОДТВЕРЖДЕНО):** репозиторий под **GPL-3.0 copyleft** (основан на ultralytics/yolov5) — несовместим с проприетарным CARS; распространение продукта потребует раскрытия исходников или платной лицензии. Веса наследуют copyleft кода.
- Унаследованный риск Ultralytics-семейства (свежие релизы — **AGPL-3.0**, ещё строже) — любые обновления усугубляют copyleft.
- Domain gap, нет IR/ночи; для коммерции нужна замена лицензии / полное переобучение на permissive-стеке.

**Применимость в пайплайне CARS:** not_recommended для коммерческого CARS из-за GPL-3.0 copyleft. Допустимо только как **внутренний research-эталон точности** (планка для сравнения чистых аналогов). Для продакшна — либо платная коммерческая лицензия (если предложат), либо отказ в пользу YuNet/FaceDetectIR.
**Edge / Jetson Orin Nano:** yolov5n-0.5/yolov5n (0.45–1.7M params) реально влезают в Orin Nano с хорошей latency, документирован TensorRT FP16. **Но лицензия блокирует коммерцию** — edge-пригодность не применима в продукте.

### MTCNN (facenet-pytorch)
**~2016 (Zhang et al.) / Multi-task Cascaded CNN (каскад P-Net → R-Net → O-Net с image-pyramid, выход bbox + 5 keypoints) / очень малый (три крошечные сети, суммарно единицы MB) / КОД MIT, ВЕСА — unclear (портированы из TF-репозитория Sandberg, нет явного SPDX) / нативно PyTorch, единого готового ONNX НЕТ / каскад плохо ложится на TensorRT/DeepStream**

Плюсы:
- MIT-код, широчайшая распространённость, минимальная интеграционная стоимость в Python/PyTorch.
- Крошечные сети + 5 keypoints; хороший baseline для близких фронтальных лиц.
- Авто-загрузка весов, нулевой порог входа для прототипа.

Минусы:
- **Каскад P/R/O-Net + image-pyramid плохо экспортируется в единый ONNX/TensorRT** (переменные формы, динамическое число proposals, CPU-постпроцессинг между стадиями) — не вписывается в DeepStream nvinfer-пайплайн CARS.
- **Происхождение/лицензия ВЕСОВ — unclear (ПОДТВЕРЖДЕНО):** веса P/R/O-Net «initialized using parameters ported from David Sandberg TensorFlow FaceNet repository», без явного SPDX на сами веса. По адверсариальному правилу — относиться осторожно до проверки первоисточника весов.
- Заметно ниже точность современных детекторов, особенно на мелких/дальних/повёрнутых лицах; устаревший подход.

**Применимость в пайплайне CARS:** marginal — только как быстрый Python-baseline для прототипа/сравнения, **не для продакшн-деплоя** в DeepStream/TensorRT (плохой экспорт каскада). Для бортового real-time на Orin Nano уступает YuNet по всем осям, кроме простоты прототипирования.
**Edge / Jetson Orin Nano:** сети крошечные, но каскадная архитектура с image-pyramid и динамическим числом proposals плохо подходит под TensorRT/DeepStream — на Orin Nano латентность нестабильна. Не рекомендуется как edge-цель.

---

## Сводная таблица аналогов

| Модель | Год | Архитектура | Параметры/размер | Лицензия (комм.?) | Форматы (ONNX/TRT) | Edge-fit (Orin Nano) | Домен-fit | Вердикт |
|---|---|---|---|---|---|---|---|---|
| YuNet (face_detection_yunet) | 2023–2026 | anchor-free tiny (depthwise-sep + FPN), bbox + 5 kpts | ~0.083M (~75 856), ONNX ~340 KB FP32 + int8 | **MIT — ДА** (код+веса, подтв.) | ONNX (вкл. int8) нативно; ONNX→TRT подтв. на Jetson | Отлично (~0.31 GFLOPs, ~1.6 ms @320, по заявл. авторов) | WIDER FACE web, gap к бортовому POV; нет IR/ночи; слаб на мелких | **primary** (день/RGB) |
| NVIDIA FaceDetectIR (faceirnet) | 2021+ | DetectNet_v2 (GridBox), ResNet-18 | единицы-десятки MB `.etlt`, вход 384×240 | Model EULA — ДА деплой (НЕ permissive; форум-ответ, не EULA) | `.etlt`→ONNX/TRT через TAO | Отлично (то же семейство, что FaceNet; INT8 штатно) | **Лучший: IR/ночь подтв. (primary use=IR)**; слаб на мелких <10% | **primary** (ночь/ИК) |
| Ultra-Light-Fast-Generic-Face-Detector-1MB | 2019 | лёгкий SSD-подобный (slim / RFB) | ~1 MB FP32 / ~300 KB INT8; 90–109 MFLOPs | **MIT — ДА** (код+веса, подтв.) | готовый ONNX; ONNX→TRT штатный | Идеален (минимальный из всех) | WIDER FACE web, нет IR; слаб на мелких/дальних | **secondary** |
| MediaPipe Face Detection (BlazeFace) | 2019+ | BlazeFace (MobileNet-like SSD, tie-resolution), bbox + 6 kpts | доли MB (TFLite), 128×128 / full-range | Apache-2.0 — ДА (апстрим; ONNX-порты проверять отдельно) | TFLite нативно; ONNX через PINTO zoo; TRT FP16/INT8 | Отлично (мобильный GPU, доли MB) | Слабый: профиль «селфи/фронт», не бортовой POV; нет IR | **secondary** |
| SCRFD (InsightFace) | 2021 | SCRFD (anchor-based), варианты 0.5–34 GF, _KPS | 0.57M/0.5G … 9.80M/34G | КОД MIT, **ВЕСА NC research-only — НЕТ** (подтв.) | штатный ONNX-экспорт; ONNX→TRT | Технически отлично (но веса блокированы) | WIDER FACE web, нет IR; лучший баланс AP/FLOPs | **not_recommended** |
| RetinaFace (InsightFace) | 2019 | single-stage (FPN + context), bbox + 5 kpts; ResNet50/MobileNet0.25 | MobileNet0.25 ~0.4–1.7M; ResNet50 десятки M | КОД MIT, **ВЕСА NC research-only — НЕТ** (подтв.) | ONNX (вкл. det_10g); ONNX→TRT | MobileNet0.25 edge-ок; ResNet50 тяжёл (веса блокированы) | WIDER FACE web, нет IR; эталон точности | **not_recommended** |
| YOLO5Face (deepcam-cn/yolov5-face) | 2022 | YOLOv5 + Landmarks-head, bbox + 5 kpts | yolov5n-0.5 0.447M … yolov5l 46.6M | **GPL-3.0 copyleft — НЕТ** (подтв.) | TensorRT документирован; ONNX через сторонние | Лёгкие n-варианты edge-ок (но лицензия блокирует) | WIDER FACE web, нет IR; высокая Hard-AP | **not_recommended** |
| MTCNN (facenet-pytorch) | 2016 | каскад P/R/O-Net + image-pyramid, bbox + 5 kpts | единицы MB (три крошечные сети) | КОД MIT, **ВЕСА unclear** (порт из Sandberg TF, нет SPDX) | PyTorch; единого ONNX НЕТ, экспорт каскада нетривиален | Плохо (каскад/пирамида не ложится на TRT/DeepStream) | средние/крупные фронтальные; нет IR; ниже SOTA | **marginal** |

---

## Рекомендации

**Primary (день / RGB) — YuNet.** Подтверждённая чистая **MIT на код И веса** (дословно в README OpenCV Zoo) снимает legal-риск FaceNet Model EULA; готовый ONNX (вкл. int8) + подтверждённый путь ONNX→TensorRT на Jetson; ~0.083M параметров — минимальный вес для SGIE4 на Orin Nano. Это основной кандидат на замену FaceDetect в дневном/RGB-сценарии. Требует написания собственного парсера выхода (decode + NMS) и валидации на automotive-срезе.

**Primary (ночь / ИК) — NVIDIA FaceDetectIR.** Единственный аналог, штатно покрывающий IR/ночь (подтверждено docs TAO: primary use = IR-камера) — прямо закрывает доменный пробел CARS. Та же DetectNet_v2/ResNet-18, что у текущего FaceNet → переиспользование декодера и nvinfer-конфига ~1:1. **Лицензионное предостережение:** Model EULA не permissive; коммерческий деплой подтверждён офиц. ответом staff NVIDIA на форуме, но это **НЕ дословный текст EULA** — перед поставкой компилированного engine ВНЕШНИМ заказчикам **обязателен legal review текста Model EULA** с карточки NGC.

**Secondary — ULFG-1MB и MediaPipe BlazeFace.** Оба permissive (MIT и Apache-2.0 соответственно). **ULFG-1MB** — резерв по latency на крупных близких лицах (контроль доступа), подтверждённая MIT, минимальный вес; уступает YuNet на мелких/дальних лицах. **BlazeFace** — для сцен с близким фронтальным лицом (водитель смотрит в камеру); официальные веса в TFLite, для ONNX нужен сторонний порт с **отдельной проверкой его лицензии** (Apache-2.0 апстрима не переносится автоматически).

**Лицензионные блокеры (not_recommended).** **SCRFD** и **RetinaFace** (InsightFace) дают лучшую заявленную WIDER FACE AP и технически идеально ложатся на Orin Nano, но их предобученные веса — строго **non-commercial research only** (подтверждено README + issue #2022; касается и авто-загрузки через pip) при формально MIT-коде. **YOLO5Face** — **GPL-3.0 copyleft**, несовместим с проприетарным продуктом. Все трое годятся только как **внутренние research-эталоны точности** или (для SCRFD/RetinaFace) как **MIT-КОД для переобучения** на коммерчески чистых данных. **MTCNN** — marginal Python-baseline с **unclear-происхождением весов** и плохим экспортом каскада; не для продакшна.

**Доменные пробелы и путь дообучения.** Открытого детектора лиц, обученного именно на automotive-POV «через лобовое стекло» 5–50 м, **не существует** — все permissive-кандидаты (YuNet/ULFG/BlazeFace) обучены на WIDER FACE-подобном web-домене, domain gap сохраняется как у текущего FaceNet. Рекомендация: собрать **собственный automotive-срез** для валидации (см. категории WIDER FACE 14—Traffic, 5—Car_Accident, 59—people--driving--car в [03_facedetect_widerface.md](../dataset_research/03_facedetect_widerface.md)) и дообучить. Для **IR/ночи** единственный готовый ответ — NVIDIA FaceDetectIR (Model EULA); permissive-аналога для IR-лиц в открытом доступе **не найдено** — коммерчески чистый IR-детектор потребует собственной съёмки ИК-выборки + дообучения YuNet/ULFG или переобучения SCRFD-кода (MIT). RU/UA-специфика к лицам нерелевантна. Все цифры точности — по заявлению авторов, **не измерено нами** (эвалуатор FaceDetect ещё не реализован), а метрики на WIDER FACE ограничены NC-лицензией датасета.

## Ссылки

- YuNet (OpenCV Zoo): https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- YuNet README (лицензия MIT — дословно): https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md
- YuNet (зеркало HF): https://huggingface.co/opencv/face_detection_yunet
- libfacedetection.train (обучение YuNet, ShiqiYu): https://github.com/ShiqiYu/libfacedetection
- YuNet ONNX→TensorRT на Jetson (пример): https://github.com/NobuoTsukamoto/tensorrt-examples/blob/main/python/yunet/README.md
- NVIDIA FaceDetectIR (NGC): https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facedetectir
- FaceDetectIR (docs TAO — primary use = IR): https://docs.nvidia.com/tao/tao-toolkit-archive/5.2.0/text/model_zoo/cv_models/faceirnet.html
- NVIDIA forum (ответ staff о коммерческом деплое FaceNet/FaceDetect): https://forums.developer.nvidia.com/t/clarification-needed-is-facenet-facedetect-free-for-commercial-use-or-does-it-require-an-ai-enterprise-license/363787
- Ultra-Light-Fast-Generic-Face-Detector-1MB (Linzaer): https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
- ULFG-1MB LICENSE (MIT, Copyright 2019 linzai): https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/LICENSE
- MediaPipe Face Detection (Google AI Edge): https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector
- BlazeFace ONNX-конвертации (PINTO_model_zoo): https://github.com/PINTO0309/PINTO_model_zoo/tree/main/030_BlazeFace
- MediaPipe Face Detection (Qualcomm HF-порт): https://huggingface.co/qualcomm/MediaPipe-Face-Detection
- SCRFD (InsightFace): https://github.com/deepinsight/insightface/tree/master/detection/scrfd
- InsightFace README (лицензия: код MIT, данные/модели NC): https://github.com/deepinsight/insightface/blob/master/README.md
- InsightFace issue #2022 (NC-оговорка по весам): https://github.com/deepinsight/insightface/issues/2022
- SCRFD (страница проекта): https://insightface.ai/scrfd
- RetinaFace (InsightFace): https://github.com/deepinsight/insightface/tree/master/detection/retinaface
- YOLO5Face (deepcam-cn, GPL-3.0): https://github.com/deepcam-cn/yolov5-face
- MTCNN (facenet-pytorch, timesler): https://github.com/timesler/facenet-pytorch
- facenet-pytorch LICENSE (код MIT): https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md

## История изменений
- 2026-06-05 — создан в рамках исследования открытых моделей-аналогов стека CARS (многоагентный workflow, веб-поиск по первоисточникам, лицензии перепроверены адверсариально).
