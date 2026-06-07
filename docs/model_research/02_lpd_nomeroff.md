# Аналоги модели «nomeroff_lpd» для задачи «LP Detection — детекция номерного знака (+ опц. 4 угла)»

> **См. также:** спецификация исходной модели — [nomeroff_lpd.md](../about_models/nomeroff_lpd.md) · реестр моделей — [models.md](../../models.md) · исследование датасетов-аналогов (та же задача) — [02_lpd_autoria.md](../dataset_research/02_lpd_autoria.md) · сводка исследования моделей — [00_SUMMARY.md](00_SUMMARY.md)

> Документ подготовлен в рамках исследования открытых моделей-аналогов стека CARS.
> **CARS — коммерческий продукт.** Лицензионный статус каждого кандидата трактуется консервативно: пока коммерческое использование не подтверждено первоисточником явно, оно считается **запрещённым/неясным (restricted/unclear)** и помечается как блокер. Любое включение модели/весов в прод-сборку или дистрибуцию TensorRT-engine требует отдельного legal review.

---

## Контекст и требования

**Целевая модель — nomeroff_lpd** (Nomeroff Net, YOLO11x-pose, чекпойнт `yolov11x-keypoints-2026-01-21.pt`). В пайплайне CARS она работает как **детектор номерного знака** на кропе транспортного средства: на вход поступает изображение авто, на выходе — bounding box пластины **плюс 4 угловые keypoints** (углы прямоугольника номера). Эти 4 угла критичны: по ним выполняется **перспективное выравнивание (warp)** наклонённой/повёрнутой пластины в фронтальный прямоугольник перед OCR. Без выравнивания качество распознавания символов падает, особенно на бортовом POV, где номер виден под углом. Это **самая тяжёлая модель стека CARS** (YOLO11x-pose) — и именно её вес/латентность задают потолок бюджета на Jetson Orin Nano.

Рантайм — бортовой: **Jetson Orin Nano 8GB**, исполнение через **ONNX→TensorRT/DeepStream (FP16/INT8)**. Поскольку LPD-ветка стоит после PGIE-детектора и работает на каждом кропе ТС, её латентность и память напрямую конкурируют за ресурс с PGIE, SGIE-классификаторами и OCR. Тяжесть YOLO11x здесь — главный edge-риск: облегчение этой модели (на лёгкий YOLOv5/YOLOv9-tiny-класс) сняло бы основной бюджетный риск, **если** не потерять при этом keypoints и не уронить recall на RU/UA.

Целевой сценарий — **бортовой POV** (камера в движущемся авто, вид с уровня дороги, дистанции 5–50 м: контроль доступа, парковка, патруль, логистика) с режимами **day/night/IR** и регионом **RU/UA** (ГОСТ-Р 50577, кириллические/латинские пластины СНГ). Это центральный domain-фактор задачи: пластина — регионально-специфичный объект, и детектор/keypoint-голова, обученные на US (LPDNet) или Китае (CCPD), показывают **доказанный провал recall** на RU/UA. Так, легаси-LPDNet в CARS уже отвергнут с **FAIL R=0.296** против **R=0.922** у nomeroff на одном срезе (num_gt=385). Сама nomeroff_lpd обучена на AUTO.RIA (RU/UA) и измерена нами **P=0.91 / R=0.92** на num_gt=385 — это причина, по которой она остаётся сильнейшей по точности, несмотря на тяжесть YOLO11x и проблемную лицензию.

**Лицензия — жёсткое требование.** Сам Nomeroff Net — код **GPL-3.0**, а лицензия весов неясна; это уже блокер для проприетарного продукта, и аналог нужен не только под edge, но и под чистую коммерцию. AGPL/GPL/NC/EULA/отсутствие лицензии — блокеры.

**Что требуется от аналога:**
1. **Детекция пластины** — bbox одного класса «номер» с приемлемым recall на бортовом POV.
2. **Опционально, но крайне желательно — 4 угла/keypoints пластины** для перспективного warp перед OCR (ключевая функция nomeroff_lpd; bbox-only аналог не закрывает её полностью).
3. **Коммерчески-чистая лицензия на код И веса** (Apache-2.0/MIT/BSD); NC/AGPL/GPL/EULA/отсутствие лицензии — блокеры.
4. **Edge-бюджет под Orin Nano 8GB** — лёгкий вариант с реальным путём ONNX→TensorRT (FP16/INT8); приоритет — облегчить тяжёлый YOLO11x.
5. **Нативный ONNX или штатный экспорт в ONNX** — для интеграции в DeepStream/TensorRT.
6. Доменная близость к RU/UA ГОСТ-Р либо пригодность как лицензионно-чистая база для дообучения на AUTO.RIA.

> **Важно по измерениям и домену.** Точность всех аналогов ниже приведена **по заявлению авторов и нами не измерена**; recall на кириллических RU/UA-пластинах **ни у одного кандидата не верифицирован нами**. Все найденные открытые веса обучены на US, Китае (CCPD) или global/западном Roboflow — **ни один не таргетирует RU/UA ГОСТ-Р 50577**. Бортовой POV (уровень дороги, 5–50 м) ближе к этим датасетам, чем вид трафик-камеры сверху, но это не компенсирует региональный разрыв в формате пластин.

---

## Обзор аналогов

### open-image-models LicensePlateDetector (yolo-v9-t / yolo-v9-s, end2end) — ankandrew
**актуальный / YOLOv9 (one-stage detector, end2end с встроенным NMS) / семейство t (tiny) / s (small), .onnx лёгкие (tiny — единицы МБ; число параметров автор не публикует), входы 256/384/416/512/608/640 / Лицензия: КОД MIT (подтверждено), ВЕСА — отдельной лицензии НЕТ + архитектура/обучающий код YOLOv9 из WongKinYiu/yolov9 = GPL-3.0 (комм.: UNCLEAR — блокер до legal review) / Форматы: ONNX нативно (готовые .onnx, end2end)→TensorRT / Самый лёгкий кандидат, но только bbox**

YOLOv9 one-stage детектор пластины, экспортируемый сразу в ONNX (end2end, со встроенным NMS). Используется как дефолтный детектор в `fast-alpr`. Подтверждённые варианты (из документации проекта): `yolo-v9-t-256/384/416/512/640-license-plate-end2end`, `yolo-v9-s-608-license-plate-end2end`. Вход: RGB фиксированного разрешения по варианту; выход: **только bbox пластины + conf** (один класс), 4 угла/keypoints **не выдаёт**.

Плюсы:
- Готовые ONNX-веса, прямой путь ONNX→TensorRT (стандартный YOLOv9-граф, поддерживается trtexec), минимальная обвязка (`pip install open-image-models`).
- **Самый лёгкий по latency/памяти кандидат** — tiny-варианты на низком разрешении (256–416) снимают главный риск CARS (тяжесть YOLO11x на Orin Nano).
- Код под **чистым MIT** (подтверждено первоисточником); активный проект, несколько разрешений входа под edge-бюджет.

Минусы:
- **НЕ выдаёт 4 угла** (только bbox, end2end) — теряется перспективное выравнивание, ключевая функция nomeroff_lpd.
- **БЛОКЕР ЛИЦЕНЗИИ:** лицензия весов НЕ оформлена явно (подтверждено WebFetch — отдельной лицензии на .onnx ни в README, ни в карточке нет); при этом архитектура/обучающий код YOLOv9 происходят из **WongKinYiu/yolov9, чей LICENSE.md = GPL-3.0** (ВЕРИФИЦИРОВАНО первоисточником) → возможное copyleft-загрязнение определения модели/весов. **Нельзя верить ярлыку «MIT» на репозитории применительно к весам** → unclear до legal review.
- **Domain gap к RU/UA НЕ закрыт:** обучающий датасет в открытой документации даже не указан явно (WebFetch не подтвердил источник — предположение «Roboflow/западный» снято как факт); RU/UA ГОСТ-Р не заявлен, recall на кириллице **не измерен нами**.

Применимость в пайплайне CARS: кандидат на лёгкий **bbox-детектор пластины**, ЕСЛИ CARS согласится отказаться от keypoint-выравнивания ИЛИ добавить отдельную голову/модель углов. Перед использованием — обязательный legal review лицензии весов + GPL-3.0-происхождения YOLOv9, и измерение recall на RU/UA-срезе.
Edge / Jetson Orin Nano: очень хорошо — tiny-варианты на низком разрешении дают минимальную latency и память; FP16/INT8 TensorRT-engine реалистичен. Прямая противоположность тяжёлому YOLO11x. **На устройстве нами не измерено.**

### we0091234 Chinese_license_plate_detection_recognition (plate_detect)
**актуальный / YOLOv5 (производная ultralytics/yolov5) с bbox + 4 угловыми landmark + warp / YOLOv5-класса n/s, компактный .pt/.onnx (точное число параметров в README не зафиксировано; на порядок легче YOLO11x) / Лицензия: КОД GPL-3.0 (ВЕРИФИЦИРОВАНО footer репо), база ultralytics/yolov5 = AGPL-3.0; ВЕСА наследуют GPL-контекст (комм.: НЕТ) / Форматы: PyTorch .pt + ONNX (onnx_infer.py; .onnx на Baidu Cloud)→TensorRT / Архитектурный близнец nomeroff_lpd, но домен CCPD и copyleft**

YOLOv5-детектор с bbox пластины + **4 угловыми keypoints** (как лицевые landmarks в YOLOv5-face) и перспективным выравниванием (`result_warp`). Это **прямой архитектурный аналог nomeroff_lpd** (bbox+4 угла), но на YOLOv5 вместо YOLO11. Один из самых популярных открытых LPDR-стеков для китайских пластин. Вход RGB (типично 640); выход: bbox + conf + 4 угла (нормированные) → warp. Прим.: README (WebFetch) явно перечислил bbox/recognition; угловые точки задекларированы в коде/демо репо (warp), но в выдержке README прямо не процитированы.

Плюсы:
- **Повторяет контракт nomeroff_lpd:** bbox + 4 угла + warp (архитектурно) — соответствует keypoint-требованию CARS.
- Лёгкая YOLOv5-архитектура с ONNX и прямым путём в TensorRT — снимает риск тяжести YOLO11x; лёгкая keypoint-голова почти не добавляет стоимости.
- Готовый код keypoint-головы и перспективного выравнивания — **хорошая БАЗА для дообучения** под RU/UA.

Минусы:
- **GPL-3.0** (ВЕРИФИЦИРОВАНО first-party — footer репо «GPL-3.0 license») + база **ultralytics/yolov5 = AGPL-3.0** в актуальных версиях → жёсткий **copyleft-блокер** для проприетарного продукта.
- Обучено на **CCPD (Китай)** — низкий ожидаемый recall на RU/UA без дообучения; кириллица не покрыта.
- Веса .pt в репо + .onnx **только на Baidu Cloud** (неудобный доступ), лицензия весов явно не оформлена.

Применимость в пайплайне CARS: **лучший архитектурный ориентир/база** для собственного RU/UA keypoint-детектора (дообучение на AUTO.RIA), НО как готовая модель для коммерции **заблокирован GPL-3.0**. Использовать код как референс реализации keypoint+warp; веса — только для эксперимента/baseline.
Edge / Jetson Orin Nano: хорошо — YOLOv5 n/s намного легче YOLO11x, FP16/INT8 engine реалистичен. **На устройстве нами не измерено.**

### PaddleOCR text detection (PP-OCRv5 det / DBNet) как детектор области пластины
**актуальный / DBNet-стиль детектор ТЕКСТА (сегментация текстовых регионов, не LP-specific) / PP-OCRv5 mobile det — единицы МБ; server det тяжелее / Лицензия: Apache-2.0 на КОД И ВЕСА (ВЕРИФИЦИРОВАНО на карточке) (комм.: ДА) / Форматы: PaddlePaddle нативно; paddle2onnx→ONNX→TensorRT (готовые ONNX-сборки есть) / Единственный чисто-permissive кандидат, но детектор текста, не пластины**

DBNet-стиль детектор текста, выдающий **полигоны текстовых областей** (`dt_polys`, 4+ точки контура текста). Не специфичен для пластин. Вход RGB (динамическое разрешение, кратное 32); выход: полигоны/quad-боксы **текстовых** регионов (`dt_polys` + `dt_scores`), не пластины как объекта. Углы текста ≠ углы пластины, но дают quad для warp текста.

Плюсы:
- **ЧИСТАЯ Apache-2.0 на код И веса** (ВЕРИФИЦИРОВАНО на карточке PP-OCRv5_mobile_det) — **единственный полностью коммерчески-чистый кандидат списка**.
- Готовый ONNX→TensorRT-путь (paddle2onnx), лёгкий mobile-вариант, выдаёт quad-полигоны (потенциал для warp текста).

Минусы:
- Это **детектор ТЕКСТА, а не пластины** (ВЕРИФИЦИРОВАНО: general text detector): нет класса «пластина», много ложных срабатываний на дорожном тексте/вывесках/знаках.
- **Углы текста ≠ 4 угла пластины**; для бортового POV (5–50 м) мелкий текст слабо детектируется без предварительного кропа.

Применимость в пайплайне CARS: **не прямая замена** LP-детектора. Возможна **вспомогательная роль** — поиск текстового quad внутри уже кропнутой пластины для тонкого warp перед OCR, либо чистый по лицензии fallback-детектор текста. Полноценный LP-детектор с 4 углами не обеспечивает.
Edge / Jetson Orin Nano: mobile-det очень лёгкий, подходит (FP16); server-det тяжелее — выбирать mobile. **На устройстве нами не измерено.**

### xialuxi/yolov5-car-plate (детектор пластины с угловыми точками)
**актуальный / YOLOv5 с bbox + угловые keypoints (车牌角点检测) + warp (检测矫正) / YOLOv5-класса n/s, лёгкий (число параметров в README не зафиксировано) / Лицензия: dual GPL-3.0 + Apache-2.0 (ВЕРИФИЦИРОВАНО «licenses found» на GitHub), база ultralytics/yolov5 = AGPL-3.0 (комм.: НЕТ) / Форматы: PyTorch .pt; YOLOv5→ONNX→TensorRT штатный, но ONNX в репо не гарантирован / Альтернативный keypoint-референс, но copyleft + CCPD + недообучение**

YOLOv5 с детекцией bbox + угловые keypoints пластины и warp (ВЕРИФИЦИРОВАНО WebFetch: corner point detection + warping). Архитектурно близок к nomeroff_lpd и we0091234. Обучение на части CCPD (~десяток эпох по README); веса на Baidu Cloud (пароль `vd4j` — ВЕРИФИЦИРОВАНО). Вход RGB (640); выход: bbox + угловые точки → перспективная коррекция.

Плюсы:
- Выдаёт **угловые точки пластины + warp** (ВЕРИФИЦИРОВАНО) — соответствует keypoint-требованию CARS.
- Лёгкая YOLOv5-база, edge-friendly; код keypoint-головы доступен как референс.

Минусы:
- **Смешанная лицензия с GPL-3.0 компонентом** (ВЕРИФИЦИРОВАНО first-party) + YOLOv5/Ultralytics **AGPL-база** → copyleft-блокер коммерции.
- Обучено на части CCPD всего **~десяток эпох** (по README) → слабая модель; RU/UA не покрыт; ONNX-экспорт не из коробки.
- Веса **только на Baidu Cloud** (неудобный доступ).

Применимость в пайплайне CARS: альтернативный архитектурный референс keypoint+warp (наряду с we0091234). Как готовая коммерческая модель **заблокирован** лицензией; пригоден лишь как код-референс/baseline для дообучения под RU/UA (с чистой базы).
Edge / Jetson Orin Nano: хорошо — лёгкий YOLOv5, edge-friendly. **На устройстве нами не измерено.**

### WPOD-NET (Warped Planar Object Detection)
**2018 (ECCV) / полносвёрточная сеть (FCN), 8-канальная feature map: object/non-object + аффинное преобразование → 4 угла пластины / компактная FCN, файл весов единицы МБ (параметры не публикуются; легче любого YOLO11x) / Лицензия: оригинал sergiomsilva/alpr-unconstrained — UNCLEAR (Unknown + Darknet-компоненты); PyTorch-порт Pandede = GPL-3.0 (ВЕРИФИЦИРОВАНО) (комм.: НЕТ) / Форматы: Keras+TF/Darknet (оригинал) или PyTorch (порт); ONNX НЕ из коробки / Нативно даёт 4 угла, но нет ONNX-пути и copyleft/unclear**

Полносвёрточная сеть, регрессирующая 4 угла пластины через аффинное преобразование → прямое перспективное выравнивание. Семантически = задача nomeroff_lpd keypoints. Оригинал: `sergiomsilva/alpr-unconstrained` (Keras/Darknet, веса через `get-networks.sh`). PyTorch-порт: `Pandede/WPODNet-Pytorch` (веса `wpodnet.pth` в **GitHub Releases v1.0.0** — ВЕРИФИЦИРОВАНО, не только Baidu). Вход RGB (адаптивное ~208–608); выход: вероятность пластины + аффинная матрица → 4 угла для warp.

Плюсы:
- **Нативно решает задачу 4 углов/warp** (аффинная регрессия) — концептуально = keypoint-голова nomeroff_lpd.
- Лёгкая FCN, ориентирована на «unconstrained»/наклонные пластины разных стран.
- Хорошо изучена (ECCV 2018), много реализаций; веса порта доступны на GitHub Releases.

Минусы:
- **Лицензия оригинала НЕЯСНА** (Unknown + Darknet-компоненты), PyTorch-порт **GPL-3.0** (ВЕРИФИЦИРОВАНО first-party) → блокер коммерции по обоим путям.
- **Нет готового ONNX/TensorRT-пути** (подтверждено WebFetch) — требует ручного экспорта Keras→ONNX или PyTorch→ONNX + реализации аффинного декодера; edge-интеграция нетривиальна.
- Архитектура **2018 года**, точность ниже современных детекторов; RU/UA не подтверждён.

Применимость в пайплайне CARS: концептуальный референс для перспективного warp; как готовая коммерческая модель не годится (лицензия unclear/GPL + нет ONNX-пути). Только как алгоритмический baseline для углов.
Edge / Jetson Orin Nano: лёгкая FCN потенциально помещается в бюджет, НО готового ONNX/TensorRT-пути нет — требует ручного экспорта и аффинного декодера; интеграция нетривиальна. **Не измерено нами.**

### NVIDIA LPDNet (TAO / NGC) — DetectNet_v2 и YOLOv4-tiny варианты
**актуальный TAO / DetectNet_v2 (ResNet18) и YOLOv4-tiny (cspdarknet_tiny) (ВЕРИФИЦИРОВАНО docs.nvidia.com) / pruned-модели компактные, оптимизированы под Jetson (число параметров варьируется по версии) / Лицензия: НЕ SPDX, а NVIDIA Governing Terms / Model EULA; «ready for commercial use» заявлено, но редистрибуция/модификация весов и engine ограничены (комм.: UNCLEAR) / Форматы: TAO Toolkit, .etlt/.onnx, нативно ONNX→TensorRT + DeepStream SGIE / Лучший edge-путь, но доказанный FAIL по recall на RU/UA и bbox-only**

Два бэкбона: DetectNet_v2 (ResNet18) и YOLOv4-tiny (cspdarknet_tiny). Версии под US (NVIDIA-owned датасет) и под CCPD (Китай). Pruned-варианты заточены под edge. **Именно эту модель CARS уже использовал и заменил** из-за низкого recall на не-US пластинах. Вход RGB (cropped color image); выход: **только bbox + label `lpd`** (один класс, ВЕРИФИЦИРОВАНО — «return a box around each object», без keypoints). 4 угла **не выдаёт**.

Плюсы:
- **Идеальный edge/DeepStream/TensorRT-путь** (родная NVIDIA-экосистема, готовый engine, без ручного экспорта).
- Заявлено «ready for commercial use», есть unpruned-веса для дообучения под RU/UA через TAO.

Минусы:
- **Уже отвергнут в CARS:** катастрофически низкий recall на RU/UA (**R≈0.30, FAIL** — задокументировано в спеке/metrics.json) против R=0.922 у nomeroff.
- Только bbox (**нет 4 углов**); лицензия — **NVIDIA Governing Terms / Model EULA, НЕ permissive SPDX:** редистрибуция/модификация весов и engine ограничены → коммерческий деплой в дистрибутиве подтверждать с legal (unclear).

Применимость в пайплайне CARS: как готовая модель для RU/UA **не подходит** (доказанный FAIL). Потенциал только в дообучении unpruned-варианта через TAO на AUTO.RIA с деплоем в DeepStream — но это новая работа, не готовый аналог, и при условии сверки условий NVIDIA EULA на редистрибуцию.
Edge / Jetson Orin Nano: отлично — спроектировано NVIDIA под Jetson/DeepStream/TensorRT FP16/INT8, pruned-веса лёгкие, готовый engine-путь. Но это не компенсирует FAIL по recall.

### keremberke YOLOv5 license-plate (n/s/m)
**актуальный / YOLOv5 (n/s/m), детектор bbox одного класса «license-plate» / n ~1.9M … m ~21M параметров, edge-friendly / Лицензия: в YAML README поле license ОТСУТСТВУЕТ (ВЕРИФИЦИРОВАНО) → «все права защищены» по умолчанию; база YOLOv5 = AGPL-3.0 (комм.: НЕТ) / Форматы: PyTorch .pt (yolov5-библиотека); ONNX не приложен, но YOLOv5 штатно экспортируется→TensorRT / Лёгкий bbox-baseline, двойной лицензионный блокер**

YOLOv5 (n/s/m) детектор bbox одного класса. База — ultralytics/yolov5 (ВЕРИФИЦИРОВАНО: tags yolov5, library_name yolov5). Обучено на `keremberke/license-plate-object-detection` (~8.83k изображений). Вход 640; выход: bbox + score + class, **без keypoints**. Заявлено mAP@0.5≈0.988 (self-reported в model-index YAML, на собственном val — **не измерено нами**).

Плюсы:
- Готовые веса на HuggingFace, лёгкие варианты n/s, штатный ONNX→TensorRT-путь YOLOv5.
- Высокий заявленный mAP, простой контракт (один класс).

Минусы:
- **ВЕРИФИЦИРОВАНО: лицензия модели НЕ указана** (YAML без поля license) → по умолчанию «все права защищены», нет разрешения на коммерческое использование; плюс база YOLOv5 = **AGPL-3.0** (нужна Enterprise-лицензия Ultralytics) → **двойной блокер**.
- Только bbox (**нет 4 углов**), RU/UA-домен не покрыт (датасет общего/западного характера, регион не указан), метрики self-reported и не верифицированы.

Применимость в пайплайне CARS: лёгкий bbox-baseline для экспериментов, но для коммерческого CARS **заблокирован** (нет лицензии + AGPL-база). Не закрывает keypoint-требование.
Edge / Jetson Orin Nano: хорошо — YOLOv5 n/s крайне лёгкие, FP16/INT8 без проблем. **Не измерено нами.**

### morsetechlab/yolov11-license-plate-detection
**актуальный / YOLO11 (Ultralytics), варианты n..x, bbox-детектор одного класса (та же база YOLO11, что nomeroff_lpd, но БЕЗ pose/keypoint-головы) / n ~2.6M … x ~57M параметров; n/s edge-friendly / Лицензия: AGPL-3.0 ЯВНО в метаданных карточки (ВЕРИФИЦИРОВАНО) + база Ultralytics AGPL-3.0 (комм.: НЕТ) / Форматы: PyTorch .pt + ONNX (приложен)→TensorRT; DeepStream-совместимо / YOLO11-bbox, но AGPL и завышенные метрики**

YOLO11 (n..x) bbox-детектор одного класса. Та же база, что nomeroff_lpd, но **без keypoint-головы**. Веса .pt + .onnx (ВЕРИФИЦИРОВАНО). Обучено на Roboflow License Plate Recognition (~10.1k изображений, 300 epochs, 640, A100). Вход 640; выход: bbox + conf, **без 4 углов**. Метрики (x): P=0.989, R=0.951, mAP50=0.981, mAP50-95=0.726 — но автор **ЯВНО отмечает train/test contamination** в upstream-датасете → завышены (ВЕРИФИЦИРОВАНО: «reported metrics are overestimated», планируется v2 с честными метриками).

Плюсы:
- Готовые .pt и .onnx, варианты под бюджет (n..x), штатный TensorRT-путь.
- Та же база YOLO11, что и текущая модель — лёгкая миграция инференса.

Минусы:
- **AGPL-3.0 явно на весах** (ВЕРИФИЦИРОВАНО first-party) + Ultralytics AGPL → жёсткий **copyleft-блокер** (нужна платная Enterprise-лицензия Ultralytics).
- Только bbox (**нет 4 углов**), метрики завышены (train/test contamination, подтверждено автором), RU/UA не покрыт.

Применимость в пайплайне CARS: **не годится** для коммерческого CARS (AGPL-3.0). Только как ориентир, что YOLO11-детектор пластины доступен; keypoint-требование не закрывает.
Edge / Jetson Orin Nano: хорошо для лёгких вариантов (n/s); FP16/INT8 реалистичен. x-вариант так же тяжёл, как nomeroff. **Не измерено нами.**

---

## Сводная таблица аналогов

| Модель | Год | Архитектура | Параметры/размер | Лицензия (комм.?) | Форматы (ONNX/TRT) | Edge-fit (Orin Nano) | Домен-fit | Вердикт |
|---|---|---|---|---|---|---|---|---|
| open-image-models (yolo-v9-t/s) | актуальный | YOLOv9 one-stage, end2end (NMS встроен) | t/s, .onnx единицы МБ | КОД MIT; ВЕСА unclear + YOLOv9 из GPL-3.0 → **UNCLEAR (блокер)** | ONNX нативно→TensorRT | Очень хорошо (самый лёгкий) | RU/UA не закрыт, датасет не указан, **только bbox** | **secondary** |
| we0091234 plate_detect | актуальный | YOLOv5 + 4 угла + warp | n/s, компактный | КОД GPL-3.0 + YOLOv5 AGPL → **НЕТ** | ONNX (Baidu)→TensorRT | Хорошо | CCPD (Китай), RU/UA нет; даёт 4 угла | **marginal** (база/референс) |
| PaddleOCR PP-OCRv5 det (DBNet) | актуальный | DBNet детектор ТЕКСТА (не LP) | mobile единицы МБ | Apache-2.0 код+веса → **ДА** | paddle2onnx→ONNX→TensorRT | Хорошо (mobile) | Текст, не пластина; quad текста ≠ 4 угла | **marginal** (вспомог.) |
| xialuxi/yolov5-car-plate | актуальный | YOLOv5 + угловые точки + warp | n/s, лёгкий | dual GPL-3.0+Apache + YOLOv5 AGPL → **НЕТ** | ONNX не из коробки→TensorRT | Хорошо | CCPD, ~десяток эпох, RU/UA нет; даёт 4 угла | **marginal** (референс) |
| WPOD-NET | 2018 | FCN, аффинная регрессия 4 углов | компактная FCN, единицы МБ | оригинал UNCLEAR; порт GPL-3.0 → **НЕТ** | ONNX НЕ из коробки | Лёгкая, но нет ONNX-пути | unconstrained/мульти-страна; RU/UA нет; даёт 4 угла | **not_recommended** |
| NVIDIA LPDNet (TAO) | актуальный | DetectNet_v2 R18 / YOLOv4-tiny | pruned, лёгкие | NVIDIA Model EULA (не SPDX) → **UNCLEAR** | .etlt/.onnx→TensorRT/DeepStream (лучший путь) | Отлично | US/CCPD, **доказанный FAIL R≈0.30 на RU/UA**, только bbox | **not_recommended** |
| keremberke YOLOv5 (n/s/m) | актуальный | YOLOv5 bbox, 1 класс | n ~1.9M … m ~21M | license отсутствует + YOLOv5 AGPL → **НЕТ** | .pt; ONNX штатный→TensorRT | Хорошо | global/западный, RU/UA нет, только bbox | **not_recommended** |
| morsetechlab YOLO11 LP | актуальный | YOLO11 bbox, 1 класс | n ~2.6M … x ~57M | AGPL-3.0 явно + Ultralytics AGPL → **НЕТ** | .pt + ONNX→TensorRT | Хорошо (n/s) | Roboflow global, RU/UA нет, только bbox, метрики завышены | **not_recommended** |

---

## Рекомендации

**ГЛАВНЫЙ ВЫВОД (валидный отрицательный результат, подтверждён адверсариальной проверкой по первоисточникам).** Для edge-домена CARS **НЕТ открытого предобученного аналога**, который одновременно (а) лицензионно чист для коммерции, (б) выдаёт 4 угла пластины и (в) обучен на RU/UA ГОСТ-Р. Все три свойства вместе не встречаются ни у одного кандидата. Это объясняет, почему nomeroff_lpd (AUTO.RIA RU/UA, измеренные нами P=0.91/R=0.92 на num_gt=385) остаётся сильнейшим по точности, несмотря на тяжесть YOLO11x и GPL-3.0/неясную лицензию весов.

**Карта свойств (что чем закрывается):**
- **4 угла/keypoints (как у nomeroff_lpd):** дают только `we0091234` (GPL-3.0, CCPD), `xialuxi/yolov5-car-plate` (dual GPL+Apache, AGPL-база, CCPD) и `WPOD-NET` (unclear/GPL, нет ONNX). **ВСЕ заблокированы лицензией или доменом.**
- **Чистая коммерческая лицензия:** только `PaddleOCR det` (Apache-2.0, подтверждено) — но это детектор **текста**, не пластины, без класса «пластина»/4 углов. `open-image-models` — MIT-код, но веса unclear (YOLOv9 из GPL-3.0 + лицензия весов не оформлена) и только bbox.
- **Edge/TensorRT-путь:** лучший у `NVIDIA LPDNet` (DeepStream/TAO) и `open-image-models` (ONNX из коробки) — но LPDNet уже дал FAIL по recall на RU/UA (R≈0.30), а open-image-models только bbox.

**Лицензии-ловушки (подтверждено адверсариально по первоисточникам):**
- **Ultralytics YOLO (v5/v8/v11) = AGPL-3.0** → `keremberke` (нет лицензии в YAML + AGPL-база), `morsetechlab` (AGPL явно), `we0091234`/`xialuxi` (YOLOv5-база) заблокированы; платная Enterprise-лицензия Ultralytics обязательна для проприетарного продукта.
- **open-image-models:** код MIT, НО архитектура/код YOLOv9 из `WongKinYiu/yolov9 = GPL-3.0` (подтверждено LICENSE.md), а отдельная лицензия на .onnx-веса не заявлена → «unclear до legal review»; **не верить ярлыку «MIT» применительно к весам**.
- **NVIDIA LPDNet:** НЕ SPDX, а NVIDIA Governing Terms/Model EULA; «ready for commercial use» заявлено, но редистрибуция/модификация весов и engine ограничены — деплой подтверждать с legal (unclear).
- **PaddleOCR:** Apache-2.0 на код И веса (подтверждено) — единственный чисто-permissive кандидат, но не LP-детектор.

**Рекомендуемый путь для CARS (3 опции, по убыванию реалистичности):**
1. **ДООБУЧЕНИЕ собственной keypoint-головы под RU/UA на AUTO.RIA**, взяв за архитектурный референс `we0091234`/`xialuxi` (bbox+4 угла+warp), но обучая с **ЧИСТОЙ базы** детектора (НЕ Ultralytics-AGPL и НЕ WongKinYiu-GPL без Enterprise/legal), затем экспорт ONNX→TensorRT. Закрывает И домен, И keypoints — но требует лицензионно-чистой базы. Вопрос обучающих данных разобран в [исследовании датасетов-аналогов](../dataset_research/02_lpd_autoria.md).
2. **Если отказаться от keypoints в LP Detection** и делать выравнивание отдельно: лёгкий bbox-детектор `open-image-models` (ТОЛЬКО после legal review лицензии весов + GPL-3.0-происхождения YOLOv9) + Apache-2.0 `PaddleOCR det` для quad-warp текста внутри кропа. Снимает риск тяжести YOLO11x. Чистейший по лицензии компонент здесь — PaddleOCR (Apache-2.0).
3. **Платная Enterprise-лицензия Ultralytics** → легализует `morsetechlab` YOLO11 / собственный YOLO11-pose (как у nomeroff) — но это платный путь, а не открытый аналог.

> **Что искали и не нашли:** открытого предобученного LP-детектора с 4 углами, обученного именно на RU/UA/СНГ-пластинах, под permissive-лицензией (MIT/Apache/BSD) — **не существует** в открытом доступе на момент проверки (HF/GitHub/NGC/OpenVINO/Paddle). RetinaPlate как самостоятельный публичный проект с доступными весами под чёткой лицензией подтвердить **не удалось** (упоминается в литературе, открытых весов не найдено) — в список не включён, чтобы не выдумывать. Ни один из 8 включённых аналогов не оказался выдуманным — все репозитории/карточки реально существуют и проверены WebFetch (единственное, что не вскрылось напрямую — JS-рендеренная NGC-карточка LPDNet, подтверждена косвенно через docs.nvidia.com).

---

## Ссылки

- open-image-models — GitHub: https://github.com/ankandrew/open-image-models
- fast-alpr — GitHub: https://github.com/ankandrew/fast-alpr
- open-image-models — PyPI: https://pypi.org/project/open-image-models/
- WongKinYiu/yolov9 — GitHub: https://github.com/WongKinYiu/yolov9
- WongKinYiu/yolov9 — LICENSE.md (GPL-3.0): https://github.com/WongKinYiu/yolov9/blob/main/LICENSE.md
- we0091234 Chinese_license_plate_detection_recognition — GitHub: https://github.com/we0091234/Chinese_license_plate_detection_recognition
- we0091234 — DeepWiki (типы пластин): https://deepwiki.com/we0091234/Chinese_license_plate_detection_recognition/1.1-license-plate-types
- CCPD датасет — GitHub: https://github.com/detectRecog/CCPD
- PaddleOCR PP-OCRv5_mobile_det — HuggingFace: https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det
- PaddleOCR PP-OCRv5_server_det — HuggingFace: https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det
- PaddleOCR ONNX-конверсия — HuggingFace (monkt): https://huggingface.co/monkt/paddleocr-onnx
- xialuxi/yolov5-car-plate — GitHub: https://github.com/xialuxi/yolov5-car-plate
- WPOD-NET оригинал (alpr-unconstrained) — GitHub: https://github.com/sergiomsilva/alpr-unconstrained
- WPODNet-Pytorch (порт) — GitHub: https://github.com/Pandede/WPODNet-Pytorch
- WPOD-NET — paper (ECCV 2018): https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf
- NVIDIA LPDNet — NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lpdnet
- NVIDIA LPDNet — TAO docs: https://docs.nvidia.com/tao/archive/5.3.0/text/model_zoo/cv_models/lpdnet.html
- NVIDIA NGC — legal/terms: https://ngc.nvidia.com/legal/terms
- NVIDIA — блог по real-time LPD/LPR: https://developer.nvidia.com/blog/creating-a-real-time-license-plate-detection-and-recognition-app/
- keremberke yolov5m-license-plate — HuggingFace: https://huggingface.co/keremberke/yolov5m-license-plate
- keremberke yolov5n-license-plate — HuggingFace: https://huggingface.co/keremberke/yolov5n-license-plate
- keremberke license-plate-object-detection (датасет) — HuggingFace: https://huggingface.co/datasets/keremberke/license-plate-object-detection
- morsetechlab/yolov11-license-plate-detection — HuggingFace: https://huggingface.co/morsetechlab/yolov11-license-plate-detection
- Roboflow License Plate Recognition (датасет) — Universe: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e

---

## История изменений
- 2026-06-05 — создан в рамках исследования открытых моделей-аналогов стека CARS (многоагентный workflow, веб-поиск по первоисточникам, лицензии перепроверены адверсариально).
