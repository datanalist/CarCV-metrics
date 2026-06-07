# Аналоги модели «VehicleMakeNet» для задачи «Make — классификация марки (бренда) ТС»

> **См. также:** спецификация исходной модели — [vehiclemakenet.md](../about_models/vehiclemakenet.md) · реестр моделей — [models.md](../../models.md) · исследование датасетов-аналогов (та же задача) — [06_make_vmmrdb.md](../dataset_research/06_make_vmmrdb.md) · сводка исследования моделей — [00_SUMMARY.md](00_SUMMARY.md)

> ⚠️ **CARS — КОММЕРЧЕСКИЙ продукт.** Лицензионный статус каждого кандидата — определяющий критерий. По умолчанию лицензия считается `restricted/unclear`, пока обратное не подтверждено первоисточником. `unclear`-лицензия трактуется как **блокер для коммерции до legal review**. Все метрики аналогов — **«по заявлению автора, нами НЕ измерено»**, если явно не сказано иное. Факты ниже перепроверены адверсариально через первоисточники на 2026-06.

## Контекст и требования

**Целевая модель — VehicleMakeNet** (NVIDIA TAO, NGC): классификатор марки (бренда) ТС, backbone **ResNet-18 (pruned)**, вход **crop ТС 224×224×3 BGR** (offsets `(104, 117, 124)`, **без `/255`**, NCHW), выход — логиты **20 марок US/EU-рынка**. В конвейере CARS это **SGIE1** (Secondary GIE): работает поверх кропов ТС, выданных детектором TrafficCamNet (PGIE), и заполняет атрибут «марка». Целевой рантайм — **ONNX → TensorRT/DeepStream** на NVIDIA Jetson Orin Nano 8GB; вес baseline-ONNX ≈ 7.4 MB (FP32-граф, экспорт в FP16/INT8 при сборке engine на Jetson).

На реальных RU-данных baseline **проваливается**: на суррогатном in-dist-срезе mad-cars (700 изображений) измерен **Top-1 0.083** — то есть модель из 20 US/EU-марок практически бесполезна для российского/украинского автопарка (Lada/VAZ, Москвич, GAZ/UAZ, китайские Chery/Haval/Geely в словаре отсутствуют). Это ключевая боль задачи Make в CARS: нужен либо аналог с RU/СНГ-словарём, либо путь дообучения.

Целевой домен CARS — **бортовой POV**: камера с уровня дороги, дистанция 5–50 м, переменный ракурс, day/night/IR. Это принципиально отличается от каталожных/web/US-наборов (VMMRdb, Stanford Cars, huggingpics-scrape), на которых обучены все найденные open-классификаторы, — отсюда сквозной **domain gap**, который фиксируется для каждого кандидата. Коммерческая лицензия КОДА **и весов** (не только обёртки) — обязательное требование; лицензия самого baseline (NVIDIA TAO/NGC Model EULA) тоже требует отдельного legal review (см. [vehiclemakenet.md](../about_models/vehiclemakenet.md) §Лицензия).

**Что требуется от аналога:**
- классификация на уровне **make** (марка/бренд) или хотя бы make/model с возможностью агрегации в марку;
- покрытие нужного словаря, в идеале с **RU/СНГ-марками** (главный пробел baseline);
- доменная близость к бортовому/камерному POV (день/ночь/IR), а не только студийный/web-ракурс;
- формат и путь экспорта **ONNX → TensorRT** под Jetson Orin Nano, реалистичный edge-бюджет (ориентир — ResNet-18 pruned ≈ 7.4 MB);
- **коммерчески разрешённая лицензия на код И веса** (а не только на демо-обёртку), с контролем прав на исходные обучающие фото.

## Обзор аналогов

### timm fine-grained transfer-learning (ConvNeXt / EfficientNet / MobileNetV3 / MobileNetV4)
**2019–2024 / семейство backbone'ов timm (ConvNeXt-T/S, EfficientNet-B0..B3, MobileNetV3-Large, MobileNetV4) + кастомная FC-голова на N марок / от ~5M (MobileNetV3-L / EfficientNet-B0) до ~28M (ConvNeXt-T) — конфигурируется / лицензия: КОД Apache-2.0, ВЕСА — UNCLEAR (комм.: UNCLEAR) / PyTorch + ONNX + TensorRT / отличный edge-fit и единственный путь под RU-словарь — но требует обучения**

Плюсы:
- Полный контроль словаря марок — можно добавить **RU/СНГ-бренды**, которых нет ни в одном найденном готовом open-классификаторе (Lada/VAZ, Москвич, GAZ, UAZ, китайские Chery/Haval/Geely + US/EU baseline).
- Управляемый edge-бюджет (выбор backbone): MobileNetV3-L / EfficientNet-B0 дают размер/latency на уровне или лучше текущего ResNet-18 pruned; **чистый Apache-2.0 КОД** и штатный экспорт ONNX→TensorRT (export-скрипты в репо timm), INT8-калибровка штатная.
- Препроцессинг и домен задаются под бортовой POV (день/ночь/IR) при правильном обучающем датасете — то есть это единственный кандидат, способный реально закрыть domain gap.

Минусы:
- Это **НЕ готовая модель** — требует обучения: легальные данные (дефицит, см. [06_make_vmmrdb.md](../dataset_research/06_make_vmmrdb.md)), вычислительные ресурсы, разметка.
- **ЛОВУШКА ВЕСОВ:** код timm = Apache-2.0, но официальная документация timm **сама предупреждает**, что ImageNet-веса наследуют non-commercial research-ограничение ImageNet («one should assume that the original dataset license applies to the weights… seek legal advice if you intend to use the pretrained weights in a commercial product»), а часть весов (Facebook WSL/SSL/SWSL) — явно **CC-BY-NC 4.0**. Базовый ImageNet-1k-вес для коммерции БЕЗ юридической проверки **не является заведомо чистым** → нужен per-model отбор backbone с реально пермиссивным весом или обучение с нуля.
- Качество целиком зависит от обучающего корпуса; легально чистых RU-make-данных в открытом доступе почти нет (CC BY Roboflow малы/шумны).

Применимость в пайплайне CARS: **рекомендуемый ОСНОВНОЙ путь для RU/СНГ-продакшна** — дообучение make-классификатора под бортовой POV и нужный словарь марок. В CARS — SGIE поверх кропов TrafficCamNet с собственным словарём. Итоговые дообученные веса — собственность проекта. Data-стратегия: [06_make_vmmrdb.md](../dataset_research/06_make_vmmrdb.md).
Edge / Jetson Orin Nano: лучшая управляемость edge-бюджетом. MobileNetV3-L / EfficientNet-B0 — размер/latency на уровне или лучше baseline ResNet-18 pruned; экспорт ONNX→TensorRT FP16/INT8 отработан для CNN. Конкретные цифры **не измерены нами** (зависят от выбранного backbone и обучения).

### Jordo23/vehicle-classifier («Dude, What's My Car?»)
**2024 / EfficientNet-B4 (timm backbone) + FC-голова на 8949 классов / ~19M параметров, .pth ≈135MB, .onnx ≈927KB / MIT на код+веса (комм.: ДА — но риск данных) / PyTorch (.pth) + ONNX (битый, см. минусы) → TensorRT / тяжелее baseline, US/EU make+model+year**

Плюсы:
- Готовый make/model/year-классификатор с заявленной **MIT-лицензией кода+весов** и явным разрешением автора на коммерческое использование («Free for personal and commercial use», подтверждено raw README 2026-06); `.pth` выложен открыто.
- Богатая иерархия make/model/year (**8949 классов**) — агрегацией по первому токену даёт make и расширяет baseline 20 марок (US/EU).
- timm-backbone (EfficientNet-B4) — удобная стартовая точка для дообучения/дистилляции; путь ONNX→TensorRT отработан для CNN (SiLU/Swish поддержан TRT FP16).

Минусы:
- **БЛОКЕР на прямое использование штатного ONNX:** файл `.onnx` ≈927KB **физически не может** содержать полный граф B4 (~19M параметров, ~75MB FP32) и даже одну FC-голову 1792×8949 (~64MB FP32). Почти наверняка экспортирован некорректно/частично/без весов (external-data не приложен) либо это битый/заглушечный ONNX. Использовать штатный `.onnx` нельзя без верификации — реалистичный путь — **переэкспорт из `.pth`** через `torch.onnx.export` (тривиально) и верификация выходов.
- Метрики **Top-1 ~50% / Top-5 ~75–80%** — «по заявлению автора, нами НЕ измерено»; метки make+model+year требуют агрегации в марку через `class_mapping.csv`.
- Нет RU/СНГ-марок (VMMRdb=US) — **главную боль CARS не закрывает** без дообучения; препроцессинг 380×380 RGB (`/255` + mean/std) ≠ TAO BGR/offsets — нужна замена пайплайна препроцессинга.
- Правовой риск исходных VMMRdb-фото MIT-объявлением автора **не снят** → legal review происхождения данных до production.

Применимость в пайплайне CARS: кандидат на замену/дополнение VehicleMakeNet для **US/EU make**: заявленная MIT, выложенный `.pth`. В CARS — SGIE поверх кропов TrafficCamNet с агрегацией 8949→make (после переэкспорта рабочего ONNX). Реалистичный сценарий — взять как стартовые веса и **дообучить на RU-классы** (см. [06_make_vmmrdb.md](../dataset_research/06_make_vmmrdb.md)).
Edge / Jetson Orin Nano: EfficientNet-B4 на 380×380 — заметно тяжелее baseline ResNet-18 pruned (7.4MB). На Orin Nano 8GB как SGIE пройдёт во FP16, но latency выше baseline (**не измерено нами**); для жёсткого realtime — дистилляция/замена backbone на B0/MobileNet. INT8 возможен через калибровку TRT.

### dima806/car_models_image_detection
**2023 / ViT-base-patch16-224 (fine-tune от google/vit-base-patch16-224-in21k) / ~86M параметров, safetensors ≈344MB / Apache-2.0 на код+веса (комм.: ДА) / PyTorch/Transformers + safetensors (ONNX нет) → TensorRT / тяжёлый ViT, US/мировой make+model**

Плюсы:
- **Apache-2.0 на код И веса** — коммерчески чистая лицензия (подтверждена по карточке HF, 2026-06).
- Богатый fine-grained словарь (**323 класса make+model**) шире baseline 20; агрегацией по первому токену даёт make.
- Вход 224×224 совпадает по разрешению с baseline (упрощает интеграцию кропа); заявленная точность ~84% acc («по заявлению автора, нами не измерено»).

Минусы:
- **ИСПРАВЛЕНИЕ ЧЕРНОВИКА:** выход — **323 класса make+MODEL** (по `config.json` id2label: `0: Acura ILX`, `28: BMW 3-Series`, `103: Ford F-150`, `322: smart fortwo`), а **НЕ «392 марки/бренда»**. Это make+model fine-grained, прямой классификации голого make НЕТ — нужна **агрегация** из первого токена (как у Jordo23/cyberlord).
- ViT-base ~86M — **самый тяжёлый** из CNN/ViT-кандидатов; ONNX/TRT-путь для ViT менее предсказуем по latency/INT8, чем CNN; ONNX **не выложен** — нужен собственный экспорт (transformers→ONNX через optimum) и верификация.
- Нет подтверждённого покрытия RU/СНГ-марок (US/мировой набор); студийный/web-домен ≠ бортовой POV.
- Точность ~84% — «по заявлению автора», не измерена нами; per-class разброс заявлен сильный.

Применимость в пайплайне CARS: secondary для US/EU/мирового make+model при наличии бюджета на ViT, с агрегацией в make. В CARS — SGIE-классификатор на 224×224 кропе либо **teacher для дистилляции** в компактный edge-CNN. Для RU — нужен дообуч/расширение классов.
Edge / Jetson Orin Nano: ViT-base ~86M пройдёт во FP16 как вторичный классификатор, но latency существенно выше ResNet-18/EfficientNet-B0 (**не измерено нами**). Для edge желательнее дистилляция в компактный CNN; INT8 для ViT капризнее, чем для CNN.

### therealcyberlord/stanford-car-vit-patch16
**~2023 / ViT-base-patch16-224 (fine-tune от google/vit-base-patch16-224), fine-tuned на Stanford Cars / ~85.8M параметров, FP32 safetensors / Apache-2.0 на код+веса (комм.: ДА — но риск данных Stanford Cars) / PyTorch/Transformers + safetensors (ONNX нет) → TensorRT / тяжёлый ViT, узкий US-набор make+model+year**

Плюсы:
- **Apache-2.0 на веса** — коммерчески пригоден (подтверждена по карточке HF, 2026-06).
- Высокая публичная точность **~86%** на каноническом fine-grained бенчмарке Stanford Cars («по заявлению автора, нами не измерено»); хорошее покрытие US/EU-марок (репо отмечает 20/20 NGC-марок).
- Вход 224×224 как baseline; популярная воспроизводимая модель.

Минусы:
- Метки **make+model+year** (196 классов формата «2012 Tesla Model S», «2012 BMW M3 coupe»), а не голый make — нужна агрегация в марку; всего 196 узких US-классов.
- ViT-base тяжёлый для edge realtime; **ONNX не выложен** (экспорт через optimum).
- Нет RU/СНГ-марок; **устаревший датасет** (нет свежих моделей); каталожный домен (3/4 и боковые ракурсы, нет night/IR) ≠ бортовой POV; **лицензия данных Stanford Cars неясна** (ImageNet-like/research) → правовой риск исходных фото при production (см. [06_make_vmmrdb.md](../dataset_research/06_make_vmmrdb.md)).

Применимость в пайплайне CARS: secondary/benchmark для US-make sanity-check и как готовая база fine-grained признаков. Прямой деплой нецелесообразен (тяжёлый ViT, только US, model-level метки). Полезен как **teacher/контрольный срез**.
Edge / Jetson Orin Nano: ViT-base ~86M — тяжёлый для realtime (как dima806), тот же caveat по ViT INT8/latency; **не измерено нами**.

### Spectrico Vehicle MMR (MobileNet) — free + commercial
**— / MobileNet v2 (TensorFlow .pb) классификатор + отдельный YOLOv3-детектор / компактная MobileNetV2, input 128×128, ~единицы MB .pb / ПРОПРИЕТАРНАЯ — открыты только демо-обёртки (комм.: UNCLEAR → де-факто блокер) / TensorFlow .pb (+MNN в платной); ONNX не предусмотрен вендором / технически отличный edge-fit, но юридически непригоден**

Плюсы:
- Технически отличный edge-fit (MobileNetV2, 128×128, мелкий `.pb`; ~35 ms на CPU i5 «по заявлению вендора»).
- Широкий словарь (**400 make / 7000 model** по заявлению вендора) — на порядок шире baseline 20.
- Демо-обёртки (BSD/MIT) и зеркала `.pb` позволяют быстро собрать **research-прототип**.

Минусы:
- **ЛОВУШКА ЛИЦЕНЗИИ (БЛОКЕР):** сам классификатор MMR — **проприетарный коммерческий продукт Spectrico**. README сторонних репо прямо говорят: «The demo doesn't include the classifier… It is a commercial product and is available for purchase at spectrico.com». Открытым BSD/MIT лицензированы только **обёртки-демо** (YOLOv3-интеграция), **НЕ `.pb`-веса классификатора**. Явного «commercial OK» на free-версию НЕТ; распространение `.pb` через сторонние репо (josesaribeiro/chenxyzj/Austin-Ellsworth) **не означает** коммерческого разрешения правообладателя. Для коммерции CARS — фактически ближе к «no», формально оставлено `unclear`; реэкспорт в ONNX юридически рискован.
- Free-версия заявлена как «reduced accuracy» — деградированная.
- RU/СНГ-марки не подтверждены; `.pb`→ONNX не поддержан вендором; ракурс камеры (фронтальные кадры) ≠ бортовой POV.

Применимость в пайплайне CARS: только для быстрого research-прототипа/сравнения. Для production CARS **не брать** без коммерческой лицензии Spectrico — это конкурентный готовый MMR-продукт, а не open-аналог.
Edge / Jetson Orin Nano: технически — лучший edge-fit из всех (MobileNetV2 128×128, мелкий `.pb`), но **юридически непригоден** → не рассматривается как реальный кандидат.

### Pells31/Vehicle-Make-and-Model-Recognition (Stanford Cars + VMMRdb transfer)
**— / transfer learning: ImageNet-CNN с замороженными слоями + заменённый FC (backbone в README не назван) / не специфицировано / лицензия UNCLEAR — LICENSE-файл не подтверждён (комм.: UNCLEAR → блокер) / PyTorch/TF (ONNX нет; веса как артефакт не подтверждены) / US-домен make+model, низкая воспроизводимость**

Плюсы:
- Объединяет два US-датасета (Stanford Cars + VMMRdb) — широкий US/EU make-словарь и готовый рецепт transfer learning.
- Открытый код пайплайна — полезен как **референс-реализация** дообучения.

Минусы:
- **Лицензия не подтверждена (UNCLEAR, LICENSE-файл не найден)** — для коммерции блокер до review; дополнительно наследует неясные права датасетов (Stanford Cars unknown/research + VMMRdb-обёртка) → двойная неопределённость прав.
- **Веса как скачиваемый файл-артефакт не подтверждены** (доступ только через Streamlit-демо) — низкая воспроизводимость/интеграция, edge-оценка невозможна.
- Нет RU-марок; нет готового ONNX; backbone в README не специфицирован; US-перекос, каталожные ракурсы ≠ бортовой POV.

Применимость в пайплайне CARS: маргинален как готовая модель (публичные веса не подтверждены, лицензия unclear). Полезен только как **открытый референс-код** для собственного fine-tune (см. timm-путь).
Edge / Jetson Orin Nano: зависит от выбранного backbone; ImageNet-CNN экспортируем в ONNX→TRT, но без подтверждённо выложенных весов оценка невозможна — **не измерено**.

### Plate Recognizer Vehicle MMC (Make/Model/Color) SDK
**— / закрытая (не раскрывается); on-premise SDK — контейнер с собственным движком / не раскрыто / проприетарная платная лицензия (комм.: ДА — но платный/закрытый, не open-аналог) / закрытый Docker-SDK; своего ONNX/TensorRT-пути для DeepStream нет / ALPR-домен близок к CARS, но чёрный ящик**

Плюсы:
- Готовый **коммерчески-лицензируемый** MMC прямо под ALPR-домен CARS (контроль доступа/парковка/патруль); on-premise (privacy).
- Снимает разработку: make+model+color+plate в одном SDK; заявлена Jetson/ARM-поддержка.

Минусы:
- Платно (подписка/perpetual, MMR +50% к цене) — это покупка стороннего продукта, **не open-аналог**.
- Веса закрыты — **нельзя интегрировать как SGIE-модель** в наш TensorRT/DeepStream-граф, нельзя дообучить на RU.
- Vendor lock-in; RU/СНГ-покрытие не подтверждено; лицензия требует периодической онлайн-проверки (Subscription) либо Perpetual.

Применимость в пайплайне CARS: альтернатива «купить вместо обучать», если open/fine-tune путь не устроит. В наш модельный pipeline (ONNX→TensorRT SGIE) **не встраивается** — это чёрный ящик-сервис. Отмечен как платный fallback, не как заменяемая модель.
Edge / Jetson Orin Nano: on-premise SDK заявлен под edge (вкл. ARM/Jetson), но как **отдельный сервис**, не как SGIE-модель в нашем графе.

### OpenVINO Open Model Zoo: vehicle-attributes-recognition-barrier-0042 / -0039
**— / компактная CNN-классификация атрибутов (модель Intel, barrier/traffic) / 0042: 11.177M параметров, 0.462 GFLOPs / Apache-2.0 (лицензия репо подтверждена) (комм.: ДА) / OpenVINO IR; конвертация в ONNX возможна / крайне лёгкая, но МАРКУ НЕ классифицирует**

Плюсы:
- Чистая **Apache-2.0** лицензия репозитория (LICENSE подтверждён — эталон «чистого» зоопарка).
- Очень лёгкая и быстрая для атрибутов цвета/типа (0.462 GFLOPs, вход 72×72×3 BGR).

Минусы:
- **НЕ классифицирует марку/бренд** — прямой пробел для задачи Make (подтверждено README: выход только color(7) + type(4): car/van/truck/bus; color acc ~82.7%, type acc ~87.3%).
- Целевой рантайм **OpenVINO/Intel**, а не TensorRT/Jetson — лишний шаг конвертации без выгоды; per-model SPDX в README не продублирован (полагаемся на лицензию репо).

Применимость в пайплайне CARS: для задачи Make **не применима** — документирует ПРОБЕЛ: open-зоопарки атрибутов ТС дают type/color, но **не марку**. Может пригодиться в CARS для смежных атрибутов (цвет/тип), не для Make.
Edge / Jetson Orin Nano: крайне лёгкая (0.462 GFLOPs, 72×72) — но нерелевантно, т.к. не даёт марку.

> **Дополнительная находка (не вынесена в отдельную карточку, лицензия unclear / нет RU):** `abdusah/CarViT` — ViT-base (~85.8M), 224×224, классифицирует **42 ПРОИЗВОДИТЕЛЯ на BRAND-уровне** (Acura, Audi, BMW, … Volvo, smart) — редкий пример **голого make**-классификатора (без model/year). НО: обучен через `huggingpics` (Bing/web-scrape — шумные данные, риск прав на фото), **лицензия в карточке явно НЕ указана** (поле отсутствует → unclear, по умолчанию запрет для коммерции), RU/СНГ-марок нет, acc ~0.81 (заявление автора, на маленьком наборе). Также `kingjosephm/vehicle_make_model_classifier` (574 make+model, ResNet50, LICENSE 404 → unclear). Вывод: голый make-классификатор в open-доступе **существует** (CarViT), но без коммерчески-чистой лицензии и **без RU-марок**.

## Сводная таблица аналогов

| Модель | Год | Архитектура | Параметры/размер | Лицензия (комм.?) | Форматы (ONNX/TRT) | Edge-fit (Orin Nano) | Домен-fit | Вердикт |
|---|---|---|---|---|---|---|---|---|
| timm fine-grained transfer (ConvNeXt/EfficientNet/MobileNetV3-4) | 2019–24 | CNN backbone + FC | ~5M–28M | КОД Apache-2.0 / ВЕСА UNCLEAR (NC) | ONNX штатный → TRT FP16/INT8 | отличный (выбор backbone, ≤ baseline) | единственный путь под RU/POV (при обучении) | **primary** |
| Jordo23/vehicle-classifier | 2024 | EfficientNet-B4 + FC | ~19M; .pth 135MB / .onnx 927KB (битый) | MIT (ДА; риск данных VMMRdb) | ONNX битый → переэкспорт из .pth → TRT | тяжелее baseline (380×380) | US/EU make+model+year; нет RU | **primary** |
| dima806/car_models_image_detection | 2023 | ViT-base-p16-224 | ~86M; 344MB | Apache-2.0 (ДА) | ONNX нет (optimum) → TRT | тяжёлый ViT (>> baseline) | US/мировой make+model; нет RU | **secondary** |
| therealcyberlord/stanford-car-vit-patch16 | ~2023 | ViT-base-p16-224 | ~85.8M | Apache-2.0 (ДА; риск данных Stanford) | ONNX нет (optimum) → TRT | тяжёлый ViT (>> baseline) | узкий US make+model+year; нет RU | **secondary** |
| Spectrico Vehicle MMR (MobileNet) | — | MobileNet v2 (.pb) | ~единицы MB, 128×128 | ПРОПРИЕТАРНАЯ; открыты только обёртки (UNCLEAR→блокер) | .pb→ONNX не поддержан вендором | технически отличный, юр. непригоден | 400 make (заявл.); нет RU; камерный | **marginal** |
| Pells31/Vehicle-Make-and-Model-Recognition | — | ImageNet-CNN transfer (backbone n/a) | не указано | UNCLEAR (нет LICENSE → блокер) | ONNX нет; веса не подтверждены | не оценить (нет весов) | US make+model; нет RU | **marginal** |
| Plate Recognizer Vehicle MMC SDK | — | закрытая | не раскрыто | проприетарная платная (ДА, но закрытый) | закрытый Docker; не встраивается в наш граф | отдельный сервис, не SGIE | ALPR-домен; RU не подтверждён | **not_recommended** |
| OpenVINO OMZ vehicle-attributes-barrier-0042/-0039 | — | компактная CNN (атрибуты) | 11.177M, 0.462 GFLOPs | Apache-2.0 (ДА) | OpenVINO IR (Intel, не TRT) | очень лёгкая, но не даёт марку | только color+type; НЕ make | **not_recommended** |

## Рекомендации

**Primary (для production CARS):**
- **timm fine-grained transfer-learning** — рекомендуемый основной путь для **RU/СНГ-продакшна**: это единственный способ закрыть пробел RU-марок (Lada/VAZ, Москвич, GAZ/UAZ, китайские бренды) и domain gap бортового POV. ВАЖНО: стартовые веса backbone выбирать **с проверкой лицензии** (ImageNet-веса по предупреждению самих авторов timm — non-commercial research-риск; FB WSL/SSL/SWSL — явно CC-BY-NC), либо **обучать с нуля**; итоговые веса — собственность проекта. Данные: [06_make_vmmrdb.md](../dataset_research/06_make_vmmrdb.md) (юридически безопасный источник — Roboflow Universe «car brands» CC BY 4.0, малый/шумный, + собственный сбор; CompCars/BoxCars116k/Frontal-103 — только research).
- **Jordo23/vehicle-classifier** — для быстрого старта по **US/EU make+model**: заявленная MIT (код+веса), но обязателен **переэкспорт рабочего ONNX из `.pth`** (штатный `.onnx` 927KB — битый блокер) и агрегация 8949→make. Подходит и как стартовые веса для дообучения на RU. Зафиксировать legal review происхождения VMMRdb-фото.

**Secondary / teacher:**
- **dima806** и **therealcyberlord** (оба Apache-2.0) — для US/мирового make+model и как **teacher для дистилляции** в компактный edge-CNN. Прямой деплой ViT-base (~86M) на Orin Nano для realtime нецелесообразен; ONNX не выложен — нужен собственный экспорт. Метки make+model требуют агрегации в марку.

**Лицензионные предостережения:**
- `unclear` = блокер до legal review. Под это попадают: **timm-ВЕСА** (код Apache, веса ImageNet/CC-BY-NC — per-model отбор обязателен), **Spectrico free** (проприетарный продукт, открыты только обёртки — фактически «no» для коммерции), **Pells31** (нет LICENSE), **CarViT/kingjosephm** (нет явной лицензии).
- Риск **исходных обучающих данных** (даже при чистой лицензии весов): VMMRdb (Jordo23), Stanford Cars (cyberlord), huggingpics-scrape (CarViT) — происхождение фото для production стоит проверить.
- **Plate Recognizer** — валидный, но платный/закрытый путь «купить вместо обучать»; в наш ONNX→TensorRT/DeepStream-граф не встраивается.
- Проверка классических ловушек: **Ultralytics AGPL** не релевантен (детекция уже у TrafficCamNet), **InsightFace non-commercial** — это лица, не марки, **NVIDIA TAO EULA** — сам baseline. Ни один make-кандидат не тянет AGPL/InsightFace-ограничений.

**Edge-предостережения:** для Orin Nano realtime ViT-base (dima806, cyberlord, CarViT ~86M) тяжелее baseline ResNet-18 pruned (7.4MB); EfficientNet-B4 (Jordo23) тоже тяжелее. Оптимальный edge-путь — timm MobileNetV3-L / EfficientNet-B0 (с проверкой лицензии веса) + дообучение, либо **дистилляция из ViT/B4-учителя** в компактный CNN, экспорт ONNX→TensorRT FP16/INT8. Все latency/размеры аналогов — **не измерены нами**.

**Главный вывод:** открытого коммерчески-чистого make-классификатора **С RU/СНГ-марками не найдено** — ни одна модель (CarViT/kingjosephm/Jordo23/dima806/cyberlord) не имеет RU-словаря. Это подтверждает диагноз baseline VehicleMakeNet (Top-1 0.083 на RU-суррогате) и [06_make_vmmrdb.md](../dataset_research/06_make_vmmrdb.md): **для RU-домена путь — ДООБУЧЕНИЕ собственного классификатора (timm primary-путь), а не готовый open-аналог.** Для US/EU make быстро — dima806/cyberlord (Apache-2.0) или Jordo23 (MIT, после переэкспорта ONNX) с агрегацией в make.

## Ссылки

- Jordo23/vehicle-classifier: https://huggingface.co/Jordo23/vehicle-classifier · https://huggingface.co/Jordo23/vehicle-classifier/tree/main · https://huggingface.co/Jordo23/vehicle-classifier/raw/main/README.md
- dima806/car_models_image_detection: https://huggingface.co/dima806/car_models_image_detection · https://huggingface.co/dima806/car_models_image_detection/tree/main
- therealcyberlord/stanford-car-vit-patch16: https://huggingface.co/therealcyberlord/stanford-car-vit-patch16
- timm (pytorch-image-models): https://github.com/huggingface/pytorch-image-models · https://huggingface.co/docs/timm/models/efficientnet · https://huggingface.co/docs/timm
- Spectrico Vehicle MMR: http://spectrico.com/car-make-model-recognition.html · https://github.com/josesaribeiro/car-make-model-classifier-yolo3-python
- Plate Recognizer Vehicle MMC: https://platerecognizer.com/vehicle-make-model-recognition-with-color/ · https://platerecognizer.com/pricing/
- OpenVINO OMZ vehicle-attributes-barrier-0042: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0042/README.md · https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE
- Pells31/Vehicle-Make-and-Model-Recognition: https://github.com/Pells31/Vehicle-Make-and-Model-Recognition

## История изменений
- 2026-06-05 — создан в рамках исследования открытых моделей-аналогов стека CARS (многоагентный workflow, веб-поиск по первоисточникам, лицензии перепроверены адверсариально).
