# Аналоги модели «VehicleTypeNet» для задачи «Type — классификация типа кузова ТС»

> **См. также:** спецификация исходной модели — [vehicletypenet.md](../about_models/vehicletypenet.md) · реестр моделей — [models.md](../../models.md) · исследование датасетов-аналогов (та же задача) — [07_type_stanford.md](../dataset_research/07_type_stanford.md) · сводка исследования моделей — [00_SUMMARY.md](00_SUMMARY.md)

> Документ подготовлен в рамках исследования открытых моделей-аналогов стека CARS.
> **CARS — коммерческий продукт.** Лицензионный статус каждого кандидата трактуется консервативно: пока коммерческое использование не подтверждено первоисточником явно, оно считается **запрещённым/неясным (restricted/unclear)** и помечается как блокер. Любое включение модели/весов в прод-сборку или дистрибуцию TensorRT-engine требует отдельного legal review.

---

## Контекст и требования

**Целевая модель — VehicleTypeNet** (NVIDIA TAO, ResNet-18 pruned, ~19.9 MB ONNX). В пайплайне CARS она работает как **SGIE2** (вторичный классификатор): на вход поступает кроп ТС, вырезанный детектором PGIE, на выходе — softmax по **6 классам кузова** `{coupe, largevehicle, sedan, suv, truck, van}`. Препроцессинг строго специфичен: вход **224×224 BGR**, смещения каналов (offsets) `104/117/124`, **без деления на 255**, layout NCHW. Это определяет роль аналога: он должен либо повторить этот интерфейс, либо стать базой для дообучения под него.

Рантайм — бортовой: **Jetson Orin Nano 8GB**, исполнение через **DeepStream/TensorRT (FP16/INT8)** либо ONNX Runtime. Это жёстко ограничивает бюджет модели: SGIE2 запускается на каждом кропе каждого кадра, поэтому ResNet-18-класс (~20 MB) — практический потолок, а тяжёлые бэкбоны (EfficientNet-B4@380, ResNet-50@224) на каждом кропе создают ощутимую нагрузку и требуют обязательного INT8.

Целевой сценарий — **бортовой POV** (контроль доступа, парковка, патруль, логистика) с режимами **day/night/IR** и регионом **RU/UA**. Тип кузова к региону почти не привязан, но ночь/ИК-подсветка и ракурс «с уровня машины» формируют **domain gap** ко всем обучающим доменам аналогов (трафик-камеры сверху, барьерные камеры, web-фото, re-ID-съёмка). Наконец, **коммерческая лицензия — жёсткое требование**: NC-датасеты в происхождении весов, EULA без явного разрешения, отсутствие LICENSE — всё это блокеры.

На текущем замере инкумбент дал **FAIL Top-1 0.357 на Stanford Cars**, но это артефакт суррогатных keyword-меток (Stanford размечен по make/model/year, а не по типу кузова) — истинная точность на корректном датасете **нами не измерена**.

**Что требуется от аналога:**
1. **Нативная классификация типа кузова** (softmax), желательно совпадающая с 6 классами CARS — иначе нужен маппинг таксономии с потерей различимости.
2. **Коммерчески-чистая лицензия на код И веса** (Apache-2.0/MIT/BSD); NC/AGPL/EULA/отсутствие лицензии — блокеры.
3. **Edge-бюджет под Orin Nano** — ResNet-18-класс или легче, с реальным путём к ONNX→TensorRT (FP16/INT8).
4. **Нативный ONNX или штатный экспорт в ONNX** — для интеграции в DeepStream/TensorRT.
5. Доменная близость к бортовому POV (day/night/IR) — либо пригодность как база для дообучения на собственных данных.

> **Важно по препроцессингу.** Все аналоги, кроме инкумбента, используют **иной** препроцессинг: OpenVINO — 72×72 BGR без CARS-offsets; PaddleClas/Jordo23/AventIQ — RGB + ImageNet-нормализация. При любой замене на не-TAO модель параметры в `evaluate.py` / DeepStream-конфиге (`net-scale-factor`, `offsets`, `model-color-format`, `input-dims`) **придётся переписать** — нельзя просто подменить `.onnx`, сохранив BGR-offsets `(104,117,124)` и 224×224.

---

## Обзор аналогов

### NVIDIA VehicleTypeNet (актуальная версия NGC TAO)
**актуальный TAO / ResNet-18 pruned / ~19.9 MB ONNX, 6 классов / Лицензия: NVIDIA Model EULA (комм.: UNCLEAR — блокер до сверки) / Форматы: ONNX (pruned_onnx_v1.1.0), нативный TensorRT/DeepStream / Идеальный edge-fit, но это сам инкумбент**

Это та же линейка, что и текущая модель (доступны `pruned_v1.0.2` и `pruned_onnx_v1.1.0`). Архитектура — ResNet-18, prune+retrain на categorical cross-entropy (наследие TAO/TF1, подтверждено docs.nvidia.com). Интерфейс **идентичен** инкумбенту: вход 224×224 BGR, offsets `(104,117,124)`, без `/255`, NCHW; выход — softmax по 6 классам `coupe/largevehicle/sedan/suv/truck/van` (подтверждено docs.nvidia.com).

Плюсы:
- **100% совместимость интерфейса** (вход/препроцессинг/6 классов/ONNX→TensorRT/DeepStream) — нулевая интеграционная стоимость, единственный кандидат с нативной 6-классовой кузовной таксономией CARS.
- Тот же edge-бюджет (~19.9 MB), ONNX поставляется напрямую, в комплекте INT8-калибровочный кэш; целевая latency ~3–4 ms (legacy-оценка, **не измерено на устройстве**).
- Каскад с DashCamNet/TrafficCamNet для smart city — DashCamNet-ветка частично ближе к бортовому/dashcam-POV, чем барьер/трафик-сверху.

Минусы:
- **Это и есть инкумбент** — не решает проблему качества (FAIL-метрика была на суррогатных метках Stanford, истинная точность на корректном датасете **не измерена**).
- **NVIDIA Model EULA** — коммерческая чистота **НЕ подтверждена** (точный текст в локальной копии отсутствует; коммерческий деплой и редистрибуция весов/собранного engine требуют сверки с актуальным NGC/TAO EULA). Тот же лицензионный блокер, что у текущей модели.
- Domain gap к бортовому POV day/night/IR не закрыт.

Применимость в пайплайне CARS: baseline/инкумбент. Включён для полноты — если новая версия NGC (или DashCamNet-каскад) даёт лучшее покрытие, это путь наименьшего сопротивления, но **лицензия (EULA) и качество не улучшаются**. Полноценной замены он не даёт.
Edge / Jetson Orin Nano: идеален — ResNet-18 pruned ~19.9 MB, INT8-кэш в комплекте, нативный DeepStream-classifier. Latency ~3–4 ms — **не измерено нами на устройстве**.

### OpenVINO vehicle-attributes-recognition-barrier-0042
**Intel OMZ / ResNet-подобная (backbone в README не назван) / 11.18 MParams, 0.462 GFlops / Лицензия: Apache-2.0 на код И веса (комм.: ДА — проверено) / Форматы: только OpenVINO IR (FP32/FP16/FP16-INT8), НЕ ONNX / Лёгкая, barrier-домен, нужен шаг IR→ONNX→TensorRT**

Свёрточная сеть, исходный фреймворк PyTorch (подтверждено карточкой OMZ и docs.openvino.ai). Backbone в README явно не назван; по `task_type=object_attributes` и косвенным признакам (анализ `model.yml`) — лёгкая ResNet18-семейства сеть, тяжелее `0039` (11.18 vs 0.63 MParams). Верифицировано по README: вход `1×3×72×72`, color order **BGR** (совпадает с CARS по цветопорядку, но размер 72×72 ≪ 224×224). Два выхода: тип (4 класса `car/van/truck/bus`) + цвет (7 классов). Тип — softmax-голова, точность типа avg **87.34%** (по README, **не измерено нами**; разброс 68.57% bus … 97.44% car).

Плюсы:
- **Apache-2.0 на код И веса** — поле `license` в `model.yml` ведёт на репозиторный Apache-2.0 `LICENSE` (проверено **прямым чтением файла**), отдельных ограничений на веса не заявлено. Редкий «clean» случай среди OMZ — **главный коммерчески-чистый кандидат подборки**.
- Совпадает по цветопорядку (BGR) с CARS; есть готовый FP16-INT8; edge-бюджет минимальный; точность типа avg 87.34% (по README, **не измерено нами**).
- Сценарий «security barrier camera» (шлагбаум/контроль доступа) доменно ближе к части кейсов CARS, чем трафик-камера сверху: фронтальный ракурс ТС с уровня барьера.

Минусы:
- **Таксономия 4 класса** (`car/van/truck/bus`) против 6 у CARS — нет раздельных `coupe/sedan/suv` (`car` = всё легковое); нужен маппинг и теряются 3 целевых класса.
- Вход **72×72** (грубо) и обучение на security-barrier домене ≠ бортовой POV day/night/IR (day/night/IR не заявлены — gap по ночи/IR); 72×72 слишком груб для тонких различий coupe/sedan.
- **OMZ НЕ отдаёт ONNX — только IR** (подтверждено docs.openvino.ai); нативный запуск через OpenVINO Runtime. Для DeepStream/TensorRT нужен путь IR→ONNX→TensorRT (прямого экспорта IR→ONNX в OMZ нет — трение). Совмещённая голова тип+цвет требует игнорировать цветовой выход или дорабатывать декодер; backbone не подтверждён.

Применимость в пайплайне CARS: прямой дроп-ин как SGIE2 **невозможен** (4-классовая таксономия, 72×72). Пригоден как **secondary** с маппингом `sedan/coupe/suv→car` либо как лёгкая база для дообучения головы под 6 классов на чистой Apache-2.0 модели. Чистейший по лицензии путь.
Edge / Jetson Orin Nano: отлично по бюджету (11M params / 0.46 GFlops, готовый FP16-INT8), нагрузка ничтожна. Основной риск — не нативный TensorRT/ONNX-движок (только IR); нужен шаг IR→ONNX→TensorRT. Конкретные цифры на устройстве **не измерены**.

### PaddleClas PULC vehicle_attribute (PP-LCNet_x1_0)
**PaddlePaddle/PaddleClas / PP-LCNet_x1_0 + multilabel-голова / ~7.2 MB / Лицензия: КОД Apache-2.0 (чисто), но ВЕСА на VeRi → non-commercial (комм.: НЕТ — блокер для готовых весов) / Форматы: Paddle Inference, штатный Paddle2ONNX→TensorRT / Лёгкая, богатая таксономия 9 типов, но NC-веса**

PP-LCNet_x1_0 (ультралёгкий MobileNet-подобный CNN) + multilabel-голова; обучение SSLD pretrained + EDA + SKL-UGI distillation (подтверждено docs). Верифицировано по docs: выход — **multilabel-вектор из 19 бинарных атрибутов** = 10 цветов + 9 типов `{sedan, suv, van, hatchback, mpv, pickup, bus, truck, estate}`. Это **НЕ** чистый softmax по 6. Вход типично 224×224, RGB-пайплайн с ImageNet-нормализацией (≠ BGR-offsets CARS — нужен переписанный препроцессинг). Точность mA **90.81%** на тесте (по docs, **не измерено нами**).

Плюсы:
- **Самая богатая таксономия (9 типов)** среди кандидатов — покрывает `sedan/suv/van/truck/pickup`, удобный маппинг к 6 классам CARS (нет отдельного coupe → маппинг).
- Очень лёгкая (~7.2 MB), спроектирована под edge/CPU; штатный экспорт **Paddle2ONNX → TensorRT**; **код Apache-2.0** позволяет переобучить с нуля на чистых данных (снимая VeRi-блокер).

Минусы:
- **БЛОКЕР коммерции для ГОТОВЫХ весов:** обучены на **VeRi**, который выдаётся только по email-запросу с аффилиацией и **явно для НЕкоммерческого использования** («to make sure the dataset is used for non-commercial purposes», vehiclereid.github.io/VeRi). Это не «регистрация», а явный non-commercial запрет; производные веса PULC наследуют NC → **коммерческое использование готовых весов ЗАПРЕЩЕНО**.
- **Multilabel-голова (тип+цвет)** и **RGB+ImageNet-нормализация** ≠ softmax-6 / BGR-offsets CARS — нужен новый декодер и препроцессинг.
- Домен VeRi (мульти-камерный re-ID трафик, фронт/тыл ТС) — фиксированные камеры, **не** бортовой POV; day/night/IR не заявлены.

Применимость в пайплайне CARS: готовые веса для коммерческого CARS **НЕпригодны** (VeRi non-commercial). Реальная ценность — как **код-база Apache-2.0** для обучения PP-LCNet-головы под 6 классов CARS **на собственном бортовом датасете** (это снимает И VeRi-риск, И multilabel/препроцессинг-несовместимость). Роль: чистая база для fine-tune SGIE2, **не** как готовая модель.
Edge / Jetson Orin Nano: отлично — PP-LCNet ~7.2 MB, через ONNX/TensorRT пойдёт легко, INT8 достижим стандартными средствами. Конкретные цифры **не измерены**.

### Jordo23/vehicle-classifier (EfficientNet-B4, VMMRdb)
**HuggingFace / EfficientNet-B4 (timm) / 130 MB .pth (~19M params); ONNX выложен (~1MB — подозрительно мал) / Лицензия: MIT, явно «free for commercial use» (комм.: ДА) / Форматы: PyTorch + ONNX (валидность ONNX под вопросом) / Решает ДРУГУЮ задачу (make/model), не тип кузова**

EfficientNet-B4 через timm, fine-tune от ImageNet (подтверждено карточкой HF). Верифицировано по карточке: вход **380×380** (RGB, ImageNet-норма — **НЕ** BGR-offsets CARS); выход — **8949 классов make/model/year** (Top-1 ~50%, Top-5 ~75–80%), **НЕ тип кузова**. Топология выхода несовместима с задачей Type.

Плюсы:
- **Чистая MIT-лицензия** с явным разрешением коммерции на код и веса (проверено по карточке HF: «MIT License — Free for personal and commercial use»); `.pth` и `.onnx` выложены.
- Подтверждённая доступность весов на HF; timm/EfficientNet-B4 — стандартный, легко дообучаемый бэкбон.

Минусы:
- **Решает другую задачу** (make/model/year, 8949 классов, Top-1 ~50%), а не тип кузова — как классификатор Type **непригоден из коробки**.
- Вход **380×380 RGB ImageNet-норма** ≠ 224×224 BGR-offsets CARS; EfficientNet-B4@380 заметно тяжелее ResNet-18@224 — на Orin Nano как SGIE2 на каждом кропе создаст ощутимую нагрузку.
- Web-домен VMMRdb (фронт 3/4) ≠ бортовой POV; **выложенный `.onnx` ~1 MB несоразмерно мал для B4** (~19M params) — возможен битый/частичный экспорт, перед использованием **обязательна валидация**.

Применимость в пайплайне CARS: только как **MIT-чистая база для дообучения** — взять EfficientNet-B4 (или легче — B0 / timm-resnet18) и обучить голову на 6 классов кузова на собственном/чистом датасете. Сама по себе для Type не годится, но снимает лицензионный риск при transfer learning.
Edge / Jetson Orin Nano: EfficientNet-B4@380 дорог для SGIE2; INT8 поможет, но 380×380 остаётся затратным. Рекомендуется брать как backbone и уменьшать (B0/ResNet-18). **Не измерено.**

### OpenVINO vehicle-attributes-recognition-barrier-0039
**Intel OMZ / ResNet-подобная (модиф. ResNet-10), исходник Caffe / 0.626 MParams, 0.126 GFlops / Лицензия: Apache-2.0 на код И веса (комм.: ДА — проверено) / Форматы: только OpenVINO IR (FP32/FP16/FP16-INT8), НЕ ONNX / Сверхлёгкая, но грубее 0042 и Caffe-путь**

Лёгкая ResNet-подобная сеть (по сторонним описаниям — модифицированный ResNet-10), исходный фреймворк **Caffe** (подтверждено README). Верифицировано по README: вход `1×3×72×72`, **BGR**; выходы — тип (4 класса `car/bus/truck/van`) + цвет (7 классов). Тип — softmax, точность типа overall **87.56%** (по README, **не измерено нами**; разброс 68.57% bus … 98.26% car).

Плюсы:
- **Apache-2.0 (код и веса, проверено по `model.yml`)** — коммерчески чисто (как и 0042).
- **Сверхлёгкая** (0.626M params, 0.126 GFlops) — практически «бесплатная» на Orin Nano; есть INT8; тип-точность overall 87.56% (по README, **не измерено нами**).

Минусы:
- **4 класса вместо 6** (тот же недостаток таксономии, что у 0042).
- **Caffe/IR без нативного ONNX** — самый трудный путь к TensorRT-движку; вход **72×72** слишком груб для различий coupe/sedan; barrier-домен ≠ бортовой POV.

Применимость в пайплайне CARS: аналог 0042, но грубее (ResNet-10, 72×72). Имеет смысл только если нужна предельно лёгкая модель и достаточно 4 классов с маппингом. **0042 предпочтительнее** по точности при той же лицензии. Резервный clean-вариант.
Edge / Jetson Orin Nano: максимально лёгкая (0.126 GFlops), есть INT8. Минус — Caffe-происхождение и IR-формат усложняют сборку нативного TensorRT-движка. **Не измерено.**

### AventIQ-AI/ResNet-50-Vehicle-Segment-classification (MIO-TCD)
**HuggingFace / ResNet-50 (torchvision) / ~25.6M params (~98 MB FP32) / Лицензия: НЕ указана (нет SPDX) + данные MIO-TCD CC BY-NC-SA 4.0 (комм.: НЕТ — блокер) / Форматы: только PyTorch, ONNX НЕ предоставлен / Surveillance-домен, смешанная таксономия, NC-веса**

ResNet-50 (torchvision), fine-tune (подтверждено карточкой HF — Resize 224 + ImageNet Normalize). Верифицировано: вход 224×224 (**RGB, ImageNet-норма** mean `[0.485,0.456,0.406]`/std `[0.229,0.224,0.225]` — **НЕ** BGR-offsets CARS). Выход — **11 классов MIO-TCD**: `articulated_truck/bicycle/bus/car/motorcycle/non-motorized_vehicle/pedestrian/pickup_truck/single_unit_truck/work_van/unknown` — смесь транспорта и пешеходов, не чистая таксономия кузова.

Плюсы:
- Готовая обученная модель типов ТС на стандартном ResNet-50 — легко переэкспортировать в ONNX.
- 224×224 совпадает по разрешению с CARS (но **не** по нормализации — ImageNet RGB ≠ BGR-offsets).

Минусы:
- **БЛОКЕР коммерции:** MIO-TCD распространяется под **CC BY-NC-SA 4.0** (NonCommercial-ShareAlike — исправлено с ошибочного «CC BY-NC-ND» в черновике, проверено на tcd.miovision.com); NC-компонент запрещает коммерческое использование производных весов. Лицензия **самой модели вообще не указана** (нет SPDX-тега, «empty or missing yaml metadata») → «все права защищены», запрещено до legal review.
- Таксономия (11 классов, смесь ТС + pedestrian/bicycle/non-motorized, без `sedan/suv/coupe`) и **surveillance-домен** (трафик-камеры сверху/сбоку, Канада/США) не подходят; **ONNX не выложен** — нужен ручной экспорт ResNet-50→ONNX→TensorRT (тривиально, но веса всё равно NC).

Применимость в пайплайне CARS: для коммерческого CARS **непригодна** из-за NC-происхождения данных (MIO-TCD CC BY-NC-SA) и отсутствия лицензии на саму модель. Может служить лишь референсом качества/таксономии MIO-TCD. **Не использовать в продукте.**
Edge / Jetson Orin Nano: ResNet-50@224 тяжелее ResNet-18 инкумбента (~25.6M params) — для SGIE2 на каждом кропе заметно дороже; INT8 обязателен. Моот — веса всё равно NC. **Не измерено.**

### dchen327/car-body-type-classifier (fast.ai, 5 классов)
**GitHub / fast.ai CNN (предположительно ResNet-backbone) / размер не указан, весов в репо НЕТ / Лицензия: LICENSE отсутствует (404) → «все права защищены» (комм.: НЕТ — блокер) / Форматы: fast.ai/PyTorch, ONNX нет / Учебное демо без весов и спецификации**

fast.ai CNN, демо-приложение (форк render-examples/fastai-v3). 5 классов: `sedan, suv, truck, van, convertible` (подтверждено README). Точный вход/препроцессинг (fast.ai-стандарт RGB) не задокументирован. Обученные веса в репозитории **ОТСУТСТВУЮТ** (проверено через GitHub API: только `.dockerignore`, `.gitignore`, `Dockerfile`, `README.md` 345 байт, `app/`, `requirements.txt` — никаких `.pth/.pkl`).

Плюсы:
- Таксономия (`sedan/suv/truck/van/convertible`) концептуально близка к части CARS.
- Открытый референс-код пайплайна обучения body-type на fast.ai.

Минусы:
- **Нет LICENSE** (GitHub API `/license` → 404) → коммерческое использование запрещено по умолчанию.
- **Нет обученных весов** в репозитории (подтверждено listing'ом contents); нет ONNX/edge-пути, нет спецификации входа/метрик — не production-grade.

Применимость в пайплайне CARS: только как **код-референс** подхода к обучению body-type классификатора. Как поставляемая модель непригодна (нет весов, нет лицензии). **Не использовать в продукте.**
Edge / Jetson Orin Nano: неопределим — нет весов и спецификации; fast.ai-демо не предназначено для edge-деплоя.

---

## Сводная таблица аналогов

| Модель | Год | Архитектура | Параметры/размер | Лицензия (комм.?) | Форматы (ONNX/TRT) | Edge-fit (Orin Nano) | Домен-fit | Вердикт |
|---|---|---|---|---|---|---|---|---|
| NVIDIA VehicleTypeNet (актуальный NGC) | актуальный TAO | ResNet-18 pruned | ~19.9 MB ONNX, 6 классов | Model EULA (**UNCLEAR** — блокер) | ONNX напрямую, нативный TensorRT/DeepStream | Идеален: ~19.9 MB, INT8-кэш, ~3–4 ms (не измерено) | DashCam/Traffic-каскад, частично dashcam; 6 классов = таксономия CARS | not_recommended (инкумбент) |
| OpenVINO barrier-0042 | OMZ | ResNet-подобная (PyTorch, backbone не назван) | 11.18 MParams, 0.462 GFlops | **Apache-2.0 код+веса (ДА)** | Только IR; IR→ONNX→TRT (трение) | Отлично: 0.46 GFlops, FP16-INT8 готов | Barrier/контроль доступа; 4 класса; 72×72; нет day/night/IR | secondary |
| PaddleClas PULC (PP-LCNet_x1_0) | PaddleClas | PP-LCNet_x1_0 + multilabel | ~7.2 MB | Код Apache-2.0; веса VeRi **NC (НЕТ)** | Paddle2ONNX→TRT (штатно) | Отлично: ~7.2 MB, INT8 | VeRi re-ID (фикс. камеры); 9 типов; RGB+ImageNet; нет day/night/IR | secondary (только как код-база для fine-tune) |
| Jordo23/vehicle-classifier | HF | EfficientNet-B4 (timm) | 130 MB .pth, ~19M params; ONNX ~1MB (подозрит.) | **MIT (ДА)** | PyTorch + ONNX (валидность под вопросом) | B4@380 тяжёл для SGIE2; нужен INT8/уменьшение | Web/VMMRdb make-model; **другая задача**; 380×380 RGB | marginal (только backbone) |
| OpenVINO barrier-0039 | OMZ | ResNet-10-подобная (Caffe) | 0.626 MParams, 0.126 GFlops | **Apache-2.0 код+веса (ДА)** | Только IR (Caffe); путь к ONNX/TRT труден | Сверхлёгкая: 0.126 GFlops, INT8 | Barrier; 4 класса; 72×72 (грубее 0042); нет day/night/IR | marginal |
| AventIQ ResNet-50 (MIO-TCD) | HF | ResNet-50 (torchvision) | ~25.6M params, ~98 MB | Нет SPDX + MIO-TCD **CC BY-NC-SA (НЕТ)** | Только PyTorch; ONNX не выложен | ResNet-50@224 дорог; INT8 обяз.; моот (NC) | Surveillance трафик-камеры; 11 смешанных классов; RGB+ImageNet | not_recommended |
| dchen327/car-body-type-classifier | GitHub | fast.ai CNN | весов нет; размер н/д | LICENSE отсутствует (404) **(НЕТ)** | fast.ai/PyTorch; ONNX нет | Неопределим (нет весов/спеки) | Учебное демо, домен неизвестен; 5 классов | not_recommended |

---

## Рекомендации

**Идеального открытого дроп-ин аналога для задачи «тип кузова из бортового POV (day/night/IR, RU/UA, 6 классов)» НЕТ.** Открытой предобученной модели именно с 6-классовой кузовной таксономией CARS (`coupe/largevehicle/sedan/suv/truck/van`), обученной на бортовом/dashcam-POV с RU/UA-покрытием и нативным ONNX/TensorRT, найти не удалось. Все найденные кандидаты либо урезаны (OpenVINO — 4 класса), либо имеют иной набор (PULC — 9, MIO-TCD — 11), либо решают другую задачу (Jordo23 — make/model). Реальный расклад:

- **Primary (готовый дроп-ин под чистой лицензией): отсутствует.** Ближайший «чистый» готовый кандидат — OpenVINO barrier-0042 (см. ниже), но только с маппингом 4→6 классов.

- **Secondary №1 — OpenVINO vehicle-attributes-recognition-barrier-0042.** Единственный коммерчески-чистый (**Apache-2.0 на код И веса**, проверено прямым чтением `model.yml`) готовый кандидат с приемлемой точностью типа (87.34%, не измерено нами). Берётся с маппингом таксономии `sedan/coupe/suv → car` (теряются 3 целевых класса) либо как лёгкая чистая база для дообучения головы под 6 классов. **Edge-предостережение:** OMZ отдаёт **только IR (не ONNX)** — потребуется путь IR→ONNX→TensorRT (трение интеграции); вход 72×72 груб для различий coupe/sedan.

- **Secondary №2 (как код-база, не как готовая модель) — PaddleClas PULC / PP-LCNet.** **Лицензионное предостережение — критично:** готовые веса обучены на **VeRi (явный non-commercial, email-gated)** → коммерческое использование весов **ЗАПРЕЩЕНО**. Ценность — только **код-база Apache-2.0** для обучения PP-LCNet-головы под 6 классов CARS с нуля на собственных данных.

- **Marginal — Jordo23/EfficientNet-B4 (MIT)** как чистый backbone для transfer learning (не как готовый Type-классификатор; ONNX требует валидации, B4@380 уменьшить до B0/ResNet-18); **OpenVINO barrier-0039** — резервный сверхлёгкий clean-вариант, но грубее 0042 и Caffe-путь.

- **Not recommended — инкумбент NVIDIA VehicleTypeNet** (EULA unclear, качество не улучшается), **AventIQ/MIO-TCD** (CC BY-NC-SA + нет SPDX → блокер), **dchen327** (нет LICENSE, нет весов). **Ultralytics YOLO-cls (AGPL-3.0)** в подборку не включён намеренно: для классификации кузова есть Apache-2.0/MIT-альтернативы, предпочтительнее AGPL-ловушки.

**Лицензионные ловушки (проверено по первоисточникам):** (а) OpenVINO OMZ Intel-модели — Apache-2.0 на код И веса (`model.yml` license → репозиторный Apache LICENSE) — **ЧИСТО**; (б) NVIDIA VehicleTypeNet — Model EULA, не Apache/MIT → **unclear/блокер**; (в) MIO-TCD — CC BY-NC-SA 4.0 (NC+SA) → производные коммерчески запрещены; (г) VeRi — явный non-commercial (email-gated) → веса PULC коммерчески запрещены (**no**, не unclear); (д) HF-модели без SPDX (AventIQ) и GitHub без LICENSE (dchen327) — «все права защищены».

**Рекомендуемый путь (раз чистого дроп-ина нет):** дообучение лёгкого **Apache-2.0/MIT-бэкбона** (timm ResNet-18 / PP-LCNet через Apache-2.0 код PaddleClas / EfficientNet-B0) под ровно **6 классов кузова CARS** на **собственном бортовом датасете** (day/night/IR). Это снимает И domain gap (бортовой POV), И лицензионный риск (обучение с нуля на своих данных). Чистые отправные точки: **PaddleClas-КОД** (Apache-2.0, но не его VeRi-веса), **Jordo23/EfficientNet** (MIT). Для готового дроп-ина под лицензией — только **OpenVINO 0042** с маппингом 4 классов. Вопрос обучающих данных под дообучение разобран в [исследовании датасетов-аналогов](../dataset_research/07_type_stanford.md).

---

## Ссылки

- OpenVINO vehicle-attributes-recognition-barrier-0042 — README: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0042/README.md
- OpenVINO 0042 — model.yml: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0042/model.yml
- OpenVINO 0042 — docs: https://docs.openvino.ai/2023.3/omz_models_model_vehicle_attributes_recognition_barrier_0042.html
- OpenVINO OMZ — LICENSE (Apache-2.0): https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE
- OpenVINO vehicle-attributes-recognition-barrier-0039 — README: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0039/README.md
- OpenVINO 0039 — docs: https://docs.openvino.ai/2023.3/omz_models_model_vehicle_attributes_recognition_barrier_0039.html
- OpenVINO 0039 — model.yml: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0039/model.yml
- PaddleClas PULC vehicle_attribute (EN): https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/PULC/PULC_vehicle_attribute_en.md
- PaddleClas PULC vehicle_attribute (ZH): https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_vehicle_attribute.md
- PaddleClas PULC config (PPLCNet_x1_0.yaml): https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml
- VeRi dataset (non-commercial): https://vehiclereid.github.io/VeRi/
- NVIDIA VehicleTypeNet — NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehicletypenet
- NVIDIA VehicleTypeNet — docs (TAO 5.3): https://docs.nvidia.com/tao/archive/5.3.0/text/model_zoo/cv_models/vehicletypenet.html
- NVIDIA VehicleTypeNet — docs (TLT 3.0): https://docs.nvidia.com/metropolis/TLT/archive/tlt-30/text/purpose_built_models/vehicletypenet.html
- Jordo23/vehicle-classifier — HF: https://huggingface.co/Jordo23/vehicle-classifier
- AventIQ-AI/ResNet-50-Vehicle-Segment-classification — HF: https://huggingface.co/AventIQ-AI/ResNet-50-Vehicle-Segment-classification
- MIO-TCD dataset (CC BY-NC-SA 4.0): https://tcd.miovision.com/challenge/dataset.html
- dchen327/car-body-type-classifier — GitHub: https://github.com/dchen327/car-body-type-classifier

---

## История изменений
- 2026-06-05 — создан в рамках исследования открытых моделей-аналогов стека CARS (многоагентный workflow, веб-поиск по первоисточникам, лицензии перепроверены адверсариально).
