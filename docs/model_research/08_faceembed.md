# Аналоги модели «Face embedding (None / Trainable)» для задачи «Face embedding — векторное представление лица (планируемая модель)»

> **См. также:** спецификация исходной модели — [face_embedding.md](../about_models/face_embedding.md) · реестр моделей — [models.md](../../models.md) · исследование датасетов-аналогов (та же задача) — [08_faceembed_widerface.md](../dataset_research/08_faceembed_widerface.md) · сводка исследования моделей — [00_SUMMARY.md](00_SUMMARY.md)

> **Статус документа:** аналитический обзор открытых моделей-аналогов. Все лицензионные пометки проверены по первоисточникам (WebFetch / WebSearch) на 2026-06-05 (см. раздел «Ссылки»). По умолчанию лицензия трактуется как `restricted / unclear`, пока коммерческое разрешение не подтверждено явно НА ВЕСА (а не только на код). **CARS — коммерческий продукт**, поэтому любая модель с лицензией весов academic / non-commercial / research-only / NC / с неясным провенансом обучающих данных трактуется как непригодная для production без отдельного legal review или коммерческого лицензирования. Точность ни одной модели **не измерена нами** на целевом домене — все цифры accuracy приведены по заявлению авторов.

## Контекст и требования

**Целевая модель — Face embedding** (планируемая, обучаемая; в `models.md` — строка `Face embedding | None | None | Trainable`, то есть модель ещё НЕ выбрана). Её роль в пайплайне CARS — построить компактное векторное представление лица поверх кропов, выдаваемых детектором FaceDetect: на вход подаётся выровненный кроп лица (по спеке `docs/about_models/face_embedding.md` — ориентировочно 112×112 RGB, TBD), на выходе — L2-нормированный эмбеддинг (128 или 512-D, TBD) для сравнения по косинусной / евклидовой близости. Этот вектор обслуживает верификацию 1:1 (тот же человек / нет) и идентификацию 1:N (поиск по галерее), работая как вторичный инференс (SGIE) поверх детекций FaceDetect.

**Целевое устройство и рантайм.** Бортовая система на NVIDIA **Jetson Orin Nano 8GB**; модели разворачиваются как **ONNX → TensorRT** (FP16 / INT8) в составе **DeepStream**-пайплайна. Это жёстко сужает выбор: модель должна иметь рабочий путь экспорта в ONNX и конвертацию в TensorRT-engine без экзотических слоёв (или с поддерживаемыми плагинами), а по размеру/латентности — укладываться в бюджет edge-устройства рядом с детекторами и прочими SGIE.

**Целевой домен** — бортовой POV: лицо через лобовое стекло, дистанция 5–50 м, разные углы, **day / night / IR**. Это принципиально расходится с доменом обучения всех найденных аналогов (фронтальные web-фото селебрити). RU/UA-специфика для лиц нерелевантна (в отличие от ГРЗ/OCR-задач стека). **Domain gap зафиксирован для ВСЕХ аналогов и нами не закрыт замерами.**

**Коммерческая лицензия — обязательное требование.** Биометрия лиц вдобавок несёт privacy/legal-нагрузку (сбор и хранение шаблонов лиц), поэтому к лицензии весов отношение особенно строгое.

**Что требуется от аналога:**
- **Лицензия НА ВЕСА**, разрешающая коммерческое использование (пермиссивная MIT/Apache/BSD/public-domain или платная коммерческая); код-MIT при NC-весах не годится.
- **Рабочий ONNX → TensorRT путь** под Jetson Orin Nano (готовый ONNX — большой плюс; чистый CNN-граф предпочтительнее трансформер-гибридов).
- **Edge-бюджет**: компактный backbone (MobileFaceNet/GhostNet/EdgeNeXt-класс, ориентир ~<2M params) либо хотя бы проходимый в FP16/INT8 R50/R100.
- **Open-set эмбеддинг** (без фиксированного словаря классов), вход выровненного кропа ~112×112, выход L2-норм. вектора, сравнение по косинусу.
- В идеале — устойчивость к низкому качеству / дальним / IR-лицам (бортовой POV). На практике этого не даёт никто из открытых.

## Обзор аналогов

### SFace (OpenCV Zoo, `face_recognition_sface_2021dec`)
**2021 / MobileFaceNet backbone + SFace (sigmoid-constrained hypersphere) loss / ONNX ~37 MB FP32 + int8-quant вариант, ~1–1.2M params / Apache-2.0 НА КОД И ВЕСА (комм.: ДА) / форматы: ONNX (вкл. int8), OpenCV DNN, прямой ONNX→TensorRT / отличный edge-fit, домен web-фото.**

Плюсы:
- Единственный в подборке с **реально открытой коммерческой лицензией Apache-2.0 НА ВЕСА** (не только код): файл `models/face_recognition_sface/LICENSE` — полный текст Apache 2.0 без NC-оговорок, README каталога дословно «All files in this directory are licensed under Apache 2.0 License» → покрывает и ONNX-файлы весов. Royalty-free коммерческое использование, без legal-блокеров.
- Готовый ONNX + квантованный int8-вариант прямо в OpenCV Zoo; прямой и тривиальный ONNX→TensorRT (стандартный 112×112 CNN, conv/BN/FC без экзотики); минимальный размер под Orin Nano.
- Проверенная точность ~99.4% LFW (по заявлению авторов Zoo, **не измерено нами**); зрелая интеграция в OpenCV DNN (`cv::FaceRecognizerSF`, `DIS_FR_COSINE`).

Минусы:
- Обучен на web-фото (CASIA-WebFace-класс), фронтальные лица → **сильный domain gap к automotive POV через стекло/IR, точность в целевых условиях не подтверждена** (вероятна деградация на маленьких/боковых/IR-лицах).
- Эмбеддинг всего 128-D — слабее 512-D ArcFace на крупных галереях идентификации 1:N.
- SOTA-точность ниже топовых ArcFace/AdaFace R100; на сложных кадрах (низкое разрешение, профиль) ожидается заметная деградация.

Применимость в пайплайне CARS: **PRIMARY** коммерчески-чистый стартовый embedding поверх кропов FaceDetect — SGIE-вектор для верификации 1:1 и идентификации 1:N. Готов к деплою как baseline; дальнейшее дообучение/fine-tune на собственных бортовых данных закрывает domain gap. Маппинг классов не требуется (open-set).
Edge / Jetson Orin Nano: ~1M params, ~37MB FP32, есть int8-quant → идеально под 8GB; FP16/INT8 TensorRT-engine тривиален, low-latency для SGIE в DeepStream. Целевую точность на бортовом домене — **не измерено нами**.

### dlib face recognition (`dlib_face_recognition_resnet_model_v1`)
**2017 / ResNet-34-reduced (29 conv, half-фильтры), metric/triplet-loss на ~3M лиц / веса ~21.4 MB (.dat) / веса public domain + код dlib Boost-1.0 (комм.: ДА) / форматы: нативный dlib .dat (C++/Python), НЕТ официального ONNX / маленький, но CPU-ориентированный, домен web-фото.**

Плюсы:
- Единственный полностью **public-domain вариант НА ВЕСА** + код под Boost Software License 1.0 → нулевые юридические риски для коммерции (README `davisking/dlib-models` дословно: «anyone can do whatever they want with these model files as I've released them into the public domain»).
- Очень маленький файл весов (~21MB), простая зрелая интеграция (dlib / `face_recognition`), хорошо документирован.

Минусы:
- **Нет ONNX/TensorRT-пути** — несовместим с целевым рантаймом DeepStream/TensorRT на Orin Nano без ручной реимплементации архитектуры (dlib-специфичный формат, нетривиально). Это главный практический блокер.
- Устаревшая (2017) точность/архитектура; собственный формат выравнивания и вход 150×150 усложняют интеграцию поверх кропов FaceDetect (которые планируются 112×112).
- Тот же domain gap к automotive POV/IR, оценок в целевых условиях нет (**не измерено нами**); 128-D, ~99.38% LFW по заявлению автора.

Применимость в пайплайне CARS: **MARGINAL** — ценен только как лицензионно-чистый эталон/референс для оффлайн-сравнения качества. Для бортового TensorRT-пайплайна не подходит из-за отсутствия ONNX-пути и dlib-специфичного рантайма (CPU-ориентирован, без аппаратного ускорения нейросети на Jetson).
Edge / Jetson Orin Nano: размер мал (~21MB), но dlib-рантайм без TensorRT → на Jetson нет ускорения; для DeepStream-пайплайна непригоден без переписывания. **Не измерено нами.**

### FaceNet (davidsandberg TF / timesler facenet-pytorch, Inception-ResNet-v1; py-feat/facenet HF)
**2015–2018 / Inception-ResNet-v1, triplet/softmax-эмбеддинг / ~107–111 MB, ~23M+ params / код MIT, ВЕСА — UNCLEAR (комм.: UNCLEAR → блокер до legal review) / форматы: готовый ONNX в py-feat/facenet (HF), конверсия TF/PyTorch→ONNX известна, ONNX→TensorRT возможен / средний edge-fit, домен web-фото.**

Плюсы:
- Код MIT, очень популярная и документированная база; простая интеграция в Python-сервис (ONNX Runtime); готовый ONNX в `py-feat/facenet` (HF, MIT-метка).
- 512-D эмбеддинг, проверенное качество (~99.65% LFW по заявлению timesler, **не измерено нами**); есть готовые PyTorch→ONNX пути.

Минусы:
- **Лицензия исходных ВЕСОВ `davidsandberg` не определена SPDX** (README лишь просит «give proper credit to those providing the training dataset»), а сами веса обучены на **VGGFace2 и CASIA-WebFace — research-only академические лицензии**; `py-feat/facenet` сам советует проверить лицензию VGGFace2 для коммерции → по адверсариальному правилу проекта = **запрещено до legal review**.
- **Несогласованный вход между дистрибуциями**: классический davidsandberg/timesler — 160×160, переупакованный py-feat — 112×112 (оба 512-D). Нужно жёстко зафиксировать чекпойнт под конкретный препроцессинг.
- Тяжелее edge-кандидатов: Inception-ResNet-v1 (~23M params, ~110MB) даёт более сложный TensorRT-граф; domain gap к automotive POV/IR не оценён.

Применимость в пайплайне CARS: **SECONDARY/условно** — технически удобен для R&D и Python-сервиса (особенно готовый ONNX py-feat), но коммерческий деплой требует legal-проверки лицензии весов и датасетов обучения. Без неё — не в прод.
Edge / Jetson Orin Nano: ~23M params, ~110MB — средний; в FP16 проходит на Orin Nano, но тяжелее MobileFaceNet/SFace; Inception-ResNet-v1 даёт более сложный TensorRT-граф. **Не измерено нами.**

### GhostFaceNets (HamadYA)
**2022 / GhostNet backbone (cheap operations) + Subcenter-ArcFace head / GhostNet-класс, единицы млн params (W1.3 S1/S2) / КОД MIT (чист), ВЕСА — UNCLEAR из-за провенанса данных (комм.: UNCLEAR → блокер на готовые веса до legal review) / форматы: TF/Keras-веса, экспорт в ONNX возможен, ONNX→TensorRT реалистичен / отличный edge-fit, домен web-фото (MS1M).**

Плюсы:
- Очень эффективная edge-архитектура (cheap GhostNet operations), высокая точность LFW для своего размера (~99.78% по заявлению авторов, **не измерено нами**); **КОД под MIT** — пермиссивен и пригоден для дообучения/конверсии.
- Открытый код и веса в релизах, 512-D эмбеддинг; в отличие от EdgeFace/ElasticFace на сам код/LICENSE нет CC-NC ограничения.

Минусы:
- **ИСПРАВЛЕНИЕ относительно черновика:** ранее LICENSE был указан как CC BY-NC-ND 4.0 (двойной блокер) — **это неверно**. Файл `HamadYA/GhostFaceNets/LICENSE` содержит полный текст MIT License (Copyright (c) 2022 HamadYA), GitHub-лейбл репо — «MIT license». CC BY-NC-ND относится ТОЛЬКО к академической статье на ResearchGate, а не к коду/весам. Жёсткого ND-блока на код/веса НЕТ.
- **Веса обучены на MS1MV2/MS1MV3 (Refined MS-Celeb-1M, отозван Microsoft)** → коммерческий статус именно ВЕСОВ неясен из-за провенанса данных; нужен legal review до прода.
- TensorFlow/Keras-исток требует TF→ONNX конверсии; domain gap к automotive POV/IR (**не измерено нами**).

Применимость в пайплайне CARS: **MARGINAL** (повышено с not_recommended после исправления лицензии). КОД MIT → отличная база для СОБСТВЕННОГО обучения на лицензионно-чистых данных. Готовые ВЕСА в прод **не брать** до legal review провенанса MS1M, но архитектура и код коммерчески пригодны для переобучения.
Edge / Jetson Orin Nano: GhostNet — мало FLOPs/params → технически отлично под Orin Nano. **Точность на целевом домене не измерена нами.**

### AdaFace (mk-minchul/AdaFace)
**2022 / Quality-adaptive margin loss поверх IResNet (R18/R50/R100) / R18 лёгкий → R100 ~65M params / КОД MIT, ВЕСА — UNCLEAR (комм.: UNCLEAR → блокер до legal review) / форматы: PyTorch .ckpt, экспорт в ONNX (IResNet стандартный), ONNX→TensorRT прямой / R18/R50 edge-реалистичны, домен web-фото.**

Плюсы:
- **Quality-adaptive margin** теоретически устойчивее на низкокачественных/смазанных лицах — концептуально ближе к дальним/нечётким бортовым кадрам (единственный кандидат с таким преимуществом по идее).
- Код MIT, стандартный IResNet → лёгкий ONNX→TensorRT; выбор размеров backbone под бюджет.

Минусы:
- **Лицензия весов не определена явно** (README предлагает считать их MIT «unless otherwise specified»), но обучены они на MS1MV2/MS1MV3/CASIA-WebFace/VGGFace2/WebFace4M/WebFace12M (research-only / отозванные данные) → коммерция неясна, **по правилу проекта запрещена до legal review**.
- **Важная техническая правка:** модель ожидает **BGR-порядок каналов** (как cv2), а не RGB (явно в README, отличие от InsightFace) — легко ошибиться в препроцессинге кропов FaceDetect.
- Топовые веса на R100 тяжёлые для Orin Nano; лёгкие R18-веса менее точны; domain gap к automotive POV/IR не подтверждён замерами (**не измерено нами**).

Применимость в пайплайне CARS: **MARGINAL** — сильный кандидат по идее (quality-adaptive) и код MIT для СОБСТВЕННОГО обучения, но готовые веса коммерчески неясны. Путь — переобучить AdaFace-loss на чистых данных, либо legal review весов.
Edge / Jetson Orin Nano: R18/R50 реалистичны (FP16); R100 тяжёлый. Backbone стандартный, edge-friendly в малых вариантах. **Не измерено нами.**

### ONNX Model Zoo ArcFace (LResNet100E-IR) / OpenVINO `face-recognition-resnet100-arcface-onnx`
**2018–2019 / ArcFace LResNet100E-IR (ResNet-100, additive angular margin), оригинал MXNet от InsightFace → ONNX / ~250 MB ONNX, ~65M params / формально Apache-2.0, но провенанс весов спорный (комм.: UNCLEAR → блокер до legal review) / форматы: готовый ONNX в обоих зоопарках, OpenVINO IR, прямой ONNX→TensorRT / тяжёлый edge-fit, домен web-фото (MS1M).**

Плюсы:
- Готовый ONNX в OpenVINO/ONNX зоопарках с формальной пометкой Apache-2.0 и прямым, хорошо отлаженным ONNX→TensorRT (классический ArcFace-граф).
- 512-D ArcFace SOTA-класса (~99.7%+ LFW по заявлению, **не измерено нами**); зрелая интеграция.

Минусы:
- **Провенанс весов = InsightFace/MS1M** (Refined MS-Celeb-1M — отозванный датасет, исходно non-commercial), реализация унаследована от InsightFace/MXNet → **заявленный «Apache на дампе» сомнителен**; по правилу проекта «Apache на упаковке» не перекрывает исходную лицензию весов → запрещено до legal review.
- Тяжёлый R100 (~65M params, ~250MB) — на Orin Nano проходит в FP16/INT8, но ест ресурсы; хуже edge-кандидатов по latency/памяти.
- Domain gap к automotive POV/IR (**не измерено нами**).

Применимость в пайплайне CARS: **MARGINAL** — формально «Apache» и технически удобен, но провенанс весов (MS1M/InsightFace) делает коммерческий статус спорным. Использовать только после legal-подтверждения цепочки лицензий; иначе предпочесть SFace.
Edge / Jetson Orin Nano: R100 — тяжёлый, проходит в FP16/INT8 ценой ресурсов; не edge-оптимален против SFace/EdgeFace. **Не измерено нами.**

### ArcFace / InsightFace model packs (buffalo_l → w600k_r50, antelopev2, buffalo_s)
**2019–2021 / ArcFace (additive angular margin) поверх ResNet-50 (w600k_r50) / ResNet-100 (antelopev2), MS1MV-обучение / w600k_r50.onnx ~166–174 MB, buffalo_s лёгкий / КОД MIT, ВЕСА — NON-COMMERCIAL (комм.: НЕТ → БЛОКЕР) / форматы: готовый ONNX, отлаженный ONNX→TensorRT, примеры DeepStream / технически идеален, домен web-фото.**

Плюсы:
- SOTA-качество (~99.85% LFW по заявлению авторов, **не измерено нами**) и зрелейшая ONNX/TensorRT/DeepStream-экосистема — технически лучший embedding в индустрии.
- Готовые веса 512-D, есть лёгкий buffalo_s (MobileFaceNet-класс) под edge; де-факто референс бенчмарков верификации.

Минусы:
- **БЛОКЕР для CARS:** веса «non-commercial research purposes only» (README `deepinsight/insightface` дословно: «The training data ... and the models trained with these data ... for non-commercial research purposes only») → коммерческий деплой запрещён без платной лицензии InsightFace.
- **Ловушка:** README «MIT» относится ТОЛЬКО к коду; веса под отдельным NC-режимом (требуется обращение `recognition-oss-pack@insightface.ai`).
- Domain gap к automotive POV/IR сохраняется (**не измерено нами**).

Применимость в пайплайне CARS: **NOT_RECOMMENDED** для коммерческого деплоя как есть. Допустимо только: (a) как точностный референс в R&D-замерах, (b) после покупки коммерческой лицензии InsightFace. В проде CARS веса использовать нельзя.
Edge / Jetson Orin Nano: buffalo_s — отлично под Orin Nano; w600k_r50 (R50) тяжелее, но в FP16/INT8 проходит. Технически edge-friendly — блокирует только лицензия. **Не измерено нами.**

### EdgeFace (Idiap — XXS / S-GAMMA / Base)
**2023 / гибрид EdgeNeXt (CNN+Transformer) + low-rank linear layer, победитель compact-трека IJCB 2023 / XXS ~1.24M params, 94.72 MFLOPs / КОД MIT, ВЕСА — CC BY-NC-SA 4.0 (комм.: НЕТ → БЛОКЕР) / форматы: PyTorch-веса, ONNX-экспорт выполним, ONNX→TensorRT возможен (трансформер-блоки требуют аккуратности) / лучший edge-fit по точности/размеру, домен web-фото.**

Плюсы:
- **Лучшее соотношение точность/параметры среди edge-моделей** (XXS ~1.24M params, ~95 MFLOPs при высокой точности) — прямой кандидат для Orin Nano по ресурсам.
- Активно поддерживается Idiap, есть готовые веса нескольких размеров на HuggingFace; код MIT пригоден как архитектурный ориентир.

Минусы:
- **БЛОКЕР:** веса CC BY-NC-SA 4.0 (non-commercial + ShareAlike) — подтверждено model card `Idiap/EdgeFace-XXS` («EdgeFace is released under CC BY-NC-SA 4.0», метаданные `license: cc-by-nc-sa-4.0`); несмотря на MIT в README кода, коммерческий деплой весов запрещён.
- Гибридная EdgeNeXt-архитектура (трансформер-блоки) усложняет ONNX→TensorRT против чистого CNN.
- Domain gap к automotive POV/IR не оценён (**не измерено нами**).

Применимость в пайплайне CARS: **NOT_RECOMMENDED** для прода из-за NC-лицензии весов. Архитектура — отличный ориентир для СОБСТВЕННОГО обучения (код MIT): переобучить EdgeFace на лицензионно-чистых/собственных данных → чистые веса.
Edge / Jetson Orin Nano: технически лучший edge-кандидат по ресурсам (XXS ~1.24M params), но блокирует лицензия. **Не измерено нами.**

### ElasticFace (fdbtrs)
**2022 / Elastic margin loss (вариация ArcFace/CosFace) поверх ResNet-100 / ~65M params, тяжёлый / CC BY-NC-SA 4.0 на репозиторий/веса (комм.: НЕТ → БЛОКЕР) / форматы: PyTorch, ONNX-экспорт возможен (IResNet), ONNX→TensorRT прямой / тяжёлый edge-fit, домен web-фото (MS1M).**

Плюсы:
- Высокая SOTA-точность верификации (по заявлению авторов CVPRW2022, **не измерено нами**); код доступен для академического воспроизведения.
- Стандартный IResNet backbone — технически конвертируем в ONNX/TensorRT.

Минусы:
- **БЛОКЕР:** CC BY-NC-SA 4.0 (non-commercial, подтверждено README репо: «licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license») — коммерческий деплой кода и весов запрещён, включая ShareAlike-вирусность.
- Тяжёлый R100, обучён на отозванном MS1MV2 — провенанс-риск данных.
- Domain gap к automotive POV/IR (**не измерено нами**).

Применимость в пайплайне CARS: **NOT_RECOMMENDED** для прода (NC-лицензия). Только как академический референс точности.
Edge / Jetson Orin Nano: R100 тяжёлый (в FP16/INT8 проходит), не самый edge-эффективный + лицензия NC. **Не измерено нами.**

> **NVIDIA NGC (проверено первоисточником):** специальной FACE-EMBEDDING модели в NGC НЕТ. NGC «FaceNet» = DetectNet_v2 **ДЕТЕКТОР** лиц (под NVIDIA Model EULA), для эмбеддинга непригоден. ReIdentificationNet = person-reID (тело по ResNet), не лицо. Edge-каталог NVIDIA задачу эмбеддинга лиц не закрывает.

## Сводная таблица аналогов

| Модель | Год | Архитектура | Параметры/размер | Лицензия (комм.?) | Форматы (ONNX/TRT) | Edge-fit (Orin Nano) | Домен-fit | Вердикт |
|---|---|---|---|---|---|---|---|---|
| **SFace (OpenCV Zoo)** | 2021 | MobileFaceNet + SFace loss | ~1–1.2M / ONNX ~37MB + int8 | **Apache-2.0 на код+ВЕСА — ДА** | готовый ONNX (+int8), ONNX→TRT прямой | **отличный** (FP16/INT8) | web-фото, gap к POV/IR | **primary** |
| **dlib face recognition** | 2017 | ResNet-34-reduced (29 conv) | ~21.4MB .dat | веса public domain + код Boost-1.0 — **ДА** | **нет ONNX/TRT-пути** | мал, но CPU-only, без TRT | web-фото, gap к POV/IR | marginal (нет ONNX) |
| **FaceNet (sandberg/timesler/py-feat)** | 2015–18 | Inception-ResNet-v1 | ~23M / ~110MB | код MIT, ВЕСА **UNCLEAR** (VGGFace2/CASIA) | ONNX (py-feat), TRT возможен | средний (FP16), сложнее граф | web-фото, gap к POV/IR | marginal (legal review) |
| **GhostFaceNets** | 2022 | GhostNet + Subcenter-ArcFace | единицы млн | КОД MIT, ВЕСА **UNCLEAR** (MS1M) | TF→ONNX, ONNX→TRT реалистичен | отличный | web-фото (MS1M), gap к POV/IR | marginal (код MIT; веса legal review) |
| **AdaFace** | 2022 | quality-adaptive margin / IResNet | R18→R100 (~65M) | КОД MIT, ВЕСА **UNCLEAR** (MS1M/WebFace) | ONNX (IResNet), TRT прямой | R18/R50 ок, R100 тяж.; **вход BGR** | web-фото, quality-adaptive | marginal (legal review) |
| **ONNX/OpenVINO ArcFace-R100** | 2018–19 | ArcFace LResNet100E-IR | ~65M / ~250MB | формально Apache, провенанс **UNCLEAR** (MS1M) | готовый ONNX, ONNX→TRT прямой | тяжёлый (FP16/INT8) | web-фото (MS1M), gap к POV/IR | marginal (provenance legal review) |
| **InsightFace packs (buffalo_l/r50, antelopev2)** | 2019–21 | ArcFace / R50–R100 | r50 ~166–174MB; buffalo_s лёгкий | код MIT, ВЕСА **NON-COMMERCIAL — НЕТ** | готовый ONNX, отлажен TRT/DeepStream | buffalo_s отлично; R50 в FP16 | web-фото, SOTA, gap к POV/IR | **not_recommended** (NC) |
| **EdgeFace (Idiap)** | 2023 | EdgeNeXt (CNN+Transformer)+low-rank | XXS ~1.24M, 94.72 MFLOPs | код MIT, ВЕСА **CC BY-NC-SA — НЕТ** | PyTorch→ONNX, TRT (трансформер) | лучший по ресурсам, но NC | web-фото, gap к POV/IR | **not_recommended** (NC) |
| **ElasticFace (fdbtrs)** | 2022 | Elastic margin / ResNet-100 | ~65M, тяжёлый | **CC BY-NC-SA — НЕТ** | ONNX (IResNet), TRT прямой | тяжёлый, NC | web-фото (MS1M), gap к POV/IR | **not_recommended** (NC) |

## Рекомендации

**Главный вывод.** Открытой face-embedding модели, ОДНОВРЕМЕННО (а) с пермиссивной лицензией **НА ВЕСА**, (б) edge-оптимальной (~<2M params), (в) валидированной на automotive/IR-домене — **не существует**. Лучший доступный компромисс — SFace; домен под него придётся дообучать.

**Primary (в прод сразу):**
- **SFace (OpenCV Zoo, Apache-2.0)** — единственная модель с подтверждённой коммерческой лицензией НА ВЕСА + готовым ONNX + int8-quant + прямым ONNX→TensorRT + минимальным edge-бюджетом. Берётся как baseline-эмбеддинг (SGIE поверх FaceDetect), сразу конвертируется в TensorRT FP16/INT8. Точность на целевом бортовом домене (POV через стекло, 5–50м, day/night/IR) **нами не измерена** — обязательна собственная валидация и, скорее всего, дообучение.

**Secondary / референс:**
- **dlib** (public domain) — лицензионно-чистый эталон для оффлайн-сравнения качества, но **без ONNX-пути** → в бортовой TensorRT-пайплайн не идёт.
- **FaceNet (py-feat ONNX), AdaFace, ONNX/OpenVINO ArcFace-R100, GhostFaceNets** — технически удобны и/или edge-пригодны, но веса либо без явного SPDX, либо обучены на research-only/отозванных датасетах (VGGFace2/CASIA/MS1M). Все они — **только R&D-референс**; в прод не брать до **legal review** цепочки лицензий весов.

**Лицензионные предостережения (БЛОКЕРЫ):**
- **NON-COMMERCIAL веса = запрет для прода:** InsightFace (NC research), EdgeFace (CC BY-NC-SA), ElasticFace (CC BY-NC-SA). Код-MIT у InsightFace/EdgeFace — типичная ловушка, на веса не распространяется.
- **UNCLEAR = по умолчанию запрещено до legal review:** FaceNet (нет SPDX на веса + VGGFace2/CASIA research-only), AdaFace (нет явной лицензии весов + MS1M/WebFace), ArcFace-R100-дамп («Apache на упаковке» не перекрывает MS1M/InsightFace-провенанс), GhostFaceNets (КОД MIT чист, но веса MS1M-провенанс).
- **MS-Celeb-1M (MS1MV2/MS1MV3) отозван Microsoft** — любые веса с этим провенансом несут legal/этический риск даже при формальной «Apache»-метке дампа.

**Edge-предостережения:** R100-модели (ArcFace-R100, ElasticFace, antelopev2) проходят на Orin Nano только в FP16/INT8 и съедают ресурсы рядом с детекторами; предпочтительны MobileFaceNet/GhostNet/EdgeNeXt-класс (SFace, buffalo_s, EdgeFace-XXS, GhostFaceNets). Гибридные трансформер-блоки EdgeFace усложняют ONNX→TensorRT. У AdaFace — **вход BGR**, легко ошибиться в препроцессинге. У FaceNet вход не согласован между дистрибуциями (160×160 vs 112×112) — фиксировать чекпойнт.

**Если готового prod-ready аналога нет — путь дообучения (целевой план):**
1. **Baseline в прод:** SFace (Apache-2.0) → TensorRT, немедленно.
2. **Целевой план:** взять КОД с пермиссивной лицензией (SFace / EdgeFace-MIT-код / AdaFace-MIT-код / GhostFaceNets-MIT-код) и **дообучить на лицензионно-чистых + собственных бортовых данных** → чистые веса под automotive/IR. Подходящего открытого датасета с identity-метками для этого нет — см. [08_faceembed_widerface.md](../dataset_research/08_faceembed_widerface.md) (WIDER FACE для эмбеддинга непригоден — детекционный, без идентичностей; единственный кандидат без privacy-блокера — синтетический DigiFace-1M, но он R-UDA non-commercial). Биометрия → privacy/legal до сбора собственных данных.
3. **Альтернатива:** купить коммерческую лицензию InsightFace для максимальной точности из коробки.

Любые NC-веса (InsightFace, EdgeFace, ElasticFace) и unclear-веса (FaceNet, AdaFace-готовые, ArcFace-R100-дамп, GhostFaceNets-готовые) в прод **НЕ брать** без legal review.

## Ссылки

- SFace (OpenCV Zoo): https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface — LICENSE (Apache-2.0): https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE — оригинал SFace: https://github.com/zhongyy/SFace — paper: https://arxiv.org/abs/2205.12010
- dlib face recognition: https://github.com/davisking/dlib-models — API: http://dlib.net/face_recognition.py.html — blog (high-quality face recognition): https://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html
- FaceNet (TF оригинал): https://github.com/davidsandberg/facenet — facenet-pytorch (timesler): https://github.com/timesler/facenet-pytorch — py-feat/facenet (HF, ONNX): https://huggingface.co/py-feat/facenet
- GhostFaceNets: https://github.com/HamadYA/GhostFaceNets — LICENSE (MIT): https://github.com/HamadYA/GhostFaceNets/blob/main/LICENSE
- AdaFace: https://github.com/mk-minchul/AdaFace
- ONNX Model Zoo ArcFace: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface — OpenVINO OMZ (face-recognition-resnet100-arcface-onnx): https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/face-recognition-resnet100-arcface-onnx/README.md
- InsightFace: https://github.com/deepinsight/insightface — лицензирование recognition: https://www.insightface.ai/solutions/face-recognition-licensing — веса (HF): https://huggingface.co/public-data/insightface/blob/main/models/buffalo_l/w600k_r50.onnx
- EdgeFace (Idiap): https://huggingface.co/Idiap/EdgeFace-XXS — https://huggingface.co/Idiap/EdgeFace-S-GAMMA — код: https://github.com/otroshi/edgeface — model card: https://huggingface.co/Idiap/EdgeFace-XXS/resolve/main/README.md?download=true
- ElasticFace (fdbtrs): https://github.com/fdbtrs/ElasticFace
- Локальные спеки: `docs/about_models/face_embedding.md`, `models.md`; датасеты-аналоги: `docs/dataset_research/08_faceembed_widerface.md`

## История изменений

- 2026-06-05 — создан в рамках исследования открытых моделей-аналогов стека CARS (многоагентный workflow, веб-поиск по первоисточникам, лицензии перепроверены адверсариально). Подтверждено первоисточником: SFace = Apache-2.0 на веса (primary); dlib = public domain + Boost-1.0 (нет ONNX → marginal); InsightFace/EdgeFace/ElasticFace = NC-веса (not_recommended); FaceNet/AdaFace/ArcFace-R100-дамп/GhostFaceNets = unclear-веса (legal review). **Исправление:** GhostFaceNets LICENSE — это MIT (не CC BY-NC-ND, как в черновике; CC относится только к статье на ResearchGate) → not_recommended повышен до marginal, остаётся риск провенанса весов (MS1M). NVIDIA NGC face-embedding модели нет (NGC «FaceNet» — детектор). Domain gap к бортовому POV/IR зафиксирован для всех аналогов, нами не измерен.
