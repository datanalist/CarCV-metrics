# Аналоги модели «nomeroff_ocr» для задачи «OCR — распознавание текста номерного знака»

> **См. также:** спецификация исходной модели — [nomeroff_ocr.md](../about_models/nomeroff_ocr.md) · реестр моделей — [models.md](../../models.md) · исследование датасетов-аналогов (та же задача) — [04_ocr_autoria.md](../dataset_research/04_ocr_autoria.md) · сводка исследования моделей — [00_SUMMARY.md](00_SUMMARY.md)

## Контекст и требования

**Целевая модель — nomeroff_ocr (Nomeroff Net OCR, пресет model_v3.3 RU).** Это распознаватель текста номерного знака архитектуры **CRNN (ResNet-18 + CTC)**, ~27 MB. Вход — numpy-кроп уже выровненной пластины (BGR uint8), выход — строка в RU-алфавите. В конвейере CARS модель работает **сразу после LP Detection** (детектор/выравниватель пластины, см. `nomeroff_lpd.md`): получив прямоугольный кроп номера, она читает однострочный российский регистрационный знак по **ГОСТ Р 50577** — 22-символьный алфавит (10 цифр + 12 кириллических букв, визуально совпадающих с латиницей: А, В, Е, К, М, Н, О, Р, С, Т, У, Х). Результат OCR — это финальный текст номера, который пишется в БД и используется для контроля доступа, мониторинга парковки и логистики.

**Технические ограничения цели.** Развёртывание — на **NVIDIA Jetson Orin Nano 8GB** в реальном времени; целевой рантайм — **ONNX → TensorRT / DeepStream (nvinfer)** с приоритетом FP16/INT8. Текущая nomeroff_ocr поставляется как **Python-сервис на PyTorch .ckpt без локального ONNX/TensorRT-engine** — экспорт под Jetson в проекте CARS не оформлен (TBD). Аналог должен либо иметь готовый ONNX, либо штатно экспортироваться в ONNX и далее в TensorRT, при малом весе engine (OCR-узел работает на коротком кропе строки — latency должна быть минимальной).

**Доменные требования.** Бортовой POV — пластина читается с движущегося автомобиля на дистанции и под углом, в режимах **day / night / IR**, после детектора/выравнивателя. Алфавит — строго **RU/ГОСТ-22** (с возможным расширением на СНГ: UA/KZ/BY/GE — кириллица-как-латиница). Аналог должен покрывать кириллицу либо допускать сужение/переопределение словаря до ГОСТ-22. Заявленная точность текущей модели — char_acc 0.9995 (PASS на AUTO.RIA), но с **риском train-leakage** (модель обучалась на данных AUTO.RIA, на них же измерялась) — это завышенная оценка, не независимая.

**Коммерческая лицензия — жёсткое требование.** CARS — коммерческий продукт. Текущий nomeroff_ocr поставляется под **GPL-3.0** (код подтверждён GitHub API: `repos/ria-com/nomeroff-net` spdx_id=GPL-3.0), а веса наследуют copyleft — это **прямой блокер для проприетарного продукта** и главная причина искать замену. Non-commercial веса (NC/NC-ND), copyleft (GPL/AGPL), EULA-условия и невыясненное происхождение весов/обучающих данных трактуются адверсариально как **блокеры до legal review или переобучения**. Особое внимание — ловушке «MIT/Apache на код, но NC/без-лицензии на веса или обучающие данные».

**Что требуется от аналога:**
- распознавание однострочного номера с выровненного кропа пластины, выход — строка;
- покрытие кириллицы (RU; желательно СНГ) либо переопределяемый словарь до ГОСТ-22;
- **коммерчески разрешённая лицензия и на код, и на веса** (приоритет — permissive SPDX: MIT/Apache-2.0; перепроверять лицензию обучающих данных!);
- готовый ONNX или штатный экспорт ONNX → TensorRT, малый вес engine под бюджет OCR-узла на Orin Nano;
- по возможности — плате-специфичность (обучение на кропах номеров), а не общий scene-text/документный OCR.

> **Дисклеймер.** Все цифры точности ниже — **по заявлению авторов** на их бенчмарках, **не измерено нами** на российских номерах с бортового POV. Опубликованной отдельной plate/full-plate accuracy на РФ-номерах для PaddleOCR-cyrillic, EasyOCR-cyrillic и fast-plate-ocr **не найдено**: все известные цифры — либо общий Cyrillic-текст (PP-OCRv5 cyrillic 80.27% — НЕ номера), либо latency-бенчи автора на GPU (НЕ на Orin Nano). Все edge-latency на Orin Nano для кандидатов **не измерены** в этом проекте — требуют отдельного бенча после экспорта в TensorRT. Лицензии всех кандидатов перепроверены адверсариально по первоисточникам (GitHub API / HF API, июнь 2026).

---

## Обзор аналогов

### PaddleOCR cyrillic_PP-OCRv5_mobile_rec
**2024–2025 / PP-OCRv5 recognition = SVTR (Single Visual Text Recognizer, трансформер строк) + PP-LCNet backbone + CTC (НЕ CRNN, но тоже последовательный распознаватель) / mobile-класс, несколько M параметров, ~10 MB (точный размер на карточке не указан) / Apache-2.0 (комм.: ДА — ПОДТВЕРЖДЕНО и код, и веса) / PaddlePaddle inference нативно, ONNX через paddle2onnx, ONNX→TensorRT рабочий, OpenVINO поддержан / отличный edge-fit, кириллица в домене, но domain gap к ГОСТ-кропам пластин**

Плюсы:
- **Чистая лицензия и КОДА, и ВЕСОВ** — Apache-2.0 подтверждён GitHub API (`repos/PaddlePaddle/PaddleOCR` spdx_id=Apache-2.0) И HF API (`cardData.license=apache-2.0` + tag `license:apache-2.0` для cyrillic_PP-OCRv5_mobile_rec, downloads=1453, файлы `inference.pdiparams` реально доступны). Снимает GPL-блокер текущей nomeroff_ocr; коммерчески безопасно. **Единственный кандидат с подтверждённой чистой лицензией И КОДА, И ВЕСОВ для кириллицы.**
- **Нативная поддержка кириллицы** (RU/UA/BY) — правильный алфавитный домен, в отличие от US-LPRNet/PARSeq (латиница).
- Компактный mobile-вариант (~10 MB) + штатный путь **paddle2onnx → ONNX → TensorRT** и предусмотренная линейкой INT8-квантизация — хорошо ложится на Orin Nano.
- Активный мейнтейнер (Baidu), регулярные релизы, большая экосистема; веса реально качаются с HF.

Минусы:
- **Общий OCR-строк, не специализирован под номера/ГОСТ** — нужен файнтюн на кропах пластин и сужение charset до 22 ГОСТ-символов. Заявленная точность 80.27% — на общем Cyrillic-бенчмарке (текст, **не номера**), реальная plate-accuracy на бортовом POV неизвестна, **не измерена нами**.
- **SVTR+CTC отличается от CRNN** — другой препроцессинг (вход [3,48,320], RGB, паддинг ширины до 320), интеграцию в Python-сервис CARS придётся переписать.
- Словарь — общий кириллический charset PaddleOCR (все рус/укр буквы), **не** кастомный ГОСТ-22 — нужна постфильтрация/переопределение под RU-алфавит.
- Paddle-экосистема (paddle2onnx) добавляет шаг конвертации; возможны редкие edge-кейсы динамической формы.

**Применимость в пайплайне CARS:** **primary-кандидат** на замену GPL-зависимого nomeroff_ocr для RU/UA OCR-ступени после LP Detection. Подтверждённо чистая лицензия (код + веса) + кириллица + edge-дружелюбность. Требует файнтюна на кропах номеров и сужения словаря до ГОСТ-22; интегрируется как Python/ONNX-сервис вместо текущего CRNN.
**Edge / Jetson Orin Nano:** отлично — mobile rec ~10 MB легко влезает в бюджет Orin Nano 8GB; штатная INT8-квантизация; engine FP16/INT8 под Jetson реалистичен; latency на коротком кропе строки минимальна (по заявлению — быстрый на CPU). **Латентность на Orin Nano нами НЕ измерена.**

### PaddleOCR PP-OCRv5_mobile_rec (универсальный) + кастомный RU-файнтюн
**2024–2025 / SVTR + PP-LCNet + CTC (универсальная мультиязычная rec-модель v5, как старт для файнтюна) / ~9.6 MB (mobile rec) / Apache-2.0 (комм.: ДА — ПОДТВЕРЖДЕНО код и веса) / PaddleOCR train→export→paddle2onnx→ONNX→TensorRT (FP16/INT8) / отличный edge-fit; универсальная база, домен закрывается файнтюном под ГОСТ**

Плюсы:
- **Полностью чистая лицензия (Apache-2.0) КОДА (GitHub API) и ВЕСОВ** — HF API `cardData.license=apache-2.0` для PP-OCRv5_mobile_rec, downloads=54594, файлы реально доступны. Файнтюн на собственных лицензионно-чистых RU-данных даёт полностью чистую модель — снимает GPL-блокер.
- Компактная база (~9.6 MB) + штатный фреймворк дообучения под ГОСТ-словарь; **управляемый charset** — можно жёстко зафиксировать 22 ГОСТ-символа, повысив plate-accuracy.
- Целевой edge-стек закрывается без лицензионных рисков: paddle2onnx → ONNX → TensorRT/INT8.

Минусы:
- **Требует РАЗМЕТКИ и обучения на RU-кропах** (затраты на данные/обучение) — не plug-and-play.
- Базовая модель не плате-специфична — **без файнтюна точность на номерах не гарантирована**.
- Сдвиг с CRNN на SVTR+CTC — переписать препроцессинг/интеграцию в Python-сервис CARS.

**Применимость в пайплайне CARS:** **secondary/стратегический** — эталонный коммерчески-чистый ПУТЬ к замене nomeroff_ocr: взять Apache-2.0 PP-OCRv5_mobile_rec (код+веса подтверждены чистыми), дообучить на лицензионно-чистых RU-кропах с фиксированным ГОСТ-словарём 22 символа, экспортировать в TensorRT. Снимает GPL-блокер ценой обучения. По сути это «промышленный» вариант primary-пути PaddleOCR, когда готовый cyrillic-чекпойнт не даёт нужной plate-accuracy.
**Edge / Jetson Orin Nano:** отлично — mobile-база ~10 MB; после файнтюна и INT8-квантизации легко на Orin Nano; короткая строка на входе → низкая latency. **На Orin Nano не измерено.**

### fast-plate-ocr (ankandrew) — cct-xs-v2-global / cct-s-v2-global
**2024–2025 / CCT (Compact Convolutional Transformer) — лёгкий конв.-трансформер с фиксированным числом классификационных голов (max_plate_slots), без CTC; чисто плате-ориентированный OCR / XS/S-варианты, очень малые (~0.47 мс/плейт XS, ~0.68 мс/плейт S на GPU; точное число параметров/MB не указано) / КОД MIT (комм.: ДА), но ВЕСА — UNCLEAR/фактический NC-блокер по обучающим данным / ONNX-native (.onnx через GitHub Releases), штатный экспорт CoreML/TFLite/ONNX, ONNX→TensorRT прямой / идеальный edge-fit и лучший plate-домен, но лицензия данных блокирует готовые веса**

Плюсы:
- **Архитектурно единственный кандидат, специализированный под кропы номерных пластин** — точное совпадение с ролью nomeroff_ocr в пайплайне (OCR после детектора).
- **Код MIT** (подтверждён GitHub API `repos/ankandrew/fast-plate-ocr` spdx=MIT) + **ONNX-native** + субмиллисекундная latency автора — образцовая edge-механика для Orin Nano.
- Global v2 покрывает РФ/Беларусь (датасет Platesmania широко представлен RU/BY) — потенциально сильный domain-fit к РФ/СНГ **при переобучении**.
- Конфигурируемый `alphabet`/`max_plate_slots` + фреймворк дообучения — можно подстроить под ГОСТ-22 на своих данных.

Минусы:
- **БЛОКЕР ПО ДАННЫМ:** базовый Global License Plate Dataset (arXiv 2405.10949) выпущен под **CC BY-NC-ND 4.0 (Non-Commercial + No-Derivatives)** — это НЕ «unclear», а **фактический запрет** коммерческого использования и создания производных, наследуемый опубликованными v2-весами. Готовые v2-веса в коммерческом CARS использовать **НЕЛЬЗЯ**.
- **ССЫЛКИ НА HF-ВЕСА НЕДЕЙСТВИТЕЛЬНЫ:** `ankandrew/cct-xs-v2-global-model` и `cct-s-v2-global-model` возвращают **HTTP 401** и отсутствуют в списке моделей автора на HF; реальный источник весов — **GitHub Releases** (отдельной лицензии на сами файлы весов НЕ объявлено нигде).
- **Нормализация кириллицы в global-модели не документирована** — charset может быть латинизирован, не ГОСТ-нативный; требует фактической проверки выходного словаря.
- Архитектура с фиксированными слотами (не CTC) — иной препроцессинг/постпроцессинг (вход ~70×140), интеграцию в CARS надо писать заново; точность на RU-номерах отдельно не опубликована (**не измерено нами**).

**Применимость в пайплайне CARS:** **secondary** — только как **ДОНОР MIT-КОДА/ПАЙПЛАЙНА для ПЕРЕОБУЧЕНИЯ** собственной модели на лицензионно-чистых RU-кропах с ГОСТ-charset (не Platesmania/NC-ND данные). Это единственный коммерчески-чистый путь. **Готовые v2-веса использовать нельзя** из-за CC BY-NC-ND 4.0 обучающих данных. Понижено с potential-primary до secondary; при невозможности переобучения — неприменим.
**Edge / Jetson Orin Nano:** по механике идеален — XS/S модели субмиллисекундные (бенч автора на GPU), ONNX готов, тривиально в TensorRT FP16/INT8. Спроектированы именно под real-time после детектора пластины. **Latency на самом Orin Nano нами не измерена.**

### EasyOCR (JaidedAI) — cyrillic_g2 recognition model
**~2020–2024 / детектор CRAFT + распознаватель None-VGG-BiLSTM-CTC (VGG/ResNet feature extraction → BiLSTM → CTC) — близкий родственник CRNN / cyrillic_g2 компактна (g2-поколение в разы меньше/быстрее g1, десятки MB, точный размер не на карточке) / КОД Apache-2.0 (комм.: ДА), веса cyrillic_g2 штатно качаются (NC явно не заявлено) / PyTorch нативно; ONNX детектора рабочий, ONNX распознавателя проблемен; ONNX→TensorRT далее стандартный / средний edge-fit, кириллица корректна, но не плате-специфичен**

Плюсы:
- **Чистая лицензия Apache-2.0** (код, подтверждён GitHub API `repos/JaidedAI/EasyOCR` spdx=Apache-2.0); **кириллица из коробки** (вкл. RU/UA/BY/болг/серб/монг); веса cyrillic_g2 штатно качаются (автоскачивание в `~/.EasyOCR/model/`, modelhub / SourceForge-зеркало, HF-зеркало `xiaoyao9184/easyocr`). Снимает GPL-блокер.
- **Архитектура VGG/ResNet+BiLSTM+CTC почти идентична текущей CRNN** — минимальный сдвиг парадигмы при миграции, привычный CTC-декод.
- Зрелый, широко используемый проект, простой Python API.

Минусы:
- **ONNX-экспорт распознавателя проблематичен** (известная проблема — ONNX обрезает части сети; экспорт ДЕТЕКТОРА CRAFT рабочий через PR #478, но rec-узла нет в зоопарке) — риск для пути ONNX→TensorRT на Jetson, требует ручной доработки.
- **Общий OCR-строк, не плате/ГОСТ-специфичен** — требует файнтюна и постфильтрации словаря (charset шире ГОСТ-22).
- Связка CRAFT+rec тяжеловата; для CARS нужен **изолированный rec-узел** (детекцию даёт nomeroff_lpd), что не штатный сценарий EasyOCR.
- Точность на бортовых RU-номерах неизвестна (**не измерено нами**); лицензия обучающих данных cyrillic_g2 явно не задокументирована — **низкий, но ненулевой риск, финальная сверка происхождения данных перед продакшеном желательна** (адверсариально).

**Применимость в пайплайне CARS:** **secondary** — чистая лицензия кода (Apache-2.0) + кириллица + CRNN-родство делают его удобной заменой по алфавиту и парадигме, но нестабильный ONNX-экспорт rec-узла снижает edge-готовность. Подходит как research-бейзлайн / донор архитектуры; для продакшна нужен надёжный экспорт rec-модели.
**Edge / Jetson Orin Nano:** средняя — модель компактна, но связка CRAFT+rec тяжелее для real-time; для CARS нужен только распознаватель. **Главный риск edge — нестабильный ONNX-экспорт rec-модели под TensorRT.** Тайминги на Orin Nano **не измерены нами.**

### PARSeq (baudm) — scene text recognition (parseq-tiny/small/base)
**2022 / Permuted Autoregressive Sequence model — ViT-энкодер + перестановочный авторегрессионный декодер (трансформер), SOTA scene-text / tiny/small/base; tiny самый лёгкий (несколько M) / Apache-2.0 (комм.: ДА — ПОДТВЕРЖДЕНО код и веса tiny) / PyTorch/torch.hub, готовый torchscript_model.bin, TorchScript→ONNX→TensorRT подтверждённый путь (гайд NVIDIA) / tiny edge-пригоден, но авторегрессия дороже CTC; charset латинский — нет кириллицы**

Плюсы:
- **Полностью открытая чистая лицензия** — Apache-2.0 КОДА (GitHub API spdx=Apache-2.0) + ВЕСОВ tiny (HF API `cardData.license=apache-2.0` для parseq-tiny, файлы `pytorch_model.bin` + `torchscript_model.bin` реально доступны; ABINet=BSD, CRNN=MIT внутри репо). Коммерчески безопасно.
- SOTA-точность на scene-text; надёжный **ONNX→TensorRT путь** (есть гайд NVIDIA по scene-text inference optimization), готов torchscript-вес.
- Гибкая архитектура, есть лёгкий tiny-вариант.

Минусы:
- **Charset латинский (94 символа: 62 латиница case-sensitive + 32 пунктуации), кириллицы НЕТ из коробки** — для ГОСТ-RU потребуется ПОЛНОЕ переобучение с новым словарём. Без этого неприменим к RU-номерам.
- **Авторегрессионный декодер дороже CTC по latency** (последовательная генерация) — менее выгоден для real-time edge, чем CRNN/CCT/SVTR-CTC.
- **Scene-text домен, не плате/бортовой POV** — двойной domain gap (домен + алфавит). Вход 32×128 (H×W), patch 8×4.
- Точность на номерах/RU не публиковалась (**не измерено нами**).

**Применимость в пайплайне CARS:** **marginal** — лицензионно чист (код + веса tiny подтверждены) и точен, но требует полного переобучения под кириллицу-ГОСТ + авторегрессия невыгодна на edge. Применим как research-бейзлайн для STR или как backbone-кандидат при готовности обучать с нуля на RU-кропах.
**Edge / Jetson Orin Nano:** tiny-вариант edge-пригоден, но авторегрессионный декод дороже CTC по latency, что хуже для real-time на Orin, чем CRNN/CCT. **Не измерено нами.**

### NVIDIA LPRNet (TAO / NGC)
**~2021+ / лёгкая последовательная CNN (tuple-CNN) + CTC, специально под номерные пластины (baseline18) / deployable_onnx актуальной версии ~110 MB (мод. 27.11.2024); в репо CARS ранний US-вариант ~57.7 MB / NVIDIA TAO Models EULA / AI Product Terms (комм.: UNCLEAR — не SPDX) / deployable ONNX / encrypted .etlt → TAO Deploy → TensorRT (FP16/INT8), нативно DeepStream/Jetson / превосходная edge-механика, но RU-домен ПРОВАЛЕН (уже измерен FAIL)**

Плюсы:
- **Эталонный edge/DeepStream/TensorRT-путь**, нативно под Jetson Orin; помечена «ready for commercial use» / «NVIDIA AI Enterprise Supported» от NVIDIA.
- Очень лёгкая, плате-специфичная CNN+CTC архитектура.

Минусы:
- **US/CN алфавит — НЕ покрывает RU/ГОСТ;** в проекте CARS уже ИЗМЕРЕН FAIL на RU (char_acc 0.5904, plate 0.0621) — именно эту модель nomeroff_ocr и заменил. Без переобучения на RU-данных (TAO finetune) непригоден.
- **Лицензия — NVIDIA TAO Deep Learning Models EULA / AI Product Terms (НЕ открытая SPDX);** коммерческая редистрибуция весов/engine связана условиями EULA (приложение должно иметь material additional functionality, только на системах с NVIDIA GPU, distributable-части доступны лишь вашему приложению; trainable/unpruned-модели деплоятся только после обучения через TAO Toolkit) → адверсариально **unclear до юр.-сверки EULA** под конкретный продукт CARS.
- Формат частично закрытый (encrypted .etlt) + завязка на TAO-тулчейн, а не свободный ONNX.
- Для RU нужен полный TAO-finetune на размеченных RU-кропах (затраты на данные/обучение).

**Применимость в пайплайне CARS:** **not_recommended** в текущем виде (RU-провал подтверждён). Условно применим только как АРХИТЕКТУРА для TAO-дообучения на RU-данных с переопределением charset/CTC-blank — отдельный проект с **обязательной сверкой условий NVIDIA TAO EULA** для коммерческого продукта.
**Edge / Jetson Orin Nano:** превосходная по механике (создан для Jetson/DeepStream, готовый путь в TensorRT FP16/INT8), **НО неприменим по алфавиту для RU.** Тайминги на Orin Nano не измерены нами.

### Nomeroff Net OCR — мультирегиональные backbone (eu / kz / by / ge, иные ResNet)
**текущее поколение / та же CRNN (CNN + рекуррент + CTC), что и текущий nomeroff_ocr; другие пресеты регионов и size-варианты backbone / ~27 MB класс (как RU resnet18), есть легче/тяжелее в зоопарке / GPL-3.0 (комм.: НЕТ — ПОДТВЕРЖДЕНО, copyleft-блокер) / PyTorch .ckpt; ONNX-экспорт через пайплайн nomeroff возможен, но engine в репо CARS отсутствует (TBD) / лёгкая и идеальный RU/СНГ-домен, но лицензия убивает коммерцию**

Плюсы:
- **Идеальный RU/СНГ алфавитный домен и плате-специфичность** (тот же класс, что текущий, расширенные регионы: eu_ua_2004_2015, eu, ru, kz, by, ge — нативная кириллица-как-латиница для пост-СССР); обучен на реальных номерах (AUTO.RIA), близко к бортовому ANPR.
- **Лёгкая CRNN** (~27 MB) — edge-реалистична для Python-сервиса на Orin; прямая совместимость с уже работающим пайплайном CARS.

Минусы:
- **GPL-3.0 (copyleft, подтверждён GitHub API)** — тот же коммерческий блокер, что у текущего nomeroff_ocr; **НЕ решает проблему лицензии**, ради которой ищется замена. Веса наследуют copyleft.
- ONNX/TensorRT-engine под Jetson не оформлен (TBD); нюанс `weights_only=False` / monkey-patch `torch.load`.
- Риск **train-leakage** при оценке на AUTO.RIA; алфавит шире строгого ГОСТ-22 (нужна нормализация).

**Применимость в пайплайне CARS:** **not_recommended** как коммерческая замена (GPL-блокер не снимается). Полезен лишь как **research-бенчмарк точности / донор разметки RU-домена** для обучения чистых моделей (PaddleOCR / fast-plate-ocr).
**Edge / Jetson Orin Nano:** лёгкая (~27 MB CRNN) — реалистична для Python-сервиса на Orin; экспорт под Jetson не готов. Edge-пригодность не реализуема в коммерческом продукте из-за лицензии.

### TrOCR (microsoft) — trocr-small/base/large-printed
**~2021+ / encoder-decoder трансформер: image-трансформер энкодер (DeiT/BEiT) + текстовый трансформер декодер (MiniLM/RoBERTa), авторегрессионная генерация / тяжёлый: small ~62M, base ~334M, large ~558M / КОД MIT (комм.: MIT), но ВЕСА — UNCLEAR (HF license=None) + handwritten под NC IAM-данными / PyTorch/Transformers, ONNX через Optimum затруднён, ONNX→TensorRT нетривиален / плохой edge-fit, нет кириллицы, тройной domain gap**

Плюсы:
- **Код MIT** (подтверждён GitHub API, `unilm` root spdx=MIT); высокая точность на печатном тексте; зрелая экосистема HF/Transformers.
- Гибкость энкодер-декодера — теоретически дообучается под любой алфавит.

Минусы:
- **Лицензия весов НЕ проставлена на HF** — `cardData.license=None` подтверждён HF API для trocr-small-printed, trocr-base-printed (downloads 382k), trocr-large-printed; есть открытый issue «Missing License» (unilm#1620) и HF-дискуссия. → **unclear для коммерции.** Дополнительно: **handwritten-варианты обучены на IAM Database = ТОЛЬКО non-commercial research** → для них коммерция запрещена. Адверсариально — «unclear до явного подтверждения» даже для printed.
- **Слишком тяжёлый** (даже small 62M) + авторегрессионный декод — нереалистичен для real-time edge на Orin Nano 8GB, несоразмерен чтению короткого номера.
- **Базово латиница без кириллицы** — нужен дорогой файнтюн крупной модели под RU-ГОСТ.
- Домен документов (печать/рукопись), не номера / не бортовой POV — большой domain gap (домен + алфавит + вес).

**Применимость в пайплайне CARS:** **not_recommended** для edge-ступени OCR номера: несоразмерен бюджету Orin Nano, лицензия весов unclear (HF license=None подтверждён), нет кириллицы. Только как академический референс точности.
**Edge / Jetson Orin Nano:** плохая — 62–558M параметров + авторегрессия слишком тяжелы/медленны для real-time на Orin Nano. Не рекомендуется как edge-цель.

### OpenALPR (openalpr/openalpr) OCR (Tesseract/OCR-модели)
**legacy / классический ALPR-пайплайн: детект + сегментация + Tesseract-OCR посимвольно (не современный CRNN/трансформер) / лёгкий по символьным моделям, устаревшая парадигма; размер зависит от runtime_data региона / AGPL-3.0 (комм.: НЕТ — ПОДТВЕРЖДЕНО, сетевой copyleft-блокер) / C++ библиотека; нет современного ONNX/TensorRT-пути для OCR / слабый edge-fit, RU не поддержан штатно**

Плюсы:
- Зрелый end-to-end ALPR с региональными конфигами; большое сообщество.
- В принципе можно добавить кастомный регион/символы.

Минусы:
- **AGPL-3.0** (подтверждён GitHub API `repos/openalpr/openalpr` spdx_id=AGPL-3.0; LICENSE = GNU AFFERO GPL v3) — **сетевой copyleft**, коммерческий блокер (включая SaaS/сетевой доступ); коммерческая лицензия — только через Rekor (платно).
- **Устаревшая Tesseract/сегментационная OCR-парадигма** — ниже точность, нет ONNX→TensorRT edge-пути; плохо на грязных/наклонных/IR-кадрах бортового POV.
- **RU из коробки нет** (открытые issue «How to add russia»); добавление региона трудоёмко.
- Плохо ложится на Jetson/DeepStream real-time.

**Применимость в пайплайне CARS:** **not_recommended** — AGPL-блокер + устаревшая OCR-архитектура + нет RU и edge-пути. Не рассматривать для коммерческого CARS.
**Edge / Jetson Orin Nano:** слабая — устаревший сегментационный Tesseract-OCR, нет ONNX→TensorRT, ниже точность на сложных кадрах. Не edge-цель.

---

## Сводная таблица аналогов

| Модель | Год | Архитектура | Параметры/размер | Лицензия (комм.?) | Форматы (ONNX/TRT) | Edge-fit (Orin Nano) | Домен-fit | Вердикт |
|---|---|---|---|---|---|---|---|---|
| PaddleOCR cyrillic_PP-OCRv5_mobile_rec | 2024–2025 | SVTR + PP-LCNet + CTC | ~10 MB (несколько M) | **Apache-2.0 — ДА** (код+веса подтв.) | paddle2onnx→ONNX→TRT; OpenVINO | Отлично (~10 MB, INT8 штатно) | Кириллица в домене, но общий текст, не ГОСТ-кропы (gap) | **primary** |
| PaddleOCR PP-OCRv5_mobile_rec + RU-файнтюн | 2024–2025 | SVTR + PP-LCNet + CTC (универсальная база) | ~9.6 MB | **Apache-2.0 — ДА** (код+веса подтв.) | train→export→paddle2onnx→ONNX→TRT (FP16/INT8) | Отлично (~10 MB + INT8 после файнтюна) | Универсальный; ГОСТ-домен закрывается файнтюном | **secondary** (стратегический путь) |
| fast-plate-ocr (cct-xs/s-v2-global) | 2024–2025 | CCT (конв.-трансформер, фикс. слоты, без CTC) | XS/S, ~0.47–0.68 мс/плейт (GPU); MB не указан | КОД MIT, **ВЕСА UNCLEAR / NC-ND по данным — НЕТ для готовых весов** | ONNX-native (.onnx); ONNX→TRT прямой | Идеален (субмиллисекунда, ONNX готов) | Лучший plate-домен (RU/BY в данных), но кириллица не документирована | **secondary** (только донор MIT-кода для переобучения) |
| EasyOCR cyrillic_g2 | 2020–2024 | CRAFT + None-VGG-BiLSTM-CTC | десятки MB (g2 компактна) | КОД Apache-2.0 — ДА; веса штатно (данные unclear) | rec-ONNX проблемен (обрезка сети); ONNX→TRT далее | Средняя (CRAFT+rec тяжеловата; экспорт rec — риск) | Кириллица корректна, но общий текст, не плате/ГОСТ | **secondary** |
| PARSeq (parseq-tiny/small/base) | 2022 | ViT-энкодер + перестановочный авторегр. декодер | tiny несколько M; base крупнее | **Apache-2.0 — ДА** (код+веса tiny подтв.) | torchscript→ONNX→TRT (гайд NVIDIA) | tiny edge-ок, но авторегрессия дороже CTC | Латиница (94), кириллицы НЕТ; scene-text — двойной gap | **marginal** |
| NVIDIA LPRNet (TAO/NGC) | 2021+ | tuple-CNN + CTC, плате-специфичная | deployable_onnx ~110 MB (ранее US ~57.7 MB) | TAO Models EULA / AI Product Terms — **UNCLEAR** (не SPDX) | deployable ONNX / .etlt→TAO Deploy→TRT (FP16/INT8) | Превосходная механика (нативно Jetson/DeepStream) | US/CN алфавит, RU ПРОВАЛЕН (изм. FAIL char 0.59 / plate 0.06) | **not_recommended** |
| Nomeroff Net OCR (eu/kz/by/ge backbone) | текущее | CRNN (CNN + рекуррент + CTC), как текущий | ~27 MB | **GPL-3.0 — НЕТ** (copyleft, подтв.) | PyTorch .ckpt; ONNX возможен, engine TBD | Лёгкая (~27 MB), экспорт под Jetson не готов | Идеальный RU/СНГ плате-домен, но лицензия блокирует | **not_recommended** |
| TrOCR (trocr-*-printed) | 2021+ | DeiT/BEiT энкодер + RoBERTa-декодер (авторегр.) | small 62M / base 334M / large 558M | КОД MIT; **ВЕСА license=None (UNCLEAR) + NC IAM у handwritten — НЕТ** | ONNX через Optimum затруднён; TRT нетривиален | Плохая (62–558M + авторегрессия — слишком тяжело) | Документы, латиница, не номера/бортовой — тройной gap | **not_recommended** |
| OpenALPR OCR (Tesseract) | legacy | детект+сегментация+Tesseract посимвольно | лёгкий, устаревшая парадигма | **AGPL-3.0 — НЕТ** (сетевой copyleft, подтв.) | C++; нет ONNX/TensorRT-пути для OCR | Слабая (нет ONNX→TRT, устаревший OCR) | RU не штатно; сегментация хуже на грязных/IR-кадрах | **not_recommended** |

---

## Рекомендации

**Primary — PaddleOCR cyrillic_PP-OCRv5_mobile_rec.** Единственный кандидат с **подтверждённой чистой лицензией Apache-2.0 И КОДА (GitHub API), И ВЕСОВ (HF API)** для кириллицы — снимает GPL-блокер текущего nomeroff_ocr и коммерчески безопасен. Нативная кириллица (RU/UA/BY) + компактный mobile-вариант (~10 MB) + штатный путь paddle2onnx → ONNX → TensorRT с INT8 хорошо ложатся на Orin Nano. Это основной кандидат на OCR-ступень после LP Detection. **Оговорка:** это общий OCR-строк, не плате/ГОСТ-специфичен; заявленные 80.27% — на общем Cyrillic-тексте, НЕ на номерах (**не измерено нами**). Для промышленной plate-accuracy потребуется файнтюн на кропах номеров и сужение charset до ГОСТ-22.

**Secondary (стратегический путь) — PaddleOCR PP-OCRv5_mobile_rec + RU-файнтюн.** Эталонный коммерчески-чистый ПУТЬ: взять Apache-2.0 базу (код+веса подтверждены чистыми), дообучить на лицензионно-чистых RU-кропах с жёстко фиксированным ГОСТ-словарём 22 символа, экспортировать в TensorRT FP16/INT8. Снимает GPL-блокер ценой разметки/обучения. По сути это «продакшн-уровень» primary-кандидата, когда готовый cyrillic-чекпойнт не даёт нужной точности на пластинах.

**Secondary — fast-plate-ocr (только MIT-код) и EasyOCR cyrillic_g2.** **fast-plate-ocr** — архитектурно единственный плате-специфичный кандидат (CCT, точное совпадение с ролью nomeroff_ocr) и образцовая edge-механика (ONNX-native, субмиллисекунда), НО его **готовые v2-веса коммерчески неприменимы**: обучающий Global License Plate Dataset под **CC BY-NC-ND 4.0** (Non-Commercial + No-Derivatives) — это блокер по данным, не «unclear». Брать **только MIT-код/пайплайн для переобучения** на собственных (не Platesmania) RU-кропах с ГОСТ-charset. Дополнительно: HF-ссылки на веса мертвы (HTTP 401), реальный источник — GitHub Releases. **EasyOCR cyrillic_g2** — Apache-2.0 код + кириллица + CRNN-родство (минимальный сдвиг парадигмы), но **нестабильный ONNX-экспорт rec-узла** снижает edge-готовность; research-бейзлайн/донор архитектуры, для продакшна нужен надёжный экспорт. Лицензию обучающих данных cyrillic_g2 желательно сверить перед продакшеном (адверсариально).

**Marginal — PARSeq.** Лицензионно чист (Apache-2.0 код + веса tiny подтверждены) и точен на scene-text, но **charset латинский (94), кириллицы нет** — нужен полный претрейн под RU-ГОСТ; авторегрессионный декод невыгоден на edge. Только как research-бейзлайн STR или backbone при готовности обучать с нуля.

**Лицензионные блокеры и edge-провалы (not_recommended).** **NVIDIA LPRNet** — превосходная edge/TensorRT-механика, но US/CN-алфавит → **RU уже доказанно ПРОВАЛЕН в этом проекте** (char_acc 0.5904, plate 0.0621 — именно эту модель nomeroff_ocr и заменил), плюс лицензия — TAO EULA (не SPDX, unclear до юр.-сверки). **Nomeroff Net (мультирегиональные пресеты)** — идеальный RU/СНГ-домен, но **GPL-3.0** (тот же блокер, что у текущего) → не решает задачу. **TrOCR** — лицензия весов **license=None** (unclear) + handwritten на NC IAM-данных, слишком тяжёл (62–558M) и без кириллицы. **OpenALPR** — **AGPL-3.0** (сетевой copyleft) + устаревший Tesseract-OCR без edge-пути + нет RU. Все четыре — максимум внутренний research-эталон или донор архитектуры/разметки.

**Доменный пробел и путь дообучения (честный вывод).** **ИДЕАЛЬНОГО открытого аналога** — «плате-специфичный + нативный ГОСТ-RU-charset + чистая коммерческая лицензия И КОДА, И ВЕСОВ + готовый ONNX/TensorRT» — **НЕ найдено.** Все чистые по весам кандидаты (PaddleOCR, EasyOCR, PARSeq) — общий/scene-text, требуют файнтюна на RU-кропах и сужения словаря до ГОСТ-22; единственный плате-специфичный с RU-данными (fast-plate-ocr) несёт NC-ND-лицензию ОБУЧАЮЩИХ ДАННЫХ, блокирующую коммерческое использование готовых весов. **Рекомендуемый путь:** дообучить чистую Apache-2.0 базу (PP-OCRv5_mobile_rec, лицензия кода+весов подтверждена) на лицензионно-чистых RU-кропах с фиксированным ГОСТ-словарём 22 символа → export paddle2onnx → ONNX → TensorRT FP16/INT8 на Orin Nano. Альтернатива — переобучить MIT-пайплайн fast-plate-ocr на собственных (не Platesmania/NC-ND) данных. Источник чистых RU-кропов и оговорки по train-leakage AUTO.RIA — см. [04_ocr_autoria.md](../dataset_research/04_ocr_autoria.md). Все цифры точности — по заявлению авторов, **не измерено нами**; latency на Orin Nano для всех кандидатов **не измерена** и требует отдельного бенча после экспорта в TensorRT.

## Ссылки

- PaddleOCR cyrillic_PP-OCRv5_mobile_rec (HF): https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec
- PaddleOCR PP-OCRv5_mobile_rec универсальный (HF): https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec
- PaddleOCR (GitHub, Apache-2.0): https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR — получение ONNX-моделей (paddle2onnx): https://www.paddleocr.ai/latest/en/version3.x/deployment/obtaining_onnx_models.html
- PaddleOCR PP-OCRv5 collection (HF): https://huggingface.co/collections/PaddlePaddle/pp-ocrv5
- fast-plate-ocr (ankandrew, GitHub, MIT): https://github.com/ankandrew/fast-plate-ocr
- fast-plate-ocr — Releases (реальный источник весов): https://github.com/ankandrew/fast-plate-ocr/releases
- fast-plate-ocr — Model Zoo (бенчи): https://ankandrew.github.io/fast-plate-ocr/latest/inference/model_zoo/
- Global License Plate Dataset (arXiv 2405.10949, CC BY-NC-ND 4.0): https://arxiv.org/abs/2405.10949
- EasyOCR (JaidedAI, GitHub, Apache-2.0): https://github.com/JaidedAI/EasyOCR
- EasyOCR LICENSE: https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE
- EasyOCR modelhub (cyrillic_g2): https://www.jaided.ai/easyocr/modelhub/
- EasyOCR cyrillic вес (HF-зеркало): https://huggingface.co/xiaoyao9184/easyocr/blob/master/cyrillic.pth
- PARSeq (baudm, GitHub, Apache-2.0): https://github.com/baudm/parseq
- PARSeq LICENSE: https://github.com/baudm/parseq/blob/main/LICENSE
- PARSeq parseq-tiny (HF, Apache-2.0): https://huggingface.co/baudm/parseq-tiny
- NVIDIA — scene text recognition inference optimization (гайд): https://developer.nvidia.com/blog/robust-scene-text-detection-and-recognition-inference-optimization/
- NVIDIA LPRNet (NGC): https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lprnet
- LPRNet (docs TAO): https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/tensorflow_1/character_recognition/lprnet.html
- NVIDIA TAO Toolkit Models EULA: https://developer.download.nvidia.com/licenses/tao_toolkit_21-08_models_eula.pdf
- DeepStream LPR app (NVIDIA-AI-IOT): https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app
- Nomeroff Net (ria-com, GitHub, GPL-3.0): https://github.com/ria-com/nomeroff-net
- Nomeroff Net History (модели/регионы): https://github.com/ria-com/nomeroff-net/blob/master/History.md
- Nomeroff Net (сайт/модели): https://nomeroff.net.ua/
- TrOCR small-printed (HF): https://huggingface.co/microsoft/trocr-small-printed
- TrOCR (microsoft/unilm, GitHub, код MIT): https://github.com/microsoft/unilm/tree/master/trocr
- TrOCR «Missing License» issue (unilm#1620): https://github.com/microsoft/unilm/issues/1620
- TrOCR base-printed — HF-дискуссия по лицензии: https://huggingface.co/microsoft/trocr-base-printed/discussions/15
- OpenALPR (GitHub, AGPL-3.0): https://github.com/openalpr/openalpr
- OpenALPR LICENSE (AGPL v3): https://github.com/openalpr/openalpr/blob/master/LICENSE
- OpenALPR — issue «How to add russia»: https://github.com/openalpr/openalpr/issues/558

## История изменений
- 2026-06-05 — создан в рамках исследования открытых моделей-аналогов стека CARS (многоагентный workflow, веб-поиск по первоисточникам, лицензии перепроверены адверсариально).
