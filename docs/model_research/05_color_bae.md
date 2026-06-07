# Аналоги модели «bae_model_f3» для задачи «Color — классификация цвета кузова»

> **См. также:** спецификация исходной модели — [bae_model_f3.md](../about_models/bae_model_f3.md) · реестр моделей — [models.md](../../models.md) · исследование датасетов-аналогов (та же задача) — [05_color_madcars.md](../dataset_research/05_color_madcars.md) · сводка исследования моделей — [00_SUMMARY.md](00_SUMMARY.md)

## Контекст и требования

Целевая модель `bae_model_f3` — кастомный (внутренний) классификатор **цвета кузова ТС на 15 классов** (`beige, black, blue, brown, gold, green, grey, orange, pink, purple, red, silver, tan, white, yellow`). Архитектура — **EfficientNet-B3** (MBConv + Squeeze-and-Excitation + Swish/SiLU), сверено непосредственно по ONNX-файлу; legacy-доки ошибочно называют backbone «ResNet». Вход — `[N, 3, 384, 384]` RGB NCHW, выход — `[N, 15]` **сырых логитов** (softmax не зашит, применяется в постобработке). Препроцессинг: BGR→RGB, `/255`, `(x − mean)/std` с нестандартными `mean=[0.43, 0.40, 0.39]`, `std=[0.27, 0.26, 0.26]`. Размер FP32-графа ≈70 MB. В конвейере CARS модель работает как **Python-сервис после детекции/кропа ТС** (детектор/трекер отдаёт crop автомобиля → сервис нормализует → выдаёт один из 15 цветов с уверенностью).

Ключевой блокер исходной модели: **веса лежат вне версионируемого репозитория CarCV-metrics, на NGC отсутствуют, лицензия неизвестна**. В валидационной кампании Color помечен `UNDEF` («нет модели»). Поэтому поиск аналогов — это не «улучшение бейзлайна», а попытка либо найти коммерчески чистую готовую замену, либо обосновать путь дообучения собственной модели (рекомендация FINAL_REPORT №4).

Целевое устройство — **NVIDIA Jetson Orin Nano 8GB**, рантайм ONNX Runtime GPU, в перспективе ONNX→TensorRT (FP16/INT8) для DeepStream-конвейера. Поэтому штатный формат аналога важен: модель, распространяемая только как OpenVINO IR, на нативный стек Jetson ложится плохо. Целевой домен — **бортовой/придорожный POV** с уровня дороги (day/night/IR, дистанция и углы реальной эксплуатации), регион RU/UA. Цвет как признак гео-нейтрален (RU/UA-специфики на уровне таксономии нет), но **условия съёмки (бортовой POV, ночь, ИК-подсветка) формируют domain gap** относительно обучающих доменов всех найденных аналогов (трафик-камеры, барьеры, web-каталоги, ReID-камеры).

**КОММЕРЧЕСКАЯ лицензия — жёсткое требование** (CARS — коммерческий продукт). По умолчанию любая лицензия с пометкой non-commercial / academic-only / EULA / «не указана» считается **блокером** до legal review. Отдельно фиксируем ловушку «MIT/Apache на код, но отдельная (или NC) лицензия на веса» и ловушку «Apache-ярлык на веса не отменяет NC-ограничения обучающих данных».

**Что требуется от аналога:**
- Покрытие таксономии 15 классов CARS (или максимально близкое подмножество); приоритет дефицитным `tan`, `gold`, `pink`, `purple`, разделению `grey`/`silver`.
- Вход — кроп ТС под прямой инференс (классификация на изображении-кропе), совместимый с пайплайном «детектор → crop → color».
- Штатный или хотя бы реальный путь экспорта **ONNX→TensorRT** для Jetson; OpenVINO-only — минус.
- **Явно коммерчески-чистая лицензия и на код, и на веса, и на обучающие данные.**
- Домен ближе к бортовому POV (day/night/IR); любой иной домен — фиксируемый риск падения точности.

## Обзор аналогов

### timm EfficientNet-B3 (дообучение собственного 15-классового классификатора)
**2019 (арх.) / EfficientNet-B3 (MBConv + SE + SiLU) / ~12M params backbone, ~46–49MB FP32 (близко к целевым ~70MB при 384px и доп. голове) / лицензия кода Apache-2.0, веса итоговые — собственность CARS (комм.: UNCLEAR — ImageNet-оговорка претрейна) / PyTorch → ONNX → TensorRT (FP16/INT8) / точное соответствие целевой модели, штатный edge-путь**

Плюсы:
- Единственный путь к **ТОЧНОМУ соответствию**: те же 15 классов, тот же вход 384×384, та же архитектура EfficientNet-B3 — drop-in замена `bae_model_f3` с уже известной интеграцией.
- Полный контроль домена (бортовой POV, day/night/IR, RU/UA) и лицензионной чистоты итоговых весов — при условии, что обучающие данные чисты.
- Штатный путь PyTorch → ONNX export → ONNX→TensorRT FP16/INT8 для Jetson; код timm под Apache-2.0 (подтверждено README rwightman/timm). Можно ужать до B0/B1 ради latency.

Минусы:
- **Нет готовых color-весов — требуется обучение.** Ни один доступный датасет (MAD-Cars color, UFPR-VCR 11, Chen 2014 8 классов) не покрывает все 15 классов; `tan`/`gold`/`pink`/`purple` дефицитны.
- **ImageNet-претрейн несёт research/ImageNet-оговорку лицензии** (подтверждено README timm: «It is your responsibility to ensure you comply with licenses of dependent datasets»). Для коммерции нужен legal review ImageNet-претрейна либо старт со случайной инициализации / коммерчески-чистого претрейна. Поэтому `license_commercial_ok = unclear` — блокер до проверки.
- Требует размеченных данных под бортовой POV и редкие классы (beige/tan/gold/silver) — затраты на сбор/разметку. Часть кандидатных датасетов (UFPR-VCR, VeRi) сами NC → на них **нельзя** обучать коммерческую модель напрямую (только внутренняя валидация).

Применимость в пайплайне CARS: **рекомендуемый PRIMARY-путь** (совпадает с рекомендацией FINAL_REPORT №4). Дообучить EfficientNet-B3 (timm) под 15 классов и бортовой POV **только на коммерчески-чистых данных** (Roboflow CC BY после проверки происхождения + собственная разметка; NC-датасеты VeRi/UFPR-VCR/MAD-Cars — лишь для внутренней валидации, не для обучения), экспорт ONNX→TensorRT. Прямая замена `bae_model_f3` с известным интерфейсом (384×384, 15 логитов, softmax в постобработке).
Edge / Jetson Orin Nano: идеально — B3@384 ровно то, что проектировалось под Orin Nano как SGIE/сервис; ONNX→TensorRT FP16/INT8 штатно. Latency на Orin Nano **не измерена нами** (заявленные System Design ~15 мс — аспирационная цифра).

### OpenVINO vehicle-attributes-recognition-barrier-0039
**2018–2019 / компактная CNN (исходный фреймворк Caffe), две головы color(7)+type(4) / 0.626 MParams, 0.126 GFLOPs / Apache-2.0 на код И веса (комм.: ДА) / штатно только OpenVINO IR (.xml/.bin); community-ONNX/TensorRT у PINTO0309 / крошечная, но IR вне нативного стека Jetson**

Плюсы:
- **Чистая лицензия Apache-2.0 на код И веса** (файл LICENSE репозитория OMZ покрывает каталог `models/intel`, без отдельных NC-оговорок; per-model README несёт только дисклеймер о товарных знаках). Intel в community-форуме подтверждал коммерческое использование Intel pre-trained моделей. Один из немногих по-настоящему коммерчески чистых готовых color-классификаторов.
- Крошечная (0.6M params, 0.126 GFLOPs), готовы FP16/FP16-INT8 — почти нулевая нагрузка на Orin Nano.
- Softmax уже зашит в граф; есть готовый barrier-демо пайплайн (детектор + атрибуты); community-ONNX (PINTO0309/PINTO_model_zoo, MIT) даёт путь на TensorRT.

Минусы:
- **Только 7 цветов** (`white, gray, yellow, red, green, blue, black`) вместо 15 — отсутствуют `beige/brown/gold/orange/pink/purple/tan` и разделение `grey/silver`. Покрытие классов CARS неполное.
- **Штатно только формат IR**; официального ONNX→TensorRT из OMZ нет. Есть лишь **НЕофициальная** community-конвертация (PINTO0309), не гарантирующая точность/сопровождение — для DeepStream без OpenVINO нужна доп. работа.
- Домен трафик-камеры/барьера (вид сверху/сбоку на барьере), **не бортовой POV CARS** — ожидаем domain gap. Заявленная Intel точность 81.15% color avg на их тест-сете (**не измерено нами**); классы yellow/green слабые (до 54%). RU/UA-специфики нет.

Применимость в пайплайне CARS: кандидат как **чистый коммерческий бейзлайн для 7 базовых цветов** ИЛИ как teacher/референс при дообучении PRIMARY-модели. Не закрывает 15-классовую задачу. Маппинг: 7 классов → подмножество CARS (`gray`→`grey`, без `silver`-разделения). Хорош для быстрого PoC.
Edge / Jetson Orin Nano: по размеру идеален (0.6M params, INT8 готов), но штатный формат IR/OpenVINO плохо ложится на нативный стек Jetson (TensorRT/DeepStream). Через community-ONNX (PINTO) TensorRT-путь технически достижим — **не измерено нами**.

### OpenVINO vehicle-attributes-recognition-barrier-0042
**2018–2019 / CNN-классификатор (исходник PyTorch), две головы color(7)+type(4) / 11.177 MParams, 0.462 GFLOPs / Apache-2.0 на код И веса (комм.: ДА) / штатно только OpenVINO IR; официального ONNX/TensorRT нет / лёгкая, но IR вне нативного стека Jetson**

Плюсы:
- **Apache-2.0 на код и веса** (тот же LICENSE репозитория OMZ, каталог `intel/` без отдельных NC-оговорок; per-model README — только дисклеймер о товарных знаках). Intel подтверждал коммерческое использование Intel pre-trained моделей. Коммерчески чисто.
- Чуть выше точность, чем 0039: color avg 82.71%, black до 96.84% (по заявлению Intel — **не измерено нами**).
- Готовый softmax-выход и пайплайн security-barrier.

Минусы:
- **7 цветов вместо 15** — то же неполное покрытие классов CARS (нет `beige/brown/gold/orange/pink/purple/tan`, нет разделения `grey/silver`).
- **Штатно формат IR без официального ONNX→TensorRT** — для нативного Jetson DeepStream нужна смена рантайма на OpenVINO либо ручная конвертация. У 0042 (в отличие от 0039) явной готовой community-ONNX в данных нет.
- Домен трафик-камеры, **не бортовой POV**; yellow слабый (61.5%, по заявлению Intel — **не измерено нами**). RU/UA-специфики нет.

Применимость в пайплайне CARS: альтернатива 0039 при чуть большем бюджете и потребности в точности по 7 базовым цветам. Те же системные ограничения (формат IR, домен барьера, 7 классов). Роль — secondary-бейзлайн/PoC.
Edge / Jetson Orin Nano: 11M params — всё ещё легко для Orin Nano, INT8 доступен, но штатный формат IR вне нативного стека Jetson. Пригодность под TensorRT **не измерена нами**.

### PaddleClas PULC vehicle_attribute (PPLCNet_x1_0)
**2021–2022 / PPLCNet_x1_0 (лёгкий CNN) + multi-label голова (color 9 + type 10) / ~7.2M (масштаб PPLCNet_x1_0) / код+веса Apache-2.0, НО обучено на VeRi-776 — NC-датасет (комм.: UNCLEAR — реальный блокер) / Paddle inference, штатный Paddle2ONNX→ONNX→TensorRT / лёгкая, реальный edge-путь, но NC-происхождение весов**

Плюсы:
- **Реальный, документированный путь Paddle2ONNX → ONNX → TensorRT** — совместим со штатным Jetson-стеком (в отличие от OpenVINO IR). Есть Paddle Lite для embedded.
- Лёгкая edge-модель (~7.2M), готовые inference-веса в открытом прямом доступе (bcebos), код/веса под Apache-2.0.
- **9 цветов** (`yellow, orange, green, gray, red, blue, white, golden, brown`) — больше, чем у OpenVINO (7), включая `brown` и `golden` — ближе к набору CARS.

Минусы:
- **БЛОКЕР: веса обучены на VeRi-776 — подтверждённо НЕКОММЕРЧЕСКОМ датасете** (доступ по email-запросу, прямое требование «non-commercial purposes», запрет передачи третьим лицам). Apache-ярлык на веса **не отменяет** NC-ограничения обучающих данных → коммерческое распространение/использование весов юридически спорно. `license_commercial_ok = unclear` — реальный блокер до legal review.
- VeRi-домен (трафик-ReID, вид сверху/сбоку) ≠ бортовой POV CARS; ожидаем падение точности. Заявленная авторами точность атрибута ~90.81% mA на VeRi (**не измерено нами**).
- **9 цветов вместо 15** (нет `beige/pink/purple/tan` и отдельного `silver`); multi-label формат требует адаптации под single-label 15-классов. Вход RGB 256×192 (по конфигу PULC) ≠ целевые 384×384.

Применимость в пайплайне CARS: технически сильный готовый edge-классификатор 9 цветов с реальным ONNX→TensorRT путём, но из-за NC-происхождения весов (VeRi) для коммерции CARS пригоден **только после юридического подтверждения**. Безопаснее использовать как референс/учитель или переобучить на чистых данных. Роль — secondary (с оговоркой по лицензии).
Edge / Jetson Orin Nano: хорошо — PPLCNet спроектирован под CPU/edge (~7.2M), штатный ONNX→TensorRT. Подходит как SGIE/сервис. Конкретная latency на Orin Nano **не измерена нами**.

### nikalosa/Vehicle-Make-Color-Recognition (ResNeXt-101)
**~2020 / ResNeXt-101-32x8d + кастомная FC-голова цвета (2048→512→9) / ~88M params backbone (тяжёлый) / код MIT, color-весов НЕТ в репо, разметка через проприетарный Sighthound API (комм.: UNCLEAR) / PyTorch → ONNX → TensorRT возможен, но весов нет / тяжёлый для edge, web-домен СНГ**

Плюсы:
- MIT-лицензия на код; чистый пример transfer-learning пайплайна цвета на PyTorch (голова цвета `LINEAR(2048→512)+ReLU+BN+LINEAR(512→9)`).
- Данные цвета собраны с myauto.ge (грузинский авто-портал) — регион СНГ, потенциально ближе к RU/UA по составу авто, чем западные датасеты (небольшой гео-плюс).

Минусы:
- **Тяжёлый ResNeXt-101 (~88M params)** — для Orin Nano избыточен по латентности/памяти без замены backbone.
- **Готовых выложенных color-весов в репо НЕТ** (подтверждено: только Jupyter-ноутбуки обучения) → реального артефакта для деплоя нет, модель надо обучать заново.
- Домен web-объявлений (myauto.ge), **не бортовой POV**; разметка получена через **проприетарный Sighthound API** → коммерческое использование производных меток + лицензия исходных данных юридически неясны → коммерческий риск. `license_commercial_ok = unclear`.

Применимость в пайплайне CARS: только как **код-референс методики** (transfer learning, голова цвета). Для деплоя нужна замена backbone на лёгкий и переобучение под бортовой домен / 15 классов на чистых данных. Роль — marginal.
Edge / Jetson Orin Nano: ResNeXt-101 заметно тяжелее EfficientNet-B3 — без замены backbone для Orin Nano не оптимален. Не оценивался (весов нет) — **не измерено нами**.

### UFPR-VCR dataset + benchmark (Toward Enhancing VCR in Adverse Conditions)
**2024 / бенчмарк нескольких backbone (EfficientNet-V2 / MobileNet-V3 / ResNet-34 / ViT-b16) — это ДАТАСЕТ+эксперименты, готовых весов НЕТ / зависит от backbone / academic-only non-commercial, по подписанному agreement (комм.: НЕТ — блокер) / только данные; модель обучается самостоятельно → ONNX→TensorRT / лучший доменный fit (day/night/adverse), но NC**

Плюсы:
- **Лучший домен-fit среди найденного:** явно включает nighttime / adverse lighting / окклюзии и фронт/зад виды — близко к бортовому POV CARS (day/night/IR). 10039 изображений, 9502 ТС.
- **11 цветов** (`beige, black, blue, brown, gray, green, orange, red, silver, white, yellow`) с silver/beige/brown/orange — широкое пересечение с набором CARS; ценный ресурс для исследовательской валидации.

Минусы:
- **БЛОКЕР для коммерции:** датасет **ПОДТВЕРЖДЁННО non-commercial / academic-only** (README: «released for academic research only … non-commercial purposes»; доступ только по подписанному licensing agreement с университетской почты). Обучать коммерческую модель CARS на этих данных **НЕЛЬЗЯ** — только внутренняя валидация. `license_commercial_ok = no`.
- Это **датасет/бенчмарк, а не готовые веса** — модель надо обучать. Нет 4 классов CARS (`gold/pink/purple/tan`).
- Регион Бразилия (не RU/UA), хотя условия съёмки релевантны.

Применимость в пайплайне CARS: только как **НЕКОММЕРЧЕСКИЙ исследовательский/валидационный ресурс** по day/night/adverse-домену (нельзя в обучении коммерческой модели). Для PRIMARY-пути нужны коммерчески-чистые данные; UFPR-VCR — лишь для оценки робастности до поставки. Роль — marginal. См. также разбор датасета в [05_color_madcars.md](../dataset_research/05_color_madcars.md).
Edge / Jetson Orin Nano: n/a (данные). При обучении MobileNet-V3 / EfficientNet-V2-S на этих данных результат отлично подходит Orin Nano — но это не коммерчески-допустимый путь обучения.

### Roboflow Universe car-colors classification projects (CC BY 4.0)
**2022–2024 / Roboflow Train / лёгкие классификационные CNN (пользователь обучает и экспортирует) / зависит от выбора модели / CC BY 4.0 на данные/проект, но первичное происхождение картинок не гарантировано (комм.: UNCLEAR) / Roboflow export → ONNX/TensorRT / разнородный crowd-домен, до 13 классов**

Плюсы:
- **CC BY 4.0** (коммерция при атрибуции, подтверждено на страницах проектов) — потенциально единственный коммерчески-допустимый источник **размеченных данных** по цвету среди найденных; в лучших проектах (starco22) до ~13 классов — ближе всего к CARS.
- Готовый ONNX/TensorRT-экспорт через Roboflow; быстрый PoC и пополнение train-сета для дообучения 15-классовой модели.

Минусы:
- CC BY покрывает проект/данные, но **первичное происхождение картинок (web-scrape) и право автора их так лицензировать НЕ гарантированы** → нужна проверка чистоты данных до коммерческого обучения. `license_commercial_ok = unclear`.
- Это в основном **датасеты/auto-train, а не зрелые проверенные веса**; число и набор классов разнятся, **ни один проект не даёт ровно 15 классов CARS**; домен не бортовой (часто web/трафик), RU/UA не гарантирован.

Применимость в пайплайне CARS: наиболее перспективный **коммерчески-допустимый источник размеченных данных** по цвету (CC BY, особенно starco22 ~13 цветов) для дообучения собственного EfficientNet-B3 — при условии проверки происхождения изображений и корректной атрибуции. **Не готовая production-модель.** Роль — marginal (ресурс под PRIMARY-путь).
Edge / Jetson Orin Nano: зависит от выбранной модели; при экспорте в ONNX→TensorRT и лёгком backbone подходит Orin Nano. **Не измерено нами.**

### Spectrico Car Color Classifier (MobileNetV3)
**~2020+ / MobileNetV3 (TensorFlow .pb) / мобильная, вход 224×224 / free-версия testing-only/low-accuracy, full — платная проприетарная (комм.: НЕТ — блокер) / TF .pb; tf2onnx→ONNX→TensorRT теоретически возможен / лёгкая, но коммерческий деплой заблокирован**

Плюсы:
- **14 цветов** (`black, white, grey, silver, blue, red, green, brown, beige, golden, bordeaux, yellow, orange, violet`) — лучшее покрытие набора CARS среди готовых моделей (маппинг `golden`→`gold`, `violet`→`purple`, `bordeaux`≈dark red; нет `pink/tan`).
- Лёгкий MobileNetV3, есть бесплатная версия для быстрого PoC/сравнения.

Минусы:
- **БЛОКЕР: коммерческий деплой бесплатной версии ЗАПРЕЩЁН** (testing/NC only, подтверждено); полная версия — платная проприетарная. `license_commercial_ok = no`.
- **Закрытые проприетарные веса**, нет открытого источника на HF/GitHub/NGC; нет гарантий по reproducibility/лицензии.

Применимость в пайплайне CARS: только как **референс-бенчмарк набора классов** (14 цветов близки к CARS) и идея маппинга. Для коммерческой поставки CARS не пригоден без покупки лицензии. Роль — not_recommended.
Edge / Jetson Orin Nano: MobileNetV3 хорошо ложится по размеру/latency, но **коммерческий деплой заблокирован лицензией** — edge-пригодность нерелевантна.

### chenxyzj/car-color-classifier-yolo3-python (Spectrico-веса в MIT-обёртке)
**~2019+ / MobileNetV2-классификатор (.pb) + YOLOv3/v4 детектор отдельно / мобильная, вход 224×224 / КОД MIT, но ВЕСА = проприетарный Spectrico .pb (комм.: НЕТ — ловушка «MIT на код, не на веса») / TF .pb; tf2onnx→ONNX→TensorRT возможен / лёгкая, но веса проприетарны + Darknet-зона детектора**

Плюсы:
- Готовый рабочий **код-пример** (детектор + цвет), MIT на сам код — удобно как reference-пайплайн.
- Набор цветов шире 7 (наследует Spectrico).

Минусы:
- **БЛОКЕР: веса классификатора — Spectrico-производные, фактически проприетарные/NC** (коммерческий деплой запрещён). Ловушка «MIT на код, не на веса» подтверждена (отдельной лицензии на `.pb` в репо нет). `license_commercial_ok = no`.
- Связка с **YOLOv3/v4 детектором** тянет отдельные лицензионные риски (Darknet-зона) при заимствовании пайплайна целиком.

Применимость в пайплайне CARS: не пригоден для коммерции из-за весов Spectrico. Может служить только как **пример архитектуры пайплайна** детектор→color-кроп→классификатор. Роль — not_recommended.
Edge / Jetson Orin Nano: MobileNetV2 лёгкий, но коммерческий блок по весам Spectrico + отдельный лицензионный риск YOLO-детектора делают edge-пригодность нерелевантной.

## Сводная таблица аналогов

| Модель | Год | Архитектура | Параметры/размер | Лицензия (комм.?) | Форматы (ONNX/TRT) | Edge-fit (Orin Nano) | Домен-fit | Вердикт |
|---|---|---|---|---|---|---|---|---|
| **timm EfficientNet-B3 (дообучение)** | 2019 (арх.) | EfficientNet-B3 (MBConv+SE+SiLU) | ~12M backbone, ~46–49MB FP32 | код Apache-2.0; веса CARS — UNCLEAR (ImageNet-оговорка претрейна) | PyTorch→ONNX→TensorRT FP16/INT8 (штатно) | идеально (B3@384 под Orin Nano; **не измерено**) | полный контроль домена (бортовой POV, day/night/IR), 15/15 классов | **primary** |
| **OpenVINO barrier-0039** | 2018–19 | компактная CNN (Caffe), color+type | 0.626 MParams, 0.126 GFLOPs | Apache-2.0 (код+веса) — ДА | штатно только IR; community-ONNX/TRT (PINTO0309, MIT) | крошечная, INT8 готов; но IR вне нативного стека (**не измерено**) | трафик/барьер, не бортовой POV; 7/15 цветов | secondary |
| **OpenVINO barrier-0042** | 2018–19 | CNN (PyTorch), color+type | 11.177 MParams, 0.462 GFLOPs | Apache-2.0 (код+веса) — ДА | штатно только IR; офиц. ONNX/TRT нет | лёгкая, INT8 готов; IR вне нативного стека (**не измерено**) | трафик/барьер, не бортовой POV; 7/15 цветов | secondary |
| **PaddleClas PULC vehicle_attribute** | 2021–22 | PPLCNet_x1_0 + multi-label (color 9) | ~7.2M | код/веса Apache-2.0, НО обучено на VeRi-776 (NC) — UNCLEAR (блокер) | Paddle2ONNX→ONNX→TensorRT (штатно) | хорошо (~7.2M, edge-ориент.; **не измерено**) | VeRi трафик-ReID, не бортовой POV; 9/15 цветов | secondary |
| **nikalosa Vehicle-Make-Color-Recognition** | ~2020 | ResNeXt-101-32x8d + FC-голова (9) | ~88M backbone (тяжёлый) | код MIT; разметка Sighthound API — UNCLEAR; весов цвета НЕТ | PyTorch→ONNX→TRT (но весов нет) | тяжёлый для Orin Nano; без замены backbone не оптим. | web-объявления myauto.ge (СНГ), не бортовой; 9 цветов | marginal |
| **UFPR-VCR dataset + benchmark** | 2024 | бенчмарк backbone (EffNet-V2/MNv3/RN34/ViT) — данные, не веса | зависит от backbone | academic-only non-commercial, agreement — НЕТ (блокер) | данные → обучить → ONNX→TensorRT | n/a (данные); при обучении MNv3/EffNet-V2-S подходит | лучший: day/night/adverse/окклюзии; 11/15 цветов; Бразилия | marginal |
| **Roboflow Universe car-colors (CC BY 4.0)** | 2022–24 | Roboflow Train / лёгкие CNN | зависит от выбора | CC BY 4.0 (данные); происхождение картинок — UNCLEAR | Roboflow export → ONNX/TensorRT | зависит от модели; лёгкий backbone подходит (**не измерено**) | crowd web/трафик, не бортовой; до 13 классов | marginal |
| **Spectrico Car Color (MobileNetV3)** | ~2020+ | MobileNetV3 (TF .pb) | мобильная, 224×224 | free=testing/NC, full=платная проприетарная — НЕТ (блокер) | TF .pb; tf2onnx→ONNX→TRT теор. | лёгкая, но деплой заблокирован лицензией | авто-фото/MMR; 14 цветов (лучшее покрытие, но проприет.) | not_recommended |
| **chenxyzj car-color-classifier (Spectrico .pb)** | ~2019+ | MobileNetV2 (.pb) + YOLOv3/v4 | мобильная, 224×224 | код MIT, веса Spectrico проприет. — НЕТ (ловушка код≠веса) | TF .pb; tf2onnx→ONNX→TRT; YOLO — Darknet-зона | лёгкий, но веса заблокированы + риск детектора | авто-фото; набор Spectrico, не бортовой | not_recommended |

## Рекомендации

**Главный вывод (подтверждён адверсариально):** готовых открытых color-классификаторов ровно под 15 классов CARS (`beige/black/blue/brown/gold/green/grey/orange/pink/purple/red/silver/tan/white/yellow`) в коммерчески-чистом открытом доступе **НЕТ**. Все готовые модели дают 7–9 цветов (OpenVINO 0039/0042 — 7, PaddleClas — 9), более широкие наборы либо проприетарны (Spectrico 14), либо упираются в NC-датасет (UFPR-VCR 11), либо это данные, а не веса (Roboflow до 13). Ни одна готовая модель не обучена на бортовом POV с уровня дороги (day/night/IR) — **domain gap фиксируется для всех аналогов**.

**PRIMARY (рекомендуется):** **дообучить собственный EfficientNet-B3 (timm, код Apache-2.0)** под 15 классов и бортовой POV — это совпадает с рекомендацией FINAL_REPORT №4 и даёт drop-in замену `bae_model_f3` (тот же вход 384×384, те же 15 классов, известная интеграция). Обучать **исключительно на коммерчески-чистых данных**: Roboflow CC BY (после проверки происхождения изображений и с корректной атрибуцией) + собственная разметка под редкие классы (`gold/pink/purple/tan`). Экспорт ONNX→TensorRT FP16/INT8. **Лицензионное предостережение:** ImageNet-претрейн timm несёт ImageNet-оговорку — перед коммерческой поставкой нужен legal review претрейна либо старт с коммерчески-чистого претрейна/случайной инициализации; итоговые веса CARS чисты только если чисты все обучающие данные.

**SECONDARY (бейзлайны для сравнения / PoC):**
- **OpenVINO barrier-0039 / 0042** — единственный по-настоящему коммерчески чистый готовый артефакт (Apache-2.0 на код И веса), но только 7 цветов и штатно формат IR. Для Jetson — либо OpenVINO-рантайм, либо community-ONNX (PINTO0309, MIT) → TensorRT. Брать как чистый бейзлайн на 7 базовых цветов и/или teacher.
- **PaddleClas PULC vehicle_attribute** — 9 цветов и штатный ONNX→TensorRT, НО **блокер: веса обучены на VeRi-776 (NC-датасет)** — Apache-ярлык на веса не снимает NC-ограничение данных. Использовать только после legal review; безопаснее — как референс/учитель.

**MARGINAL (ресурсы/референс, не продакшн-веса):** Roboflow CC BY (основной коммерчески-допустимый источник размеченных данных под PRIMARY-путь — после проверки происхождения), UFPR-VCR (лучший доменный fit day/night/adverse, но NC — **только внутренняя валидация, не обучение**), nikalosa (код-референс методики, весов цвета нет, тяжёлый backbone).

**NOT_RECOMMENDED:** Spectrico (проприетарные веса, free=NC) и chenxyzj (ловушка «MIT на код, проприетарные веса Spectrico» + Darknet-зона YOLO) — для коммерческой поставки CARS не пригодны.

**Лицензионные предостережения (адверсариально):**
- Apache-ярлык на веса **не отменяет** NC-ограничения обучающих данных (PaddleClas/VeRi-776) — это реальный блокер, не формальность.
- Ловушка «лицензия на код ≠ лицензия на веса» (chenxyzj: MIT-код, проприетарные веса Spectrico) — проверять веса отдельно.
- `unclear`-лицензии (timm ImageNet-претрейн, Roboflow происхождение картинок, nikalosa Sighthound) считаем **блокерами до legal review**.
- AGPL/Ultralytics-ловушки в самих color-классификаторах нет, но пайплайн chenxyzj тянет YOLOv3/v4 (Darknet-зона) — учитывать при заимствовании пайплайна целиком.

**Domain gap (фиксируем):** все готовые модели обучены на трафик-камерах/барьерах/web/ReID, **не на бортовом POV CARS**; ожидаем падение точности day/night/IR. Лучший доменный ресурс по условиям — UFPR-VCR (day/night/adverse), но он NC и в обучении коммерческой модели запрещён. Это усиливает риск PRIMARY-пути: коммерчески-чистых данных под бортовой POV и редкие классы критически мало → основной путь — Roboflow CC BY (после проверки) + собственная разметка. Детально по данным — см. [05_color_madcars.md](../dataset_research/05_color_madcars.md). **Перед коммерческой поставкой обязателен legal review происхождения и лицензий ВСЕХ данных обучения и претрейна.**

## Ссылки

- timm EfficientNet (docs): https://huggingface.co/docs/timm/models/efficientnet
- timm (pytorch-image-models): https://github.com/huggingface/pytorch-image-models
- rwightman/timm (исходный репозиторий, README про лицензии/ImageNet-оговорку): https://github.com/rwightman/timm
- EfficientNet (Tan & Le, ICML 2019): https://arxiv.org/abs/1905.11946
- OpenVINO vehicle-attributes-recognition-barrier-0039 (README OMZ): https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0039/README.md
- OpenVINO 0039 (docs): https://docs.openvino.ai/2023.3/omz_models_model_vehicle_attributes_recognition_barrier_0039.html
- OMZ LICENSE (Apache-2.0, покрывает models/intel): https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE
- PINTO0309 community-конвертация 0039 (ONNX/TFLite/TensorRT, MIT): https://github.com/PINTO0309/PINTO_model_zoo/tree/main/187_vehicle-attributes-recognition-barrier-0039
- OpenVINO vehicle-attributes-recognition-barrier-0042 (README OMZ): https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-attributes-recognition-barrier-0042/README.md
- OpenVINO 0042 (docs): https://docs.openvino.ai/2023.3/omz_models_model_vehicle_attributes_recognition_barrier_0042.html
- PaddleClas PULC vehicle_attribute (README): https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_vehicle_attribute.md
- PaddleClas PULC config (PPLCNet_x1_0, вход/классы): https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml
- VeRi(-776) project (подтверждение NC-лицензии обучающих данных): https://vehiclereid.github.io/VeRi/
- PaddleClas LICENSE (Apache-2.0): https://github.com/PaddlePaddle/PaddleClas/blob/release/static/LICENSE
- nikalosa/Vehicle-Make-Color-Recognition: https://github.com/nikalosa/Vehicle-Make-Color-Recognition
- UFPR-VCR Dataset (README, non-commercial agreement): https://github.com/Lima001/UFPR-VCR-Dataset/blob/main/README.md
- UFPR-VCR paper (arXiv 2408.11589): https://arxiv.org/html/2408.11589v2
- Roboflow Universe car-colors (starco22, ~13 классов): https://universe.roboflow.com/starco22/car-colors-rwgdq
- Roboflow Universe car-colors (tyler-yonjx): https://universe.roboflow.com/tyler-yonjx/car-colors-1smyc
- Roboflow Universe car-colors (roboflowreed): https://universe.roboflow.com/roboflowreed/car-colors-dabiy
- Spectrico Car Color Classifier: http://spectrico.com/
- Spectrico make/model recognition: http://spectrico.com/car-make-model-recognition.html
- chenxyzj car-color-classifier-yolo3-python (MIT-код, веса Spectrico): https://github.com/chenxyzj/car-color-classifier-yolo3-python
- TheDeveloperMask car-color-classifier-yolo4-python (форк): https://github.com/TheDeveloperMask/car-color-classifier-yolo4-python

## История изменений

- 2026-06-05 — создан в рамках исследования открытых моделей-аналогов стека CARS (многоагентный workflow, веб-поиск по первоисточникам, лицензии перепроверены адверсариально).
