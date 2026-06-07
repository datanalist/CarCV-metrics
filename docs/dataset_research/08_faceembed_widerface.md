# Аналоги датасета «WIDER FACE (исходно указан, но НЕ содержит идентичностей)» для задачи «Face embedding — эмбеддинги/распознавание лиц»

> **См. также:** исходный датасет — [wider_face.md](../about_datasets/wider_face.md) · индекс датасетов — [datasets.md](../../datasets.md) · сводка исследования — [00_SUMMARY.md](00_SUMMARY.md)

> **Статус документа:** аналитический обзор открытых датасетов-аналогов. Все лицензионные пометки проверены по первоисточникам на 2026-06-05 (см. раздел «Ссылки»). По умолчанию лицензия трактуется как `restricted / unclear`, пока коммерческое разрешение не подтверждено явно. **CARS — коммерческий продукт**, поэтому любой датасет с лицензией academic / non-commercial / research-only трактуется как непригодный для production без отдельного legal review или лицензионного соглашения с правообладателем.

## Контекст и требования

**Целевая модель — Face embedding** (планируемая, обучаемая, ArcFace-подобная эмбеддинг-модель распознавания лиц). По спеке `docs/about_models/face_embedding.md` модель ещё не выбрана (`models.md`: `Face embedding | None | None | Trainable`): архитектура, размер входа, размерность вектора, лицензия весов — TBD. Типовой (предположительный) подход — angular-margin metric learning (ArcFace / CosFace / FaceNet-embedding) поверх CNN-backbone (ResNet / MobileFaceNet) с выходом в виде L2-нормированного вектора (обычно 128 или 512 dim) и сравнением по косинусной близости.

**Что нужно от аналога:**
- **Для обучения** — много изображений *на личность* (identity-метки): десятки–сотни фото на персону, тысячи и более идентичностей; это даёт angular-margin разнообразие классов для metric learning.
- **Для валидации** — pair- или template-протоколы (same / different пары) под метрики верификации 1:1 (TAR@FAR, accuracy) и идентификации 1:N.

**Ключевое ограничение исходного датасета.** **WIDER FACE для задачи Face embedding непригоден.** Это **детекционный** бенчмарк: 32 203 изображения / 393 703 размеченных лица, аннотации — только bbox + атрибуты (blur / occlusion / pose / illumination / expression), лицензия **CC BY-NC-ND 4.0**. **В нём НЕТ меток идентичности персон**, поэтому ни обучить, ни валидировать ArcFace-подобный эмбеддинг на нём нельзя. Это прямо зафиксировано локально: `docs/about_datasets/wider_face.md` (раздел «Ограничения») и `docs/about_models/face_embedding.md` (строка 85). WIDER FACE остаётся валиден только для валидации **детектора FaceDetect**, а не эмбеддинга. Привязка `datasets.md` (Face embedding → WIDER FACE) для этой задачи **некорректна** — нужен отдельный датасет с identity-метками.

**Общий domain gap (касается всех аналогов ниже).** Все кандидаты — это веб-/студийные фронтальные фото или видео «в дикой природе». Ни один не покрывает бортовой POV CARS: дистанция 5–50 м, угол 0–30°, лицо через лобовое стекло, day / night / **IR**. Поэтому метрики на них оценивают инвариантность и общее качество эмбеддинга, но финальный automotive-вердикт потребует собственной целевой выборки (включая IR / night), которой в открытых наборах нет.

## Обзор аналогов

### LFW (Labeled Faces in the Wild)
**Год:** 2007 / **Объём:** 13 233 изображения, 5 749 личностей (1 680 человек с ≥2 фото); стандартный protocol — 6 000 пар (3 000 same / 3 000 diff), 10-fold CV / **Лицензия (комм.: unclear → фактически нет):** custom research (UMass); авторы **прямо предупреждают**, что результат на LFW не должен использоваться для вывода о пригодности алгоритма для коммерческих целей / **Формат:** identity-папки + `pairs.txt` (verification); есть выровненные версии (funneled / deepfunneled) / **Домен:** web (новостные фото Yahoo News), в основном фронтальные, дневной свет

- **Плюсы:** де-факто стандартный verification-бенчмарк (сопоставимость с литературой по accuracy / TAR@FAR); маленький и быстрый, чёткий 10-fold protocol — идеален для smoke-валидации; широко зеркалирован (TFDS, Kaggle, HF); производные CALFW / CPLFW добавляют срезы по возрасту / позе.
- **Минусы:** «насыщен» (SOTA >99.8%) → слабая дискриминация современных моделей; только валидация, не обучение; авторы явно предупреждают против выводов о коммерческой пригодности; domain gap к automotive, нет IR / night; известные privacy / этические претензии.
- **Применимость для валидации:** основной кандидат для smoke-валидации верификации (стандартный protocol). Дополнять CALFW / CPLFW / AgeDB-30 / CFP-FP для срезов.
- **Применимость для обучения:** непригоден (мало фото на личность; это бенчмарк).

### AgeDB-30 / CFP-FP / CALFW / CPLFW (verification-срезы)
**Год:** 2016–2018 / **Объём:** AgeDB-30 — 3 000 + 3 000 пар (возрастной разрыв); CFP-FP — 7 000 пар frontal-profile; CALFW — 3 000 + 3 000 пар (age-gap); CPLFW — 3 000 + 3 000 пар (pose-gap) / **Лицензия (комм.: unclear):** research / academic по большинству карточек; для коммерции — unclear / forbidden / **Формат:** identity-метки + готовые pair-листы (same / diff), выровненные кропы; обычно поставляются в составе InsightFace как aligned `.bin`-пакеты (зеркала на Kaggle) / **Домен:** web (селебрити; срезы по возрасту и позе / профилю)

- **Плюсы:** стандартный набор срезов рядом с LFW → сопоставимость и полная картина по осям возраст / поза; CFP-FP / CPLFW прямо тестируют устойчивость к позе / профилю (важно для бортового угла); малые, быстрые, чёткие pair-протоколы; доступны и активно используются (публикации 2024–2025).
- **Минусы:** только валидация; малый объём; не automotive, нет IR / night; лицензии неоднозначные; производность от тех же веб-источников (privacy-вопросы).
- **Применимость для валидации:** рекомендованный дополняющий набор к LFW — AgeDB-30 / CALFW (возраст), CFP-FP / CPLFW (поза / профиль).
- **Применимость для обучения:** непригодны (бенчмарки).

### VGGFace2
**Год:** 2018 / **Объём:** ~3.31 млн изображений, 9 131 личность (8 631 train + 500 test), в среднем ~362 фото на личность / **Лицензия (комм.: НЕТ):** academic / non-commercial; HF-зеркало `ProgramComputer/VGGFace2` помечено **CC BY-NC 4.0** (NonCommercial), оригинальный VGG Face — тоже CC BY-NC 4.0 / **Формат:** identity-метки (папки / CSV) + bbox + 5 keypoints на изображение, мета по личности (пол, поза, возраст) / **Домен:** web (Google Image Search): селебрити / публичные лица, сильная вариативность позы, возраста, освещения, этничности

- **Плюсы:** большой объём identity-меток (9k личностей) — каноничный train-сет для ArcFace / CosFace; богатая вариативность позы и возраста (датасет специально под cross-pose / cross-age); готовые bbox + 5 keypoints упрощают alignment; чище CASIA-WebFace, стандартный baseline-train.
- **Минусы:** **лицензия non-commercial → для production CARS прямое использование запрещено** без отдельного разрешения / legal review; официальные download-ссылки с сайта Oxford **удалены**, доступ только через зеркала (HF ~40 GB, Academic Torrents) с тем же NC-ограничением; domain gap к automotive (нет IR / night / через стекло); биометрия реальных лиц без согласия субъектов → privacy / legal-риски.
- **Применимость для валидации:** ограниченно (500 test-идентичностей + protocol); как research-бенчмарк, не для коммерческого вердикта.
- **Применимость для обучения:** технически каноничный train-сет, но юридически non-commercial → для CARS только R&D / прототип, не production.

### DigiFace-1M (синтетический)
**Год:** 2022 (WACV 2023) / **Объём:** 1.22 млн синтетических изображений, 110 000 идентичностей (720k / 10k id по 72 фото + 500k / 100k id по 5 фото) / **Лицензия (комм.: НЕТ):** Microsoft **R-UDA** — «solely for Computational Use for non-commercial research»; **рендеры синтетические, реальных персон нет** / **Формат:** синтетические identity-метки (папки идентичностей); вариации аксессуаров / углов; кропы уже выровнены (без реальных bbox) / **Домен:** studio / synthetic (компьютерная графика): рендеренные лица фиктивных людей, контролируемое освещение, позы, аксессуары

- **Плюсы:** **нет реальных персон → снимает privacy / биометрические legal-риски** (критично для коммерческого CARS); очень много идентичностей (110k) — большое angular-margin разнообразие классов; полный контроль вариаций (поза / свет / аксессуары) — расширяемо под нужные условия; открыто и воспроизводимо (Microsoft Research).
- **Минусы:** лицензия (R-UDA) **всё равно non-commercial research** → для production нужен legal review / иное соглашение с Microsoft; domain gap синтетика↔реальные лица (только на DigiFace точность на реальных лицах заметно ниже — нужен domain adaptation); не automotive POV; 5 фото / личность для 100k-части — мало без аугментаций.
- **Применимость для валидации:** слабо (синтетика не отражает реальное распределение); годен для sanity-check / абляций.
- **Применимость для обучения:** хорош как pretrain / дополнение реального train-сета; единственный кандидат **без privacy-блокера**, но non-commercial лицензию нужно прояснить с Microsoft перед production.

### CASIA-WebFace
**Год:** 2014 / **Объём:** 494 414 изображения, 10 575 идентичностей (~47 фото / личность), собрано из веба (IMDb) / **Лицензия (комм.: НЕТ):** CASIA Release Agreement — только non-commercial research и образование (academic-use-only подтверждено) / **Формат:** identity-метки (папка на личность); без встроенных bbox / keypoints (alignment сторонними детекторами) / **Домен:** web (IMDb): фото знаменитостей, фронтальные / околофронтальные, шумные метки

- **Плюсы:** полмиллиона изображений с identity-метками — классический baseline train-сет для ArcFace / CosFace; много готовых рецептов / чекпойнтов; достаточно фото на личность (~47) для metric learning; доступен (зеркала Kaggle).
- **Минусы:** строго non-commercial → запрет для CARS без отдельного соглашения; шумные / неточные метки (web-scraping), есть пересечения с тест-сетами (риск утечки); нет встроенных bbox / keypoints; domain gap к automotive; privacy-риски; меньше и грязнее VGGFace2 / Glint360K.
- **Применимость для валидации:** не для валидации (это train-сет).
- **Применимость для обучения:** исторический train-сет; технически работоспособен, но non-commercial и вытеснён VGGFace2. Для CARS — только R&D.

### IJB-C / IJB-B (IARPA Janus Benchmark)
**Год:** 2018 (IJB-C) / 2017 (IJB-B) / **Объём:** IJB-C — 138 000 face-изображений + 11 000 видео + 10 000 non-face, 3 531 субъект; IJB-B — 1 845 субъектов / **Лицензия (комм.: unclear):** custom (NIST / IARPA), доступ по подписанному соглашению / **Формат:** identity-метки, ground-truth bbox + keypoints + covariates; протоколы 1:1 verification и 1:N identification (template-based, still + video) / **Домен:** web + видео «в дикой природе», экстремальные позы / освещение

- **Плюсы:** современный сложный verification / identification-бенчмарк (1:1 и 1:N), не насыщен как LFW; template-based + видео + non-face дистракторы → реалистичная оценка TAR@FAR; детальные covariate-срезы.
- **Минусы (КРИТИЧНО — ДОСТУПНОСТЬ):** **NIST прекратил распространение IJB-A / IJB-B / IJB-C 14 марта 2023 г.**; страница запроса (`nigos.nist.gov/datasets/ijbc/request`) теперь редиректит на программу FRVT → официально датасет **получить нельзя**. Доступ только через неофициальные зеркала с неясным правовым статусом. Коммерческий статус и так был неоднозначен; сложная подготовка протоколов; только валидация.
- **Применимость для валидации:** ранее — сильный «жёсткий» бенчмарк, **сейчас фактически недоступен официально** → исключить из плана (или заменить эквивалентом, доступным легально).
- **Применимость для обучения:** не предназначен (бенчмарк).

### Glint360K (+ экосистема MS1MV2 / MS1MV3 InsightFace)
**Год:** 2021 (Glint360K); 2019–2020 (MS1MV2 / V3) / **Объём:** Glint360K — 17 091 657 изображений, 360 232 идентичности; MS1MV2 — ~5.8M / 85k id; MS1MV3 — ~5.1M / 93k id / **Лицензия (комм.: НЕТ):** research-only / custom; производны от **ОТОЗВАННОГО MS-Celeb-1M** (Microsoft удалил в апреле 2019 за отсутствие согласия субъектов / privacy); чёткой коммерческой лицензии нет / **Формат:** выровненные кропы 112×112 в MXNet RecordIO (`.rec` / `.idx`) + списки; готово под ArcFace / **Домен:** web, крупномасштабно

- **Плюсы:** крупнейший неотозванный (сам Glint360K) открытый train-сет (360k id / 17M img) — лучшее качество эмбеддинга среди открытых; предобработан под ArcFace (минимум усилий на pipeline); эталон для современных ArcFace / AdaFace чекпойнтов.
- **Минусы:** **юридически наиболее проблемный** — частично производен от отозванного MS-Celeb-1M (лица не-знаменитостей без согласия) → серьёзный legal / этический блокер для коммерческого продукта; лицензия research-only без коммерческого разрешения; domain gap к automotive; десятки ГБ, требует значимых GPU-ресурсов.
- **Применимость для валидации:** не для валидации (train-сет).
- **Применимость для обучения:** сильнейший открытый pretrain, **но происхождение от отозванного MS-Celeb-1M делает его непригодным для коммерческого CARS** даже после legal-разбора. Только R&D, и то с осторожностью.

## Сводная таблица аналогов

| Датасет | Объём | Лицензия (комм.?) | Формат аннотаций | Домен-fit | Validation / Training | Вердикт |
|---|---|---|---|---|---|---|
| **LFW** | 13 233 img / 5 749 id; 6 000 пар | research, комм. — **нет** (авторы против комм. выводов) | identity + `pairs.txt` (verification) | низкий (web, фронт., нет IR) | **validation** | **primary (валидация)** |
| **AgeDB-30 / CFP-FP / CALFW / CPLFW** | по 3–7 тыс. пар каждый | research / academic, комм. — **unclear** | identity + pair-листы, aligned кропы | низкий-средний (срезы возраст / поза) | **validation** | secondary (срезы) |
| **VGGFace2** | ~3.31M img / 9 131 id | **CC BY-NC 4.0 — комм. нет** | identity + bbox + 5 kpts | низкий-средний | **training** (+ слабо valid) | secondary (R&D-train) |
| **DigiFace-1M (synthetic)** | 1.22M img / 110k id | R-UDA non-commercial — комм. **нет** | синтетич. identity (aligned) | низкий, но **без privacy-блокера** | **training** (pretrain) | secondary (pretrain без privacy) |
| **CASIA-WebFace** | 494 414 img / 10 575 id | academic-only — комм. **нет** | identity (без bbox / kpts) | низкий | **training** | marginal (вытеснён VGGFace2) |
| **IJB-C / IJB-B** | 138k img + 11k видео / 3 531 субъект | NIST custom — комм. **unclear**; **распространение прекращено 2023-03-14** | identity + bbox + covariates; 1:1 / 1:N | средний | validation | **not_recommended (недоступен)** |
| **Glint360K (MS1MV2 / V3)** | 17M img / 360k id | research-only; от **отозванного** MS-Celeb-1M — комм. **нет** | aligned 112×112 RecordIO | низкий | training | **not_recommended (legal / этика)** |

## Рекомендации

**Главный вывод.** Ни один открытый датасет реальных лиц с identity-метками **не даёт чистого коммерческого разрешения** без отдельного legal review или лицензионного соглашения. VGGFace2, CASIA-WebFace, WebFace260M / 42M — все non-commercial; Glint360K / MS1MV2 / V3 — производны от отозванного MS-Celeb-1M (privacy-блокер); LFW авторы предупреждают против коммерческих выводов; IJB-C официально **снят с распространения NIST (2023-03-14)**. Единственный кандидат без privacy-блокера — синтетический DigiFace-1M, но и его R-UDA — non-commercial research. Биометрия лиц дополнительно требует privacy / legal-проверки (рекомендация 5 в `face_embedding.md`).

**Приоритет для ВАЛИДАЦИИ (smoke-метрики качества эмбеддинга):**
1. **LFW — обязательный smoke-стандарт** (verification 1:1, accuracy / TAR@FAR; сопоставимость с литературой). Приоритет №1.
2. **AgeDB-30 / CFP-FP / CALFW / CPLFW** — дополняющие срезы по возрасту (AgeDB-30, CALFW) и позе / профилю (CFP-FP, CPLFW); доступны как aligned `.bin`-пакеты InsightFace. Приоритет №2.
3. IJB-C — **исключить** из плана: официально недоступен с 2023-03-14. Если нужна «жёсткая» 1:1 / 1:N оценка — искать легально доступный эквивалент или собирать собственный template-протокол.

**Приоритет для ОБУЧЕНИЯ / ДООБУЧЕНИЯ (только R&D, не production):**
1. **DigiFace-1M (synthetic)** — pretrain без privacy-блокера (фиктивные лица), затем domain adaptation на реальных; лицензию R-UDA для production прояснить с Microsoft. Предпочтительный старт именно из-за отсутствия privacy-риска.
2. **VGGFace2** — каноничный baseline-train реальных лиц (чище CASIA), но строго non-commercial → только внутренний R&D / прототип.
3. **CASIA-WebFace** — резервный baseline (marginal; шумнее, вытеснён VGGFace2).
4. **Glint360K / MS1MV2 / V3 — не использовать** для коммерческого CARS: происхождение от отозванного MS-Celeb-1M — серьёзный legal / этический блокер.

**Для production CARS** (снятие privacy-блокера): связка «DigiFace-1M (pretrain) + домен-адаптация» и/или **собственный сбор лиц водителей с согласиями** (как предлагает спека модели). Все перечисленные real-face датасеты трактовать как ВНУТРЕННИЙ research-бенчмарк с пометкой non-commercial; до любого внешнего / продуктового использования — обязательный legal review и/или коммерческое лицензирование. Финальный automotive-вердикт (дистанция 5–50 м, угол, day / night / **IR**) потребует собственной целевой выборки — её в открытых наборах нет.

## Ссылки

- VGGFace2 (офсайт, ссылки удалены): https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ — paper: https://arxiv.org/abs/1710.08092 — HF-зеркало (CC BY-NC 4.0): https://huggingface.co/datasets/ProgramComputer/VGGFace2
- VGG Face licence (CC BY-NC 4.0): https://www.robots.ox.ac.uk/~vgg/data/vgg_face/licence.txt
- DigiFace-1M: https://github.com/microsoft/DigiFace1M — LICENSE (R-UDA): https://github.com/microsoft/DigiFace1M/blob/main/LICENSE — paper: https://arxiv.org/abs/2210.02579
- LFW (офсайт, disclaimer): https://vis-www.cs.umass.edu/lfw/ — update report: https://vis-www.cs.umass.edu/lfw/lfw_update.pdf
- CASIA-WebFace: http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html — paper: https://arxiv.org/abs/1411.7923 — зеркало: https://www.kaggle.com/datasets/debarghamitraroy/casia-webface
- Glint360K / Partial-FC / InsightFace: https://github.com/deepinsight/insightface/tree/master/recognition — paper: https://arxiv.org/abs/2010.05222
- MS-Celeb-1M отзыв (2019, privacy): https://fortune.com/2019/06/07/microsoft-facial-recognition/
- IJB-C / NIST (распространение прекращено 2023-03-14, редирект на FRVT): https://www.nist.gov/programs-projects/face-challenges — IJB-C readme: https://www.nist.gov/system/files/documents/2017/12/26/readme.pdf
- AgeDB: https://ibug.doc.ic.ac.uk/resources/agedb/ — CFP: http://www.cfpw.io/ — CALFW: http://whdeng.cn/CALFW/ — CPLFW: http://whdeng.cn/CPLFW/
- WebFace260M / 42M (agreement запрещает commercial use): https://www.face-benchmark.org/download.html
- Локальные спеки: `docs/about_datasets/wider_face.md`, `docs/about_models/face_embedding.md`

## История изменений

- 2026-06-05 — создан в рамках исследования открытых датасетов-аналогов стека CARS. Адверсариальная проверка по первоисточникам: подтверждены non-commercial у VGGFace2 (CC BY-NC 4.0), DigiFace-1M (R-UDA), CASIA-WebFace (academic-only), WebFace260M (запрет комм.); подтверждён disclaimer LFW; подтверждено происхождение Glint360K / MS1MV2 / V3 от отозванного MS-Celeb-1M. **Исправление:** IJB-C / IJB-B / IJB-A сняты с официального распространения NIST 14.03.2023 → downgrade до `not_recommended (недоступен)`.
