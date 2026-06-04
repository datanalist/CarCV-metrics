# VehicleMakeNet — классификация марки автомобиля (Make)

## Общая информация

**VehicleMakeNet** — предобученный классификатор марки (бренда) транспортного средства из набора **NVIDIA TAO** (NGC). Модель принимает кроп ТС и относит его к одной из **20 марок** (US/EU-рынок). В конвейере CARS это **SGIE1** (Secondary GIE): работает поверх кропов ТС, выданных детектором **TrafficCamNet** (PGIE), и заполняет атрибут «марка».

| Поле | Значение |
|------|----------|
| Имя | VehicleMakeNet |
| Задача в CARS | Make — классификация марки (20 классов) |
| Роль в конвейере | SGIE1 (на кропе ТС от TrafficCamNet) |
| Вендор/источник | NVIDIA TAO, NGC |
| Семейство | TAO image classification (TF1) |
| Backbone | ResNet-18 (pruned) |
| Версия (URL в models.md) | `pruned_onnx_v1.1.0` |
| Версия (колонка `version` в models.md) | `pruned_v1.0.2` |
| Версия, фактически скачиваемая | `pruned_onnx_v1.1.0` (`download_models.sh`) |
| Вход | 224×224×3 |
| Выход | логиты 20 классов → softmax → top-k |

> **Расхождение версий (зафиксировано, не ошибка документа).** В `models.md` для VehicleMakeNet колонка `version` указывает `pruned_v1.0.2`, тогда как URL в той же строке ссылается на `?version=pruned_onnx_v1.1.0`. Скрипт загрузки `deploy/scripts/download_models.sh` тянет именно `pruned_onnx_v1.1.0` (переменная `V="pruned_onnx_v1.1.0"`). Источником истины для фактически валидируемых весов считается `download_models.sh` → **`pruned_onnx_v1.1.0`**. Колонку `pruned_v1.0.2` в `models.md` следует рассматривать как устаревшую/нестыкованную запись.

**Локальное расположение (после загрузки):** `deploy/models/vehiclemakenet/resnet18_pruned.onnx` + `labels.txt` (целевой путь `download_models.sh`). На диске разработчика этих файлов в репозитории нет — каталог `deploy/models/` в репозитории отсутствует (`download_models.sh` ещё не запускался) — присутствует только дистрибутивный архив `/home/mk/Загрузки/CarCVModels/vehiclemakenet_pruned_onnx_v1.1.0.zip` (≈6.85 MB).

> **Расхождение путей в конфиге.** `configs/experiment/vehiclemakenet_eval.yaml` ссылается на `models/vehiclemakenet_pruned_onnx_v1.1.0/resnet18_pruned.onnx`, а боевой эвалуатор (`EVAL_CONFIGS["vehiclemakenet"]`) — на `models/vehiclemakenet/resnet18_pruned.onnx`. Это разные пути одной и той же модели; при прогоне сверять с конкретным запускаемым кодом.

**Ссылки:**
- Карточка NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehiclemakenet?version=pruned_onnx_v1.1.0

---

## Архитектура

- **Семейство:** TAO image classification, TensorFlow 1.x (по комментарию в `evaluate.py`), backbone ResNet-18 pruned. Это классификатор; не путать с детекционной мета-архитектурой DetectNet_v2 (TrafficCamNet/FaceDetect) — у них разное препроцессинг-семейство.
- **Backbone:** ResNet-18, **pruned** (облегчённый), под классификацию на 20 марок.
- **Голова:** полносвязный классификатор на 20 выходов (логиты).
- **Pruning:** да (`pruned` в имени версии и файла `resnet18_pruned.onnx`).
- **Quantization:** в архиве присутствует калибровочный файл INT8 (`resnet18_pruned_int8.txt`) — задел под INT8-движок TensorRT; сама ONNX-модель — FP32-граф, экспорт в FP16/INT8 выполняется при сборке engine на Jetson.

---

## Вход / выход и препроцессинг

**Семейство препроцессинга — TAO classifier** (диспетчер из `_bmad-output/project-context.md`): отличается от ImageNet- и DetectNet_v2-стеков. Перепутать их — главная причина «тихого» падения точности.

| Параметр | Значение | Источник |
|----------|----------|----------|
| Разрешение входа | 224×224 | `EVAL_CONFIGS` / `vehiclemakenet_eval.yaml` (`input_size: [224, 224]`) |
| Каналы | 3 |  |
| Цветовой формат | **BGR без свопа в RGB** | `preprocess_tao_bgr` (cv2 читает как BGR, своп не делается) |
| Масштабирование | **нет** (`/255` НЕ применяется) | `preprocess_tao_bgr` |
| Вычитание offsets | `(104, 117, 124)` по каналам **(B; G; R)** | вызов `preprocess_tao_bgr(..., offsets=(104.0, 117.0, 124.0))` |
| Формат тензора | **NCHW** | `transpose(2, 0, 1)[None]` |
| Выход | логиты 20 классов → `softmax` → top-3 | `eval_vehiclemakenet` |

> ⚠️ **Критично: BGR без свопа и БЕЗ деления на 255.** В отличие от ImageNet-классификатора `bae_model_f3` (там BGR→RGB, `/255`, затем `(x-mean)/std`) и DetectNet_v2-детекторов (BGR→RGB, `/255`), TAO-классификатор работает по **BGR** и вычитает offsets **без** масштабирования. Подача RGB вместо BGR или применение `/255` даёт корректную форму тензора, но визуально искажённый вход → softmax «уверенно» ошибается. Это типичный источник скрытой деградации точности.

Фрагмент препроцессинга (`deploy/evaluation/evaluate.py`):

```python
def preprocess_tao_bgr(img_bgr: np.ndarray, size: int = 224,
                      offsets=(103.939, 116.779, 123.68)) -> np.ndarray:
    """TAO TF1 classification preprocessing: BGR, no scaling, mean subtract.
    VehicleMakeNet/TypeNet use offsets=(104, 117, 124) (B;G;R).
    """
    img = cv2.resize(img_bgr, (size, size)).astype(np.float32)
    img[..., 0] -= offsets[0]   # B
    img[..., 1] -= offsets[1]   # G
    img[..., 2] -= offsets[2]   # R
    return img.transpose(2, 0, 1)[None]  # NCHW
```

Декодирование выхода (`eval_vehiclemakenet`):

```python
inp = preprocess_tao_bgr(img, size=224, offsets=(104.0, 117.0, 124.0))
out = sess.run(None, {input_name: inp})[0][0]   # логиты, shape (20,)
probs = softmax(out)
top_k = [labels[i] for i in probs.argsort()[::-1][:3]]   # top-3 по убыванию
```

> Значения offsets по умолчанию в сигнатуре `(103.939, 116.779, 123.68)` — ImageNet BGR-средние; для VehicleMakeNet используется явный вызов с `(104.0, 117.0, 124.0)`. Ориентироваться нужно на фактический вызов в `eval_vehiclemakenet`, а не на дефолт сигнатуры.

---

## Файлы и форматы

Дистрибутивный архив `vehiclemakenet_pruned_onnx_v1.1.0.zip` (≈6.85 MB, путь `/home/mk/Загрузки/CarCVModels/`):

| Файл | Размер | Назначение |
|------|--------|------------|
| `resnet18_pruned.onnx` | ≈7.40 MB (7 401 391 B распакованного) | веса модели (FP32-граф ONNX) |
| `labels.txt` | 133 B | 20 меток марок, разделитель `;` (нижний регистр) |
| `resnet18_pruned_int8.txt` | 6 918 B | калибровочная таблица INT8 (для сборки INT8-engine) |

> **Локальное состояние.** Каталог `deploy/models/` в репозитории **отсутствует** целиком (`download_models.sh` ещё не запускался в этой копии) — файлов модели нет. Доступен только zip-архив. ONXX-граф **не зашифрован** (это уже экспортированный `.onnx`, не `.etlt`), отдельный экспорт/расшифровка не требуется — для запуска достаточно распаковать архив в целевой путь.

Формат `labels.txt` — одна строка с разделителем `;` (поддерживается парсером `load_labels`, ветка «`;` и ≤2 строк»):

```
acura;audi;bmw;chevrolet;chrysler;dodge;ford;gmc;honda;hyundai;infiniti;jeep;kia;lexus;mazda;mercedes;nissan;subaru;toyota;volkswagen
```

---

## Классы / выходной словарь

20 классов марок (порядок = индекс выхода, `NGC_MAKES` в `evaluate.py`; `labels.txt` содержит те же марки в нижнем регистре):

| # | Класс | # | Класс |
|---|-------|---|-------|
| 0 | Acura | 10 | Infiniti |
| 1 | Audi | 11 | Jeep |
| 2 | BMW | 12 | Kia |
| 3 | Chevrolet | 13 | Lexus |
| 4 | Chrysler | 14 | Mazda |
| 5 | Dodge | 15 | Mercedes |
| 6 | Ford | 16 | Nissan |
| 7 | GMC | 17 | Subaru |
| 8 | Honda | 18 | Toyota |
| 9 | Hyundai | 19 | Volkswagen |

Все 20 марок — **US/EU-рынок**. Марок RU-рынка (VAZ/Lada, Москвич, Solaris и т. п.) в словаре **нет** — это определяющее ограничение (см. «Валидация»).

**Нормализация меток ground truth (`normalize_brand`).** При сопоставлении GT с 20 NGC-марками используется подстрочное сопоставление (substring match) в нижнем регистре:
- строки короче 3 символов **не матчатся** и возвращаются как есть — это исправление бага, при котором пустая строка `""` как подстрока попадала в `"acura"` и схлопывала всю оценку в один класс (см. комментарий в коде, отсылка к `deferred-work.md`, Goal 4);
- GT-марки **вне** 20 NGC-классов (out-of-distribution) **пропускаются** и **отдельно считаются** счётчиком `skipped_oo_dist` (исключаются из метрик).

---

## Использование для CarCV

### Применимость (✅)

- ✅ **Make для US/EU-марок.** Прямое назначение SGIE1: 20 распространённых US/EU-брендов на кропе ТС от TrafficCamNet.
- ✅ **Лёгкая модель** (~7.4 MB, ResNet-18 pruned) — подходит по бюджету памяти/латентности Jetson Orin Nano как вторичный классификатор.
- ✅ **Готовый ONNX без расшифровки** — интегрируется в onnxruntime-валидацию и в экспорт TensorRT без шага дешифровки `.etlt`.
- ✅ Совместима с TAO-препроцессингом, уже реализованным в `preprocess_tao_bgr`.

### Ограничения (❌/⚠️)

- ❌ **Только 20 US/EU-марок.** RU-рынок (VAZ/Lada, Москвич, китайские бренды и пр.) словарём не покрыт; такие ТС либо ошибочно классифицируются в одну из 20, либо в боевом конвейере дают бессмысленный ответ.
- ❌ **Cross-domain failure на RU-данных** подтверждён измерениями (см. «Валидация»): на суррогате mad-cars Top-1 = 0.0829 при пороге 0.70.
- ⚠️ **BGR-препроцессинг без `/255`** — высокий риск тихой деградации при ошибке интеграции (своп каналов / лишнее масштабирование).
- ⚠️ **Чувствительность к качеству кропа.** Модель ждёт кроп ТС (224×224); ошибки/неточности bbox PGIE напрямую бьют по точности марки.
- ⚠️ **Визуальные различия рыночных вариантов.** Даже на пересекающихся марках (Toyota, BMW и т. п.) точность едва выше случайной на RU-вариантах — рестайлинги/комплектации US-vs-RU различаются.

### Рекомендации

1. **Соблюдать TAO-препроцессинг строго:** BGR без свопа, offsets `(104, 117, 124)` по каналам (B;G;R), без `/255`, NCHW. Не переиспользовать препроцессинг от `bae_model_f3` или TrafficCamNet.
2. **Валидировать на целевом домене (VMMRdb), а не на суррогате.** Кампания переводит оценку с mad-cars на VMMRdb с загрузчиком каталогов-классов `make_model_year`; эвалуатор `eval_vehiclemakenet` уже есть.
3. **Считать только пересекающиеся марки.** OOD-марки исключать из метрик (счётчик `skipped_oo_dist`); при анализе использовать макро-усреднение по марке из-за long-tail VMMRdb.
4. **Для RU-продакшна не использовать as-is.** При требовании RU-марок — дообучать (fine-tune) на RU-классах либо переходить на embedding-классификатор (ср. эксперимент на 105 классов в `vehiclemakenet_eval.yaml` — это отдельная, дообученная модель `vmn_vmmrdb_ymad_105c`, не baseline).
5. **Не путать baseline (20 классов) и finetuned (105 классов).** Это два разных эксперимента с разными порогами и весами.

---

## Развёртывание на Jetson

| Параметр | Значение |
|----------|----------|
| Платформа | NVIDIA Jetson Orin Nano 8GB |
| Рантайм | DeepStream SDK + TensorRT |
| Engine | TensorRT, **FP16** (INT8 возможен — есть калибровочная таблица `resnet18_pruned_int8.txt`) |
| Роль | **SGIE1** (вторичный классификатор поверх кропов ТС от PGIE TrafficCamNet) |
| Целевая latency | 4–5 ms (**legacy-ориентир, НЕ измерено**) |

> Числа latency взяты из legacy-описания стека и **не подтверждены измерениями** в этой кампании. Валидация (этот репозиторий) идёт на x86 GPU через onnxruntime-gpu; точность ONNX/PyTorch считается **верхней границей** для TensorRT-engine на Jetson.

---

## Валидация

**Целевой датасет:** VMMRdb (Vehicle Make and Model Recognition Dataset) — см. спеку [docs/about_datasets/vmmrdb.md](../about_datasets/vmmrdb.md) и строку в `datasets.md` (пара Make ↔ VMMRdb).

**Пороги pass/fail** (боевой эвалуатор `eval_vehiclemakenet`, `thresholds`; совпадают с §6.3 спеки VMMRdb):

| Метрика | Порог |
|---------|-------|
| Top-1 accuracy | **≥ 0.70** |
| Top-3 accuracy | **≥ 0.85** |

> **Сноска о порогах.** `configs/experiment/vehiclemakenet_eval.yaml` содержит `target_top1: 0.90`, `target_top3: 0.97`. Это пороги для **другого** эксперимента — finetuned-модели на **105 классов** (`vmn_vmmrdb_ymad_105c`), а не для 20-классового baseline. Для baseline-кампании действуют **0.70 / 0.85** (значения в коде `eval_vehiclemakenet`).

**Статус кампании:** требуется **перегон на VMMRdb** (добавляется загрузчик каталогов-классов `make_model_year`). Эвалуатор `eval_vehiclemakenet` готов; на VMMRdb прогон ещё не выполнен (на дату документа результат VMMRdb — **TBD**).

**Предыдущий измеренный прогон (суррогат mad-cars, 700 in-distribution сэмплов)** — `results_collected/ssh9.qudata.ai/results/vehiclemakenet/metrics.json`:

| Метрика | Измерено | Порог | Вердикт |
|---------|----------|-------|---------|
| Top-1 accuracy | **0.0829** | ≥ 0.70 | ❌ FAIL |
| Top-3 accuracy | **0.2114** | ≥ 0.85 | ❌ FAIL |
| num_samples | 700 | — | — |

Лучшая марка в прогоне — Toyota (0.2857), худшие (Acura, Audi, Chrysler, Subaru, Infiniti) — 0.0.

**Причина FAIL (cross-domain):** 20 NGC-марок — US/EU; mad-cars — RU-рынок. ≈86% сэмплов (по `FINAL_REPORT.md` — 4260/4960) принадлежат маркам вне NGC (VAZ/Lada, Москвич, Solaris, Trumpchi и т. п.) и исключены из оценки; даже на пересекающихся марках точность лишь чуть выше случайной из-за визуальных различий RU-вариантов.

> **FAIL — валидный окончательный результат.** Это фиксируемый вывод о применимости модели к RU-домену, а не повод для тюнинга в рамках валидационной кампании. Перегон на VMMRdb (US-домен) проверит верхнюю границу точности модели на её «родном» рынке.

---

## Лицензия

- **Модель/веса:** NVIDIA TAO / NGC. Условия использования регулируются лицензией модели на карточке NGC (NVIDIA AI / TAO model EULA). Точный текст лицензии в локальной копии (архиве) **отсутствует** — нужно сверять с карточкой NGC.
- **Вывод для CARS (КОММЕРЧЕСКИЙ продукт):** перед коммерческим применением **подтвердить условия лицензирования NVIDIA** (право на коммерческое использование/распространение предобученных весов, требования атрибуции) — обязателен **legal review**. До прохождения review рассматривать как research/benchmark-актив.

---

## Ссылки

- [Карточка NGC: VehicleMakeNet (pruned_onnx_v1.1.0)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehiclemakenet?version=pruned_onnx_v1.1.0)
- Источник истины по моделям: `models.md` (строка Make / VehicleMakeNet)
- Пара модель×датасет: `datasets.md` (Make ↔ VMMRdb)
- Спека датасета валидации: [docs/about_datasets/vmmrdb.md](../about_datasets/vmmrdb.md)
- Эвалуатор и препроцессинг: `deploy/evaluation/evaluate.py` (`eval_vehiclemakenet`, `NGC_MAKES`, `normalize_brand`, `preprocess_tao_bgr`)
- Скрипт загрузки: `deploy/scripts/download_models.sh`
- Конфиг эксперимента: `configs/experiment/vehiclemakenet_eval.yaml`
- Измеренные метрики (mad-cars): `results_collected/ssh9.qudata.ai/results/vehiclemakenet/metrics.json` (+ `per_class_metrics.csv`)
- Сводный отчёт: `results_collected/FINAL_REPORT.md` (§2 VehicleMakeNet)

---

## История изменений

- **2026-06-04** — Создана спецификация модели в рамках документирования стека models.md.
</content>
</invoke>
