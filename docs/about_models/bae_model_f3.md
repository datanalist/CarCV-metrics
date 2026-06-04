# bae_model_f3 — распознавание цвета автомобиля (Color)

## Общая информация

**bae_model_f3** — кастомная (внутренняя) CNN-модель классификации **цвета кузова** транспортного средства на **15 классов**. В конвейере CARS отвечает за задачу **Color** и работает как **Python-сервис** после детекции ТС: детектор/трекер отдаёт crop автомобиля, сервис нормализует его и выдаёт один из 15 цветов с уверенностью.

Модель упомянута в проектной документации System Design (`docs/system-design/ML_System_Design_Document.md` §6.5) и архитектуре (`docs/architecture.md` §7) под именем файла `bae_model_f3.onnx` и предполагаемым production-путём `models/baseline/bae_model_f3.onnx`. В источнике истины по моделям `models.md` строка задачи Color теперь заполнена так:

```
Color | bae_model_f3 |  |  | /home/mk/CarCV/models/bae_model_f3.onnx
```

— **без URL и без версии, но с локальным путём `/home/mk/CarCV/models/bae_model_f3.onnx`** (абсолютный путь на машине — как и у TrafficCamNet/FaceNet). Это означает: модель кастомная/внутренняя, публично недоступна, в каталог NVIDIA NGC **не входит**, а её веса лежат **вне версионируемого репозитория CarCV-metrics** — в соседнем production-дереве `/home/mk/CarCV`.

> ⚠️ **Ключевое предупреждение.** В рамках валидационной кампании Color **не оценивался — модель отсутствует на NGC, а её веса лежат вне версионируемого репозитория** (`results_collected/FINAL_REPORT.md`, известная проблема №6: «Color recognition model (`bae_model_f3.onnx` from System Design) — not on NGC catalog. Skipped»). Локальный путь к весам теперь зафиксирован в `models.md` (`/home/mk/CarCV/models/bae_model_f3.onnx`), но в зеркало NGC `CarCVModels` и в сам репозиторий CarCV-metrics файл по-прежнему не входит, а условия его распространения не подтверждены — поэтому все метрики из System Design остаются **аспирационными/непроверенными** (см. §«Валидация»).

> **Уточнение по локальным копиям.** Файл `bae_model_f3.onnx` (≈70 MB, экспортирован PyTorch 1.11.0) лежит на машине вне репозитория CarCV-metrics — по пути `/home/mk/CarCV/models/bae_model_f3.onnx`, который теперь и записан в `models.md` (столбец Local path); копии встречаются также в каталогах `/home/mk/Загрузки/CARS/DEEPSTREAM/...`. **В самом репозитории CarCV-metrics и в зеркале NGC `CarCVModels` его нет.** Факты по архитектуре, входу/выходу и препроцессингу ниже сверены непосредственно с этим ONNX-файлом и помечены источником.

| Поле | Значение |
|------|----------|
| Имя модели | `bae_model_f3` |
| Задача | Color — классификация цвета кузова, 15 классов |
| Роль в конвейере CARS | Python-сервис (после детекции/кропа ТС) |
| Семейство / архитектура | **EfficientNet-B3** (MBConv + Squeeze-and-Excitation + Swish/SiLU) — *сверено по ONNX*; legacy-доки ошибочно называют «ResNet-based CNN» |
| Вендор / источник | Кастомная/внутренняя модель проекта (System Design §6.5) |
| Версия | не указана (в `models.md` столбец version пуст) |
| Формат рантайма | ONNX Runtime GPU (заявлено System Design) |
| Локальный путь | `/home/mk/CarCV/models/bae_model_f3.onnx` — **зафиксирован в `models.md`**, но вне репозитория CarCV-metrics (production-дерево `/home/mk/CarCV`). Legacy-доки заявляли путь `models/baseline/bae_model_f3.onnx` |
| URL модели | отсутствует (в `models.md` столбец URL пуст, на NGC модели нет) |
| Лицензия | неизвестна, веса не распространяются |

---

## Архитектура

> Источник: разбор реального файла `/home/mk/CarCV/models/bae_model_f3.onnx` (вне репозитория) через `onnxruntime` + анализ имён весов. Где данные противоречат legacy-докам — приоритет у файла.

- **Семейство: EfficientNet-B3** (timm-стиль). Это устанавливается однозначно по структуре весов и операторов:
  - имена весов `model.blocks.{stage}.{idx}.se.conv_reduce` / `se.conv_expand` → **Squeeze-and-Excitation** блоки внутри MBConv;
  - активации `Mul`+`Sigmoid` → **Swish/SiLU**;
  - финальный `model.classifier.weight` / `model.classifier.bias` → линейный классификатор;
  - **число MBConv-блоков по стадиям: `[2, 4, 4, 6, 6, 8, 2]` = 32 блока** — это в точности профиль глубины **EfficientNet-B3** (depth coefficient 1.4). Суффикс `f3` в имени файла согласуется с «B3».
- **Backbone:** EfficientNet-B3 (conv_stem → 7 стадий MBConv с SE → conv_head → global pooling → classifier).
- **Producer:** `pytorch 1.11.0`, экспорт `torch-jit-export` в ONNX (по метаданным графа).
- **Pruning / quantization:** **признаков прунинга/квантизации в файле нет** — модель экспортирована как обычный FP32-граф EfficientNet-B3. INT8/FP16 — только потенциальная цель TensorRT-конвертации на Jetson (не сделана).

> ❌ **Расхождение с legacy-доками.** `docs/system-design/ML_System_Design_Document.md` §6.5 и `docs/architecture.md` §7 описывают backbone как **«ResNet Backbone» / «ResNet-based CNN»**. По факту файла это **EfficientNet-B3**. При расхождении следует доверять файлу; legacy-формулировка «ResNet» неверна.

---

## Вход / выход и препроцессинг

**Точные тензоры (сверено по ONNX-файлу):**

| Тензор | Имя | Форма | Тип |
|--------|-----|-------|-----|
| Вход | `input` | `[batch_size, 3, 384, 384]` | `float32` |
| Выход | `output` | `[batch_size, 15]` | `float32` |

- **Разрешение входа:** **384×384**, 3 канала, формат **NCHW** (`N×C×H×W`).
- **Цветовой формат:** RGB (порядок каналов RGB; см. препроцессинг ниже).
- **Выход:** вектор из **15 значений — это сырые логиты, а не softmax-вероятности.** Проверка на нулевом входе даёт сумму выходов ≈ 22.7 и значения > 1, то есть нормировка softmax в граф **не зашита** и должна выполняться в постобработке Python-сервиса. Legacy-доки рисуют «FC → Softmax» — softmax нужно применять самому сервису.

**Препроцессинг (семейство «Generic ImageNet classifier», диспетчер из `_bmad-output/project-context.md`):**

| Шаг | Значение |
|-----|----------|
| Чтение | `cv2.imread` → **BGR** uint8 |
| Своп цвета | **BGR → RGB** |
| Масштаб | `/255` (в диапазон `[0,1]`) |
| Нормализация | `(x - mean) / std` |
| `mean` | **`[0.43, 0.40, 0.39]`** (по System Design §6.5) |
| `std` | **`[0.27, 0.26, 0.26]`** (по System Design §6.5) |
| Форма тензора | NCHW |
| Ресайз | до 384×384 |

```python
# Эталонный препроцессинг по таблице семейства Generic ImageNet classifier
# (_bmad-output/project-context.md) + mean/std из System Design §6.5.
# ВНИМАНИЕ: код-эвалуатор для Color в deploy/evaluation/evaluate.py ОТСУТСТВУЕТ
# (модель не входит в EVAL_CONFIGS), приведённый фрагмент — реконструкция.
import cv2, numpy as np

MEAN = np.array([0.43, 0.40, 0.39], np.float32)   # System Design §6.5 (НЕ стандартный ImageNet)
STD  = np.array([0.27, 0.26, 0.26], np.float32)    # System Design §6.5

def preprocess_color(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # BGR->RGB, /255
    img = cv2.resize(img, (384, 384))
    img = (img - MEAN) / STD                          # (x - mean) / std
    return np.transpose(img, (2, 0, 1))[None]         # HWC -> NCHW, add batch
```

> ⚠️ **Расхождение mean/std со стандартным ImageNet.** System Design указывает `mean=[0.43, 0.40, 0.39]`, `std=[0.27, 0.26, 0.26]` — это **отличается** от канонических значений ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Поскольку весов модели в репозитории нет, **проверить, какие именно константы использовались при обучении, невозможно**. Подавать на вход надо те значения, на которых модель обучалась; при появлении весов это нужно подтвердить экспериментально (несовпадение mean/std даёт тихую деградацию точности).

---

## Файлы и форматы

| Файл | Где заявлен / найден | Статус |
|------|----------------------|--------|
| `bae_model_f3.onnx` | заявлен `models/baseline/bae_model_f3.onnx` (`docs/architecture.md` §7; в System Design — только имя файла без пути) | **в репозитории CarCV-metrics отсутствует** |
| `bae_model_f3.onnx` (≈70 MB, FP32, pytorch 1.11.0) | `/home/mk/CarCV/models/bae_model_f3.onnx` — путь, записанный в `models.md`; копии также в `/home/mk/Загрузки/CARS/DEEPSTREAM/...` | присутствует на машине, **но вне версионируемого репозитория CarCV-metrics и вне зеркала NGC `CarCVModels`** |
| `labels` (словарь 15 цветов) | в System Design дан как inline-список (см. §«Классы») | отдельного файла меток в репозитории нет |
| calibration / INT8 | — | отсутствует (граф FP32, без квантизации) |
| `.etlt` / `.pt` / `.ckpt` | — | отсутствуют; распространяется только ONNX-экспорт |

- В EVAL_CONFIGS эвалуатора (`deploy/evaluation/evaluate.py`) записи `color`/`bae_model_f3` **нет**.
- В `deploy/scripts/download_models.sh` модель **не упоминается** (нечего скачивать — на NGC её нет).
- Файл **не зашифрован**, экспорт не требует TAO/NGC-логина; единственная проблема — отсутствие весов в источниках проекта.

---

## Классы / выходной словарь

15 классов цвета (индекс = позиция в выходном векторе `[batch_size, 15]`; порядок — по System Design §6.5 / `docs/architecture.md` §7):

```
beige, black, blue, brown, gold, green, grey,
orange, pink, purple, red, silver, tan, white, yellow
```

| # | Класс |
|---|-------|
| 0 | beige |
| 1 | black |
| 2 | blue |
| 3 | brown |
| 4 | gold |
| 5 | green |
| 6 | grey |
| 7 | orange |
| 8 | pink |
| 9 | purple |
| 10 | red |
| 11 | silver |
| 12 | tan |
| 13 | white |
| 14 | yellow |

> Порядок индексов выше — наиболее вероятная (алфавитная) интерпретация; **точный маппинг индекс→класс непроверяем без оригинального файла меток модели**. При появлении весов сверить с обучающим кодом/labels.

---

## Использование для CarCV

### Применимость

- ✅ Целевая задача §6.5 **Color** — классификация цвета кузова на 15 классов; роль Python-сервиса после детекции/кропа ТС полностью соответствует архитектуре конвейера CARS.
- ✅ Вход 384×384 RGB и выход на 15 классов подтверждены по реальному ONNX-файлу — интерфейс модели определён однозначно.
- ✅ EfficientNet-B3 — современный эффективный backbone (SE + Swish), хорошо конвертируется в TensorRT и разумно ложится на бюджет Jetson Orin Nano как SGIE/сервис-классификатор.

### Ограничения

- ❌ **Весов нет в репозитории CARS и на NGC** — модель невозможно ни прогнать в кампании валидации (`evaluate.py`), ни воспроизводимо развернуть из источников проекта. Это основной блокер.
- ❌ **Метрики не измерены.** Точечные цифры качества (Overall Accuracy 0.84, лучшие/худшие классы, latency 15 мс) взяты из `docs/architecture.md` §7 / `README.md`; System Design §6.5 задаёт лишь **целевые пороги** (>80% / >90% / >70%). И те, и другие — **аспирационные/непроверенные** (см. §«Валидация»), в CarCV-metrics не измерялись.
- ⚠️ **Архитектура в legacy-доках указана неверно** («ResNet»), фактически EfficientNet-B3 — это ставит под сомнение и другие незадокументированные детали (точные mean/std, маппинг меток, разрешение обучения).
- ⚠️ **mean/std не стандартные ImageNet** и непроверяемы без весов — риск тихой деградации точности при неверном препроцессинге.
- ⚠️ **Выход — логиты, не softmax**: сервис обязан сам применять softmax; иначе «уверенность» будет некорректной.
- ⚠️ **Зависимость от детекции/кропа.** Модель ожидает crop ТС; качество цвета наследует ошибки детектора (см. датасеты Color — это полные кадры, а не кропы).
- ⚠️ **Сложные цвета.** System Design сам признаёт challenging-классы (beige, tan, gold, silver) — их различение нестабильно (металлики/блики/освещение). Для `tan` целевой датасет MAD-Cars не имеет отдельного hex (14/15 покрытия).

### Рекомендации

1. **Зафиксировать происхождение и лицензию.** Установить владельца внутренних весов `bae_model_f3.onnx`, условия использования и легитимность распространения; до этого не включать в коммерческую поставку.
2. **Внести веса в источники проекта.** Локальный путь уже зафиксирован в `models.md` (`/home/mk/CarCV/models/bae_model_f3.onnx`), но сам файл остаётся вне версионируемого хранилища. Если модель остаётся в стеке — добавить веса (или ссылку) в версионируемое хранилище CARS, дописать в `models.md` версию и контрольную сумму, а также добавить запись в `download_models.sh`.
3. **Завести Color-эвалуатор.** Добавить `color`/`bae_model_f3` в `EVAL_CONFIGS` (`deploy/evaluation/evaluate.py`): препроцессинг семейства Generic ImageNet classifier (BGR→RGB, /255, `(x-mean)/std`, NCHW), softmax в постобработке, прогон по MAD-Cars (поле `color`).
4. **Исправить документацию.** Привести §6.5 System Design и §7 architecture.md в соответствие с фактом: backbone — **EfficientNet-B3**, выход — логиты, метрики — непроверенные.
5. **Запасной план (рекомендация FINAL_REPORT №4).** Если веса не удастся легализовать — **обучить кастомный 15-классовый классификатор** (например тот же EfficientNet-B3) на `color`-поле MAD-Cars и/или UFPR-VCR, с честной валидацией под automotive POV.

---

## Развёртывание на Jetson

> Всё ниже — **проектные предположения System Design, не измерено** (модели в стенде нет).

- **Целевой рантайм:** Python-сервис на **ONNX Runtime GPU** (CUDA EP) на Jetson Orin Nano 8GB. Для production-конвейера DeepStream возможна конвертация в **TensorRT engine (FP16/INT8)** — но calibration-данных и измерений нет.
- **Роль:** **не PGIE/SGIE DeepStream**, а отдельный **Python-сервис цвета** после детекции/кропа ТС (по аналогии с OCR-сервисом). В DeepStream-конфигах (`configs/dstest2_*.txt`) модель не фигурирует.
- **Вход на Jetson:** 384×384×3 RGB, нормализация как в §«Препроцессинг».
- **Latency:** System Design заявляет **~15 мс** на инференс — **не измерено** в этом репозитории; для EfficientNet-B3 384×384 без квантизации на Orin Nano это оптимистичная оценка, требующая подтверждения.
- **Память:** FP32-граф ≈70 MB; для бюджета 8GB рекомендуется FP16/INT8 engine — пока не построен.

---

## Валидация

**Статус: НЕ оценивалось — заблокировано отсутствием весов.** В дизайне валидационной кампании задача **Color исключена объективно — «нет модели (`UNDEF`)»** (`docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md`). В итоговом отчёте Color помечен «not evaluated / Custom model needed» (`results_collected/FINAL_REPORT.md`, таблица результатов и проблема №6). Файла `metrics.json` для Color в `results_collected/ssh9.qudata.ai/results/` **нет**.

**Целевой датасет (по `datasets.md`, задача Color):** **MAD-Cars** — поле `color` (hex RGB), покрытие **14 из 15** классов CARS (gap — `tan`). Спека датасета: [mad_cars.md](../about_datasets/mad_cars.md). Дополнительно релевантен **Chen 2014 Vehicle Color** (7/15 «лёгких» цветов, полные кадры → требуется детекция+кроп, вход 384×384), где явно описана подача в `bae_model_f3.onnx`: [chen_2014_color.md](../about_datasets/chen_2014_color.md).

| Параметр | Значение |
|----------|----------|
| Датасет (primary) | MAD-Cars, поле `color` — [mad_cars.md](../about_datasets/mad_cars.md) |
| Датасет (secondary) | Chen 2014 Vehicle Color — [chen_2014_color.md](../about_datasets/chen_2014_color.md) |
| Пороги (заявленные §6.5) | Overall Accuracy > 0.80; best (black/white/red/blue) > 0.90; challenging (beige/tan/gold/silver) > 0.70 |
| Измеренный результат | **нет** (Color-эвалуатор не заведён; веса вне репозитория) |
| Вердикт | **UNDEF / заблокировано** — оценить нельзя без весов |

**Аспирационные (LEGACY, непроверенные) заявления System Design / architecture.md:**

| Метрика | Заявлено (legacy) | Проверено в CarCV-metrics? |
|---------|-------------------|----------------------------|
| Overall Accuracy | 0.84 | ❌ не измерено |
| Best classes (>90%) | black, white, red, blue | ❌ не измерено |
| Challenging (<80%) | beige, tan, gold, silver | ❌ не измерено |
| Inference time | ~15 мс | ❌ не измерено |

> Эти цифры **нельзя выдавать за измеренные**. FAIL/UNDEF — валидный окончательный результат и не повод для тюнинга; это статус «оценка невозможна без модели». **Рекомендация:** при появлении весов — прогнать Color-эвалуатор по MAD-Cars (`color`) и Chen 2014 через crop-pipeline TrafficCamNet; иначе обучить кастомный 15-классовый классификатор (FINAL_REPORT, рекомендация №4).

---

## Лицензия

- **Лицензия модели/весов: НЕИЗВЕСТНА.** `bae_model_f3` — кастомная/внутренняя модель; в `models.md` нет ни URL, ни вендора, ни версии. Веса **не распространяются** (отсутствуют в репозитории и на NGC).
- Найденные вне репозитория копии `bae_model_f3.onnx` не сопровождаются файлом лицензии; происхождение и правовой статус не подтверждены.

**Вывод для CARS (КОММЕРЧЕСКИЙ продукт):**
- ❌ **Без подтверждённой лицензии и легального источника весов модель нельзя включать в коммерческую поставку.** Требуется установить правообладателя и условия использования.
- ⚠️ Если веса внутренние (созданы в рамках проекта) — оформить внутреннюю лицензию/owner и зафиксировать в `models.md`; **обязателен legal review** перед поставкой заказчику.
- ⚠️ Датасеты для (пере)обучения/валидации цвета имеют собственные лицензии: MAD-Cars и Chen 2014 — см. legal-разделы их спек ([mad_cars.md](../about_datasets/mad_cars.md), [chen_2014_color.md](../about_datasets/chen_2014_color.md)); ряд альтернатив (VeRi-776) — non-commercial.

---

## Ссылки

- `models.md` (корень репозитория) — источник истины: строка `Color | bae_model_f3` с локальным путём `/home/mk/CarCV/models/bae_model_f3.onnx` (столбцы URL и version не заполнены).
- `datasets.md` (корень) — пара Color → MAD-Cars.
- `docs/system-design/ML_System_Design_Document.md` §6.5 — Color Recognition (вход 384×384, mean/std, 15 классов, целевые пороги >0.80 / >0.90 / >0.70; backbone ошибочно «ResNet»).
- `docs/architecture.md` §7 — bae_model_f3 (Color Recognition), точечные метрики (Overall Accuracy 0.84, latency 15 мс) — legacy, непроверены.
- `_bmad-output/project-context.md` — таблица препроцессинга (семейство Generic ImageNet classifier: BGR→RGB, /255, `(x-mean)/std`, NCHW).
- `results_collected/FINAL_REPORT.md` — проблема №6 (нет модели на NGC, Color skipped) и рекомендация №4 (обучить кастомный 15-классовый CNN на MAD-Cars `color`/UFPR-VCR).
- `docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md` — Color исключён из кампании («нет модели», `UNDEF`).
- Спека датасета MAD-Cars: [docs/about_datasets/mad_cars.md](../about_datasets/mad_cars.md).
- Спека датасета Chen 2014 Vehicle Color: [docs/about_datasets/chen_2014_color.md](../about_datasets/chen_2014_color.md).
- [EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, ICML 2019, arXiv:1905.11946)](https://arxiv.org/abs/1905.11946) — семейство backbone (B3).

---

## История изменений

- **2026-06-04** — Создана спецификация модели в рамках документирования стека `models.md`.
- **2026-06-04** — В `models.md` зафиксирован локальный путь к весам `/home/mk/CarCV/models/bae_model_f3.onnx`; спека приведена в соответствие (Общая информация, таблицы «Поле/Значение» и «Файлы и форматы», Валидация, Рекомендации, Ссылки).
