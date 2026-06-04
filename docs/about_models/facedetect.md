# FaceDetect (FaceNet) — детекция лиц

## Общая информация

**FaceDetect** — детектор лиц на базе **NVIDIA TAO FaceNet** (каталог NGC). В конвейере CARS это **детектор лиц водителя и пассажиров**: он находит bounding box лица в кропе/кадре, чтобы система могла сохранить кроп и зафиксировать факт присутствия людей. Питает требование **FR-06** (сохранение кропов лиц), поля БД `face_count` / `face_coords` и артефакты `face_images/{track_id}.bmp`.

В `models.md` модель указана **дважды** — строки `Face Detection | FaceDetect` (стр. 6) и `Face Detector | FaceDetect` (стр. 11). Это **одна и та же модель** (один и тот же NGC-URL и версия `pruned_quantized_v2.0.1`); данная спека объединяет обе записи.

> **Важно — не путать с эмбеддингом лиц.** NVIDIA «FaceNet» в этом стеке — это **детектор** лиц (архитектура DetectNet_v2), а **не** модель эмбеддингов лиц «FaceNet» от Google (FaceNet: A Unified Embedding, 2015). Эмбеддинг лиц — отдельная **будущая** задача `Face embedding` в `models.md` (стр. 12: `None / Trainable` — модели в стеке пока нет).

| Поле | Значение |
|------|----------|
| Имя в проекте | FaceDetect |
| Вендор / источник | NVIDIA TAO, каталог NGC (модель `facenet`) |
| Версия | `pruned_quantized_v2.0.1` |
| Задача | Face Detection (детекция лиц) |
| Роль в конвейере CARS | SGIE4 (вторичный детектор в DeepStream, см. legacy-доки) |
| Архитектура | DetectNet_v2, backbone ResNet-18 |
| Классов | 1 (`face`) |
| Локальное расположение | `/home/mk/Загрузки/CarCVModels/facenet_pruned_quantized_v2.0.1` |
| URL модели | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet?version=pruned_quantized_v2.0.1 |

> **Расхождение с legacy-документацией.** `docs/architecture.md` (стр. 124) описывает FaceDetect как «NVIDIA FaceDetect / RetinaFace», формат «TensorRT FP16», файл `models/baseline/facedetect.engine`. Упоминание **RetinaFace** не подтверждается источником истины — по карточке NGC и локальной копии это **DetectNet_v2 (FaceNet)**, причём вариант **pruned_quantized** с **INT8**-калибровкой (а не FP16). Legacy-метрики/описание считать **непроверенными**.

---

## Архитектура

- **Семейство:** DetectNet_v2 (NVIDIA TAO) — тот же класс детекторов, что и TrafficCamNet/LPDNet (grid-based detection, выходы coverage + bbox).
- **Backbone:** ResNet-18 (по карточке NGC FaceNet; не сверено с локальной копией — `.etlt` зашифрован, конфига нет).
- **Классов:** один — `face`.
- **Pruning / quantization:** версия `pruned_quantized_v2.0.1` — модель **прунена** (pruned) и подготовлена под **INT8**-квантование; в локальной копии присутствует INT8 calibration cache (см. ниже).

> Детальные параметры сети (число якорей/grid, точная конфигурация ResNet-18, размер выходного грида) **в локальной копии недоступны** — `.etlt` зашифрован, локального `nvinfer_config.txt` для FaceNet нет. Сведения об архитектуре — **по карточке NGC** и по аналогии с семейством DetectNet_v2.

---

## Вход / выход и препроцессинг

FaceDetect относится к семейству **DetectNet_v2**, и для него применяется тот же препроцессинг, что и для TrafficCamNet (диспетчер препроцессинга `_bmad-output/project-context.md`, стр. 62). Оговорка: в самой таблице диспетчера (стр. 62) FaceNet **не перечислен поимённо** — там указаны `TrafficCamNet, LPDNet`; отнесение FaceDetect к ветке DetectNet_v2 сделано **по архитектуре модели** (ResNet-18 DetectNet_v2, как у TrafficCamNet).

| Параметр | Значение | Источник |
|----------|----------|----------|
| Разрешение входа (W×H) | **736 × 416** | по дизайну кампании / карточке NGC (см. оговорку) |
| Каналы | 3 | — |
| Цветовой формат | **BGR → RGB** (своп каналов) | семейство DetectNet_v2 (диспетчер, стр. 62) |
| Масштаб | **/255** (net-scale-factor) | семейство DetectNet_v2 (диспетчер, стр. 62) |
| Offsets / mean-subtract | нет (только деление на 255) | семейство DetectNet_v2 (диспетчер, стр. 62) |
| Формат тензора | **NCHW** | — |

> **Оговорка по входу.** Локального `nvinfer_config.txt` для FaceNet **нет**, поэтому источники разведены. Точный размер входа **736 × 416** взят **из дизайна валидационной кампании** (`docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md`, стр. 102–103: «1 класс «face», вход 736×416») **и карточки NGC** — про offsets в спеке кампании ничего не сказано. Отсутствие mean-offsets и деление на **/255** — это свойство **семейства DetectNet_v2** из диспетчера препроцессинга (`_bmad-output/project-context.md`, стр. 62: `DetectNet_v2 (TrafficCamNet, LPDNet)` → `/255`, без offsets), а не запись из спеки кампании. Ни одно из этих значений **не** прочитано из локального конфига. При расхождении с реальным engine — приоритет у фактического `nvinfer`-конфига продакшна.

Препроцессинг идентичен ветке TrafficCamNet в `deploy/evaluation/evaluate.py` (строки 212–213):

```python
inp = cv2.resize(img, (W, H)).astype(np.float32)
inp = inp[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0  # BGR→RGB NCHW
```

**Выход и декодирование.** DetectNet_v2 отдаёт два тензора:
- `cov` — `[num_classes, gh, gw]` (coverage / уверенность, sigmoid);
- `bbox` — `[num_classes*4, gh, gw]` (нормированные смещения `x1,y1,x2,y2` относительно центра ячейки).

Декодирование переиспользует функцию `detectnet_v2_decode` (та же, что для TrafficCamNet/LPDNet; `deploy/evaluation/evaluate.py`, строки 112–146): для FaceDetect — **1 класс** (`target_cls=0`), параметры **`stride=16`, `bbox_norm=35.0`**, затем NMS (IoU 0.5):

```python
cx = (j + 0.5) * stride          # центр ячейки в координатах входа
cy = (i + 0.5) * stride
dx1, dy1, dx2, dy2 = bbox_cls[:, i, j] * bbox_norm
x1 = cx - dx1; y1 = cy - dy1; x2 = cx + dx2; y2 = cy + dy2
```

> `stride=16` и `bbox_norm=35.0` — дефолты `detectnet_v2_decode`, подтверждённые для TrafficCamNet/LPDNet. Для FaceNet эвалуатор ещё не реализован, поэтому эти значения — **план по дизайну кампании** (переиспользование декодера), не результат измеренного прогона.

---

## Файлы и форматы

Содержимое `/home/mk/Загрузки/CarCVModels/facenet_pruned_quantized_v2.0.1`:

| Файл | Размер | Что это |
|------|--------|---------|
| `model.etlt` | 5 775 090 байт (≈ 5.78 MB) | **зашифрованный** экспорт TAO (`.etlt`) |
| `int8_calibration.txt` | 4 840 байт | INT8 calibration cache (TensorRT) |

- **`model.etlt`** — зашифрованный TAO-экспорт. Для декодирования/конвертации в ONNX или TensorRT engine требуется **ключ модели** (`tlt-model-key`, обычно `tlt_encode` / `nvidia_tlt`). **Локально ONNX отсутствует** — для прогона на x86-валидации (onnxruntime) модель нужно сперва экспортировать из `.etlt`.
- **`int8_calibration.txt`** — кэш калибровки INT8 (заголовок `TRT-8001-EntropyCalibration2`, далее per-tensor масштабы вида `input_1: 3c010a14`, `conv1/convolution: …`). Используется при сборке INT8-engine под TensorRT.
- В локальной копии **нет**: `labels.txt`, `nvinfer_config.txt`, готового `.onnx`/`.engine`.

```
/home/mk/Загрузки/CarCVModels/facenet_pruned_quantized_v2.0.1/
├── model.etlt              # 5 775 090 B — зашифрованный TAO-экспорт (нужен tlt-model-key)
└── int8_calibration.txt    # 4 840 B — INT8 calibration cache (TRT-8001-EntropyCalibration2)
```

---

## Классы / выходной словарь

Один класс:

| Индекс | Класс |
|--------|-------|
| 0 | `face` |

Отдельного локального файла `labels.txt` для FaceNet нет; единственный класс — `face` (по карточке NGC и дизайну кампании; не сверено с локальной копией — `.etlt` зашифрован, конфига нет).

---

## Использование для CarCV

### Применимость

- ✅ Детекция лиц для **FR-06**: bounding box лица → сохранение кропа `face_images/{track_id}.bmp`, заполнение `face_count` и `face_coords`.
- ✅ Лёгкая, прунено-квантованная модель (≈ 5.78 MB `.etlt`, INT8-калибровка в комплекте) — подходит под бюджет Jetson Orin Nano как SGIE.
- ✅ Семейство DetectNet_v2 переиспользует уже имеющийся декодер `detectnet_v2_decode` — реализация эвалуатора опирается на проверенный код.

### Ограничения

- ❌ **ONNX локально отсутствует:** `.etlt` зашифрован, нужен `tlt-model-key` для экспорта — без него x86-валидация невозможна.
- ❌ **Эвалуатор не реализован, метрики не измерены.** FaceDetect **нет** в `EVAL_CONFIGS` (`deploy/evaluation/evaluate.py`), не добавлен в `download_models.sh`, отсутствует каталог `results_collected/ssh9.qudata.ai/results/facedetect/` и не упомянут в `FINAL_REPORT.md`.
- ⚠️ **Доменный разрыв.** Обучающий/валидационный домен (см. WIDER FACE) — веб-фото толп, парадов, спорта; CARS — automotive POV: лицо через лобовое стекло, дистанция 5–50 м, режимы day/night/IR. Распределения расходятся.
- ⚠️ **Нет IR/ночи в валидации.** WIDER FACE — только RGB/день; ИК-режим и ночь датасетом не покрыты.
- ⚠️ **Вход и offsets — по дизайну/карточке, не из локального конфига** (нет `nvinfer_config.txt`). Перед продакшн-выводами сверить с реальным `nvinfer`.
- ⚠️ **Legacy-описание противоречиво** (RetinaFace/FP16 в `docs/architecture.md`) — не использовать как источник истины.

### Рекомендации

1. **Экспорт ONNX.** Получить `tlt-model-key` и экспортировать `model.etlt` → ONNX (вход `736×416`, 1 класс) для прогона через onnxruntime-gpu на x86.
2. **Реализовать эвалуатор `eval_facedetect` с нуля** (задача кампании `eval-facedetect`, самая рискованная, в конце): препроцессинг DetectNet_v2 + `detectnet_v2_decode` (`target_cls=0`, `stride=16`, `bbox_norm=35.0`) + парсер аннотаций WIDER FACE + detection-метрики (AP по Easy/Medium/Hard).
3. **Базовый прогон** по `WIDER_val` (3 226 изображений), AP отдельно по Easy/Medium/Hard официальным devkit.
4. **Automotive-срез.** Дополнительно считать метрики на категориях, близких к транспортной сцене: `14--Traffic`, `5--Car_Accident`, `59--people--driving--car` (опц. `42--Car_Racing`).
5. **IR/night вне WIDER FACE.** Для ночных/ИК-сценариев собрать отдельную выборку — WIDER FACE их не покрывает.
6. **Сверить препроцессинг с продакшном** (фактический `nvinfer_config.txt` engine), прежде чем переносить вердикт на Jetson.

---

## Развёртывание на Jetson

- **Продакшн-формат:** TensorRT engine (legacy-файл `models/baseline/facedetect.engine` по `docs/architecture.md`).
- **Квантование:** в комплекте INT8 calibration cache (`int8_calibration.txt`) → возможна сборка **INT8**-engine; legacy-доки упоминают FP16 (непроверено, противоречит наличию INT8-кэша).
- **Роль:** **SGIE4** — вторичный детектор в DeepStream-конвейере (PGIE TrafficCamNet → SGIE VehicleMakeNet / VehicleTypeNet / LP Detection / **FaceDetect**), по `README.md`/`docs/architecture.md`.
- **Целевая latency:** **3–5 ms** — **legacy-цель** из `docs/architecture.md` (стр. 310) и `README.md` (стр. 430), **НЕ измерено** в этой кампании.

> Все числа latency/формата для Jetson здесь — **legacy/аспирационные**, измеренного прогона на Jetson в репозитории нет.

---

## Валидация

- **Датасет:** **WIDER FACE** (val), **3 226 изображений**, 61 категория. Спека датасета: [docs/about_datasets/wider_face.md](../about_datasets/wider_face.md). Привязка пары модель×датасет — `datasets.md` (строка Face Detection → WIDER FACE; `WIDER_val`).
- **Метрика:** **Average Precision (AP)** на подмножествах **Easy / Medium / Hard** (официальный devkit WIDER FACE, ground truth `.mat`).
- **Пороги pass/fail:** **TBD.** В проекте для FaceDetect задано только **семейство** метрик (Detection: Precision/Recall/F1/AP/mAP@0.5 — дизайн кампании, стр. 168); конкретного числового порога AP для лиц в источниках **не зафиксировано**.
- **Измеренный результат:** **НЕ ИЗМЕРЕНО.** Эвалуатор для FaceDetect **надо реализовать с нуля** (задача `eval-facedetect`, помечена как самая рискованная и идёт в конце кампании). В `FINAL_REPORT.md` FaceDetect не оценивался; каталога результатов нет.
- **Вердикт:** **н/д (нет прогона).** Когда прогон будет выполнен: **FAIL — валидный окончательный результат**, а не повод для тюнинга. Ожидаемые риски занижения: масса микро-лиц тянет вниз Hard-AP; доменный разрыв (веб-фото vs automotive POV «через стекло», 5–50 м); отсутствие IR/ночи в WIDER FACE. Вердикт под CARS опирать на **automotive-срез** (`14--Traffic`, `5--Car_Accident`, `59--people--driving--car`) + отдельную **IR/night**-выборку вне WIDER FACE.

---

## Лицензия

- **Модель:** NVIDIA TAO / NGC (FaceNet). Условия использования определяются лицензией модели на карточке NGC (NVIDIA AI / TAO model EULA). Точный текст лицензии в локальной копии **отсутствует** — смотреть карточку NGC.
- **Валидационный датасет (WIDER FACE):** **CC BY-NC-ND 4.0** — **NonCommercial** (см. [docs/about_datasets/wider_face.md](../about_datasets/wider_face.md), раздел «Лицензия»).

**Вывод для CARS (КОММЕРЧЕСКИЙ продукт):**
- ⚠️ Перед использованием FaceNet в составе коммерческого продукта **подтвердить условия NVIDIA** (TAO/NGC EULA: допустимость коммерческого деплоя, дистрибуции engine, атрибуции) — **требуется legal review**.
- ⚠️ Метрики, полученные на WIDER FACE, ограничены некоммерческой лицензией датасета: внутренний research-бенчмаркинг допустим, любое внешнее использование (публикация метрик, поставка артефактов) — только после legal review (см. спеку WIDER FACE).
- Тон осторожный: до подтверждения лицензионных условий FaceNet и согласования использования WIDER FACE не выдавать результаты вовне.

---

## Ссылки

- [Карточка модели NGC — FaceNet (pruned_quantized_v2.0.1)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet?version=pruned_quantized_v2.0.1)
- [Спека датасета валидации — WIDER FACE](../about_datasets/wider_face.md)
- `models.md` (строки 6 и 11 — FaceDetect; строка 12 — будущий `Face embedding`)
- `datasets.md` (Face Detection → WIDER FACE)
- `deploy/evaluation/evaluate.py` — `detectnet_v2_decode` (строки 112–146), препроцессинг DetectNet_v2 (строки 212–213)
- `docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md` (раздел FaceDetect: 1 класс `face`, вход 736×416, эвалуатор с нуля)
- Локальные файлы: `/home/mk/Загрузки/CarCVModels/facenet_pruned_quantized_v2.0.1/{model.etlt, int8_calibration.txt}`

---

## История изменений

- **2026-06-04** — Создана спецификация модели в рамках документирования стека `models.md`.
