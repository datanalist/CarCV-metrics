# Эксперимент: FaceDetect (FaceNet, DetectNet_v2) × WIDER FACE

Дата: 2026-06-04 · Ветка: exp/eval-facedetect · GPU: RTX 3090 (CUDAExecutionProvider)

## 1. Выписка из docs (шаг 1 протокола)
- Модель: **FaceDetect** — детектор лиц **NVIDIA TAO FaceNet** (`pruned_quantized_v2.0.1`),
  архитектура **DetectNet_v2** (backbone ResNet-18), **1 класс** (`face`). НЕ путать с эмбеддингом
  лиц (Google FaceNet) — это детектор. См. `docs/about_models/facedetect.md`.
- Препроцессинг (семейство DetectNet_v2, как TrafficCamNet/LPDNet): вход **736×416 (W×H)**,
  3 канала, **BGR→RGB**, масштаб **`/255`**, **без offsets/mean-subtract**, тензор **NCHW**.
  Реализовано в `eval_facedetect`:
  `inp = cv2.resize(img, (W, H)).astype(np.float32); inp = inp[:, :, ::-1].transpose(2,0,1)[None] / 255.0`.
  > Каваэт: вход 736×416 и отсутствие offsets взяты из дизайна кампании + карточки NGC +
  > свойств семейства DetectNet_v2 (диспетчер препроцессинга), а **не** из локального
  > `nvinfer_config.txt` (его для FaceNet нет). Перед продакшн-выводами сверить с реальным engine.
- Декодирование: переиспользует `detectnet_v2_decode` (тот же, что для TrafficCamNet/LPDNet) с
  **`target_cls=0`, `conf_thr=0.4`, `stride=16`, `bbox_norm=35.0`**, затем **NMS (IoU 0.5)**.
- Источник данных: **WIDER FACE** (val), **3 226 изображений**, **61 категория событий**.
  Локально: `/home/mk/Загрузки/DATASETS/Wider Face/WIDER_val` (`facedetect.data_dir` в
  `configs/local_paths.yaml`) — присутствуют `wider_face_val_bbx_gt.txt` + `images/` (61 каталог).
  См. `docs/about_datasets/wider_face.md`.
- Формат GT-txt (порядок из `wider_face_split/readme.txt`):
  `x1 y1 w h blur expression illumination invalid occlusion pose` — **поле `invalid` стоит ПЕРЕД
  `occlusion`/`pose`** (отличается от порядка на HuggingFace). bbox = `(x, y, w, h)` абсолютные
  пиксели → парсер `parse_wider_gt` конвертирует **xywh→xyxy** (`[x1, y1, x1+w, y1+h]`), отбрасывает
  боксы с `w≤0` или `h≤0`, корректно обрабатывает строку-заполнитель из 10 нулей при `N=0`.
- **Пороги pass/fail:**
  - Headline (консервативный, Hard-уровень): **AP@0.5 ≥ 0.50** (`thresholds = {"map50": 0.50}`).
  - Полная спека: **AP@0.5 Easy ≥ 0.80 / Medium ≥ 0.70 / Hard ≥ 0.50** (официальный devkit WIDER).
  - **Automotive-срез** — отдельные метрики на категориях, близких к транспортной сцене:
    `14--Traffic`, `5--Car_Accident`, `59--people--driving--car` (`FACEDETECT_AUTOMOTIVE`).
- **Риски / каваэты (делают вердикт UNDEF валидным):**
  1. **ONNX-модель локально недоступна** → прогон невозможен → **UNDEF** (см. §3/§4).
  2. **Нет официального Easy/Medium/Hard split локально** (нет `eval_tools` devkit / `.mat`-разбора по
     сложности) → подметрики Easy/Med/Hard помечены **UNDEF**, считается только overall AP@0.5
     (+ automotive-срез).
  3. Доменный разрыв: WIDER FACE — веб-фото (толпы/парады/спорт), CARS — automotive POV «через
     стекло», 5–50 м, day/night/IR; масса микро-лиц тянет вниз Hard-AP. Нет IR/ночи в WIDER FACE.
- Расхождений docs↔план по входу/выходу/классам/формату GT не выявлено; приоритет у docs.

## 2. Подготовка
Отдельной подготовки данных не требуется: WIDER FACE val используется как есть с диска
(`/home/mk/Загрузки/DATASETS/Wider Face/WIDER_val`) — `wider_face_val_bbx_gt.txt` + `images/`.
Чтение GT реализует `parse_wider_gt(txt_path)` в `deploy/evaluation/evaluate.py`: блок на изображение
(путь → N → N строк по 10 чисел), xywh→xyxy, отбрасывание нулевых боксов, обработка заполнителя при
`N=0`. Покрыто TDD-тестом `deploy/tests/test_eval_facedetect.py` (xywh→xyxy + атрибуты blur/invalid/
occlusion; нулевой бокс отброшен) — **1 passed**.

Получение модели: `deploy/scripts/get_facenet_onnx.sh` (3 шага: NGC deployable ONNX → etlt→onnx через
tao_converter → UNDEF). Команда прогона (контроллер):
`.venv/bin/python deploy/scripts/run_local.py --models facedetect`
с `model_path`/`data_dir` из `configs/local_paths.yaml`.

## 3. Измеренные метрики
**НЕ ИЗМЕРЕНО — модель недоступна.**

Прогон детекции не выполнялся: на диске нет deployable FaceNet **ONNX**, поэтому
`eval_facedetect` возвращает ранний UNDEF ещё до инференса (проверка `model_path.exists()`).
Попытка получить модель `deploy/scripts/get_facenet_onnx.sh` провалила все три пути:
- `[1/3]` NGC deployable ONNX URL → **HTTP 404** (не скачивается);
- `[2/3]` экспорт `model.etlt` → ONNX невозможен: **`tao_converter` не установлен**
  (`.etlt` зашифрован, без `tao_converter` и ключа модели не декодируется);
- `[3/3]` **FAILED → UNDEF**, exit 1.

`run_local.py --models facedetect` отрабатывает штатно (graceful):
`facedetect: ERROR/UNDEF → model not found (FaceNet ONNX unavailable) → UNDEF`, **exit 0** (без
крэша). Раннее возвращение НЕ пишет полноценный `metrics.json` через `eval_facedetect`, поэтому
маркер UNDEF записан вручную: `results/facedetect/metrics.json` (`status: UNDEF`).

Что **готово** к прогону (как только появится FaceNet ONNX):
- `parse_wider_gt` + `eval_facedetect` (+ `FACEDETECT_AUTOMOTIVE`) реализованы и зарегистрированы в
  `EVAL_CONFIGS["facedetect"]`; `parse_wider_gt` покрыт TDD-тестом (full suite **13 passed**).
- WIDER FACE val присутствует: `wider_face_val_bbx_gt.txt` + `images/` (61 категория, все три
  automotive-категории на месте: `14--Traffic`, `5--Car_Accident`, `59--people--driving--car`).
- Не хватает **только** модели (deployable ONNX).

## 4. Вердикт
**UNDEF** (не FAIL).

UNDEF — а не FAIL — потому что модель **не удалось загрузить**: мы не проводили измерения, а не
измерили и провалили порог. Три причины недоступности модели:
1. FaceNet доступен локально **только как зашифрованный `.etlt`**
   (`/home/mk/Загрузки/facenet_pruned_quantized_v2.0.1/model.etlt`).
2. **`tao_converter` не установлен** — расшифровать/экспортировать `.etlt`→ONNX нечем.
3. **NGC deployable-ONNX URL → HTTP 404** — готовый ONNX не скачивается.

При этом **код готов, а данные на месте:** `parse_wider_gt` + `eval_facedetect` реализованы и
протестированы (full suite 13 passed), WIDER FACE val GT + images присутствуют локально (включая
automotive-категории). Не хватает единственного компонента — **deployable FaceNet ONNX**. Как только
он появится (получить `tlt-model-key` и экспортировать `.etlt`→ONNX, либо рабочий NGC-линк), прогон
запускается без изменений кода: `run_local.py --models facedetect` → AP@0.5 overall + automotive-срез
(Easy/Medium/Hard останутся UNDEF до появления официального WIDER eval_tools split локально).

Ожидаемые при будущем прогоне риски занижения (см. §1): микро-лица тянут вниз Hard-AP; доменный
разрыв (веб-фото vs automotive POV «через стекло», 5–50 м); отсутствие IR/ночи в WIDER FACE. Вердикт
под CARS опирать на automotive-срез + отдельную IR/night-выборку вне WIDER FACE.
