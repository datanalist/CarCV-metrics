# Эксперимент: nomeroff_lpd × AUTO.RIA Numberplate Detection (RU)

Дата: 2026-06-04 · Ветка: exp/eval-nomeroff-lpd · GPU: NVIDIA (см. `nvidia-smi`)

**Вердикт: DEFERRED** (прогон не выполнен — см. §4). Метрики локально **НЕ ИЗМЕРЕНЫ**.

## 1. Выписка из docs (шаг 1 протокола)
- Модель: **nomeroff_lpd** — детектор номерных пластин на базе **Nomeroff Net**
  (`ria-com/nomeroff-net`, сайт nomeroff.net.ua). Семейство — **YOLO11x (Ultralytics)** с
  keypoint/pose-головой; помимо bbox регрессирует **4 угловые точки** пластины. Стадия конвейера
  CARS — **LP Detection** (локализация пластины RU/UA-сегмента + 4 угла для перспективного
  выравнивания перед OCR). Введена как замена US-обученного LPDNet (NVIDIA TAO), который
  консервативен на не-US пластинах (низкий recall). См. `docs/about_models/nomeroff_lpd.md`.
- Запуск: **не напрямую**, а через высокоуровневый pipeline `nomeroff-net`
  (`pipeline("number_plate_localization", image_loader="opencv")`) — реализовано в
  `deploy/evaluation/evaluate.py` (`eval_nomeroff_lpd`). Препроцессинг (letterbox-resize, `/255`,
  RGB, NCHW) инкапсулирован внутри YOLO; на вход эвалуатора подаётся путь к файлу полного кадра.
- Веса (по `models.md` / по имени файла): `yolov26x-keypoints-2026-01-21.pt` (вариант
  `yolov11x/yolov26x-keypoints`, RU LP detection). В `models.md` версия = `None`, столбец
  `Local path` для строки `nomeroff_lpd` не задан — веса штатно тянутся пайплайном по требованию;
  локально найдена копия `.pt` (см. §2).
- Датасет: **AUTO.RIA Numberplate Dataset (Detection)**, версия `autoriaNumberplateDataset-2021-05-12`,
  сплит **val** (376 изображений, 405 полигонов пластин). Аннотации — формат **VGG Image Annotator
  (VIA)**, полный проектный экспорт `via_region_data.json` (ключи `_via_settings`,
  `_via_img_metadata`, `_via_attributes`). Каждый регион — полигон (обычно 4 точки,
  `all_points_x` / `all_points_y`), сводится к axis-aligned bbox через `min/max`. См.
  `docs/about_datasets/autoria_numberplate_detection_ru.md`.
- Метрика: GT-полигоны → axis-aligned bbox; предсказания — bbox из выхода pipeline; сопоставление
  по **IoU** с **conf_threshold = 0.3** (`compute_detection_metrics`, `eval_nomeroff_lpd`).
  Keypoint-метрика (4 угла GT ↔ 4 предсказанных keypoints, OKS) в текущем эвалуаторе не реализована.
- **Пороги PASS/FAIL** (по спеке кампании): **Precision ≥ 0.70**, **Recall ≥ 0.80**.
- Если docs противоречат — приоритет у docs; расхождений по входу/выходу/классам не выявлено.

## 2. Подготовка
- **Веса:** локальная копия `.pt` найдена на машине разработчика по пути
  `/home/mk/CarCV/models/nomeroff_net/object_detection/yolov26x-keypoints-2026-01-21.pt`
  (вне репозитория `CarCV-metrics`; штатно пайплайн тянет веса по требованию).
- **Данные:** AUTO.RIA detection val —
  `/home/mk/Загрузки/DATASETS/nomeroff/autoriaNumberplateDataset-2021-05-12/val`
  (376 изображений + `via_region_data.json`).
- **Зависимость `nomeroff-net` (блокер прогона):** контроллер попытался поставить `nomeroff-net`
  в `.venv`. Установка **успешна** — `uv pip install nomeroff-net` ставит nomeroff-net 4.0.1 и
  **не ломает окружение** (numpy/torch/onnxruntime остаются нетронутыми). Однако
  `import nomeroff_net` **падает** (см. §4): требуемая транзитивная зависимость `modelhub_client`
  (PyPI-имя `modelhub-client`) **отсутствует на PyPI**, поэтому пакет неимпортируем.
  Установка `modelhub-client` из произвольного git признана **вне рамок плана** («без долгих
  раскопок»). Эвал-функция `eval_nomeroff_lpd` уже существует в `deploy/evaluation/evaluate.py`,
  но запущена быть **не может** из-за неимпортируемой зависимости.

## 3. Измеренные метрики
**НЕ ИЗМЕРЕНО.** Прогон не выполнен — см. блокер в §2 и причину в §4. Локальных чисел нет;
никакие метрики не фабриковались.

## 4. Вердикт
**Вердикт: DEFERRED** (прогон не выполнялся, не FAIL).

**Причина.** Эвалуатор `eval_nomeroff_lpd` полагается на пакет `nomeroff-net`. В текущем окружении:
- `uv pip install nomeroff-net` устанавливается успешно (nomeroff-net 4.0.1) и не изменяет
  numpy/torch/onnxruntime;
- но `import nomeroff_net` падает с ошибкой:

  ```
  ModuleNotFoundError: No module named 'modelhub_client'
  ```

- требуемая зависимость `modelhub_client` (PyPI-имя `modelhub-client`) **отсутствует на PyPI** —
  uv сообщает: «Because modelhub-client was not found in the package registry ... your requirements
  are unsatisfiable.» (в venv нет pip; кастомный индекс не настроен);
- установка `modelhub-client` из произвольного git — **вне рамок плана** («без долгих раскопок»).

Это допустимый окончательный исход по плану: «Если падает → зафиксировать вердикт deferred
(с текстом ошибки)». nomeroff-net в данном окружении импортировать/использовать нельзя →
оценка `nomeroff_lpd` **отложена**.

**Прежний удалённый (qudata) результат — НЕ воспроизведён локально (ориентир, не локальное измерение):**
прежний REMOTE-прогон на ssh9.qudata.ai измерял nomeroff_lpd как **PASS**:
**P = 0.9056, R = 0.9221, F1 = 0.9138** (оба порога P≥0.70 / R≥0.80 пройдены на удалёнке).
Эти числа приводятся **исключительно как удалённый ориентир** и **локально НЕ подтверждены** —
их нельзя выдавать за результат данного (локального) эксперимента.
