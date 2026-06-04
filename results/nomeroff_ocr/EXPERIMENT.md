# Эксперимент: nomeroff_ocr × AUTO.RIA Numberplate OCR (RU)

Дата: 2026-06-04 · Ветка: exp/eval-nomeroff-ocr · GPU: NVIDIA (см. `nvidia-smi`)

**Вердикт: DEFERRED** (прогон не выполнен — см. §4). Метрики локально **НЕ ИЗМЕРЕНЫ**.

## 1. Выписка из docs (шаг 1 протокола)
- Модель: **nomeroff_ocr** — OCR-модель для чтения текста российского регистрационного знака
  (ГОСТ-Р 50577) с уже вырезанного кропа пластины. Архитектура — **CRNN (CNN ResNet-18 +
  рекуррентная часть + CTC-декодер)**, RU-вариант `model_v3.3`, чекпойнт
  `anpr_ocr_ru_2023_02_01_resnet18.ckpt`. Проект **Nomeroff Net** (`ria-com/nomeroff-net`,
  nomeroff.net.ua). Финальная ступень распознавания номера в конвейере CARS (после LP Detection
  `nomeroff_lpd`); заменяет US-обученный LPRNet (NVIDIA TAO), проваливающий RU-номера.
  См. `docs/about_models/nomeroff_ocr.md`.
- Запуск: **напрямую через `NumberPlateTextReading`** (task `"number_plate_text_reading"`),
  минуя YOLO-детекцию — реализовано в `deploy/evaluation/evaluate.py` (`eval_nomeroff_ocr`).
  Параметры пресета: `for_regions=["ru"]`, `for_count_lines=[1]` (однострочные),
  `model_path="latest"`, `off_number_plate_classification=True`; `image_loader` переопределяется
  на `DumpyImageLoader()` (numpy-кроп «как есть»). Препроцессинг (resize/нормализация/CTC-декод)
  инкапсулирован в пайплайне Nomeroff Net.
  - Нюанс окружения: PyTorch ≥ 2.6 (дефолт `weights_only=True`) ломает загрузку старого чекпойнта
    (содержит `StrLabelConverter`); в `eval_nomeroff_ocr` решается временным monkey-patch
    `torch.load(weights_only=False)` только на время инициализации (не потокобезопасен).
- Алфавит выхода: нативный RU (ГОСТ-Р 50577), 22 содержательных символа —
  `0123456789` + `A B E K M H O P C T Y X`.
- Датасет: **AUTO.RIA Numberplate OCR RU Dataset**, версия `autoriaNumberplateOcrRu-2021-09-01`,
  сплит **val = 4893 кропа** (`val/img` + `val/ann`). Один JSON на кроп (стиль Supervisely/Nomeroff).
  **Ground-truth текст берётся ТОЛЬКО из поля `description`** (не из имени файла, не из `name`,
  не из `predicted`); в эвалуаторе: `data.get("description") or data.get("name")`. Кропы часто RGBA
  → конвертация в RGB. См. `docs/about_datasets/autoria_numberplate_ocr_ru.md`.
- Метрики (`metrics.py`, `compute_ocr_metrics`): **char_accuracy** — посимвольное (позиционное)
  совпадение `zip(pred, gt)` / `max(len)`; **full_plate_accuracy** — доля точных совпадений строки;
  `char_error_rate = 1 − char_accuracy`.
- **Пороги PASS/FAIL** (`eval_nomeroff_ocr`): **char_accuracy ≥ 0.90**, **full_plate_accuracy ≥ 0.80**.
- Кавеат датасета: `autoriaNumberplateOcrRu` — первоисточник обучения Nomeroff Net OCR → риск
  train-leakage; train (49 382) для оценки НЕ использовать.
- Если docs противоречат — приоритет у docs; расхождений по входу/выходу/алфавиту не выявлено.

## 2. Подготовка
- **Веса:** чекпойнт `anpr_ocr_ru_2023_02_01_resnet18.ckpt` (~27 MB) штатно тянется пайплайном
  nomeroff-net (`model_path="latest"`); локально найдена ручная копия по пути
  `/home/mk/CarCV/models/nomeroff_net/ocr/anpr_ocr_ru_2023_02_01_resnet18.ckpt`
  (вне репозитория `CarCV-metrics`, эвалуатор её напрямую не использует).
- **Данные:** AUTO.RIA OCR RU val —
  `/home/mk/Загрузки/DATASETS/nomeroff/autoriaNumberplateOcrRu-2021-09-01/val`
  (`val/img` 4893 PNG + `val/ann` 4893 JSON); GT — из `description`.
- **Зависимость `nomeroff-net` (блокер прогона):** контроллер попытался поставить `nomeroff-net`
  в `.venv`. Установка **успешна** — `uv pip install nomeroff-net` ставит nomeroff-net 4.0.1 и
  **не ломает окружение** (numpy/torch/onnxruntime остаются нетронутыми). Однако
  `import nomeroff_net` **падает** (см. §4): требуемая транзитивная зависимость `modelhub_client`
  (PyPI-имя `modelhub-client`) **отсутствует на PyPI**, поэтому пакет неимпортируем.
  Установка `modelhub-client` из произвольного git признана **вне рамок плана** («без долгих
  раскопок»). Эвал-функция `eval_nomeroff_ocr` уже существует в `deploy/evaluation/evaluate.py`,
  но запущена быть **не может** из-за неимпортируемой зависимости.

## 3. Измеренные метрики
**НЕ ИЗМЕРЕНО.** Прогон не выполнен — см. блокер в §2 и причину в §4. Локальных чисел нет;
никакие метрики не фабриковались.

## 4. Вердикт
**Вердикт: DEFERRED** (прогон не выполнялся, не FAIL).

**Причина** (тот же корневой блокер, что у `nomeroff_lpd`). Эвалуатор `eval_nomeroff_ocr` полагается
на пакет `nomeroff-net`. В текущем окружении:
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
оценка `nomeroff_ocr` **отложена**.

**Прежний удалённый (qudata) результат — НЕ воспроизведён локально (ориентир, не локальное измерение):**
прежний REMOTE-прогон на ssh9.qudata.ai измерял nomeroff_ocr как **PASS**:
**CharAcc = 0.9995, PlateAcc = 0.9978** (оба порога char≥0.90 / plate≥0.80 пройдены на удалёнке).
Эти числа приводятся **исключительно как удалённый ориентир** и **локально НЕ подтверждены** —
их нельзя выдавать за результат данного (локального) эксперимента.
