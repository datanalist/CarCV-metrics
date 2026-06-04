# Эксперимент: Color (bae_model_f3) × MAD-Cars

Дата: 2026-06-04 · Ветка: exp/eval-color · GPU: RTX 3090 (CUDAExecutionProvider)

## 1. Выписка из docs (шаг 1 протокола)
- Модель: **bae_model_f3** — кастомная (внутренняя) CNN классификации цвета кузова,
  backbone **EfficientNet-B3** (сверено по ONNX-файлу; legacy-доки ошибочно называют «ResNet»).
  Файл `/home/mk/CarCV/models/bae_model_f3.onnx` (≈70 MB, FP32, экспорт PyTorch 1.11.0), вне
  версионируемого репозитория и вне NGC. См. `docs/about_models/bae_model_f3.md`.
- Препроцессинг (`preprocess_color` в `evaluate.py`, семейство «Generic ImageNet classifier»):
  вход **384×384**, 3 канала, **BGR→RGB**, масштаб **`/255`**, нормализация **`(x-mean)/std`**
  с `mean=[0.43, 0.40, 0.39]`, `std=[0.27, 0.26, 0.26]` (System Design §6.5), тензор **NCHW**.
- Вход/выход (сверено по ONNX): вход `[batch_size, 3, 384, 384]` float, выход `[batch_size, 15]`
  float. **Выход = сырые логиты, softmax в граф НЕ зашит** → softmax применяется в постобработке
  (`softmax(logits)`), затем top-3 (`probs.argsort()[::-1][:3]`), top-1 = первый.
- **15 классов цвета (алфавитный порядок, индекс = позиция в выходном векторе)**:
  beige, black, blue, brown, gold, green, grey, orange, pink, purple, red, silver, tan, white, yellow.
- **Источник данных — MAD-Cars** (`/home/mk/CarCV/data/external/ymad_cars`), табличный индекс
  `images_index.jsonl` (10432 строки, 2000 уникальных `car_id`, 15 различных hex-цветов).
  GT-метка цвета получается из поля `color` (hex RGB) через **HEX_TO_CARS_COLOR** (hex → имя класса CARS).
  **Dedup по `car_id`** (1 view на машину) → 2000 образцов для headline-метрики
  (без dedup ~85 views/машину сместили бы метрики).
- Пороги PASS (`eval_color`, `thresholds`): **overall (Top-1) ≥ 0.80**,
  **best_group_min (black/white/red/blue) ≥ 0.90**, **challenging_group_min (beige/tan/gold/silver) ≥ 0.70**.
- **Ожидаемое покрытие классов: 13/15**, а НЕ 14/15 (как предполагал план). Покрытие считается
  динамически по фактическим GT. Отсутствуют ДВА класса:
  - `tan` — ни один hex в HEX_TO_CARS_COLOR не маппится в `tan` (в MAD-Cars `tan` слит с beige/brown);
  - `pink` — hex `ffc0cb` (единственный → pink) в данных отсутствует (всего ~910 frames из 5.88M, в
    локальном sample 2000 машин не попал).
- **КАВЕАТЫ (делают осторожный/низкий вердикт ВАЛИДНЫМ, а не багом):**
  1. **Маппинг индекс→имя класса (COLOR_CLASSES, алфавитный) НЕПРОВЕРЯЕМ** без оригинального файла
     меток модели — модель может выдавать классы НЕ в алфавитном порядке, тогда accuracy искусственно
     занижается (вплоть до коллапса предсказаний в один-два класса).
  2. **mean/std (0.43/0.40/0.39, 0.27/0.26/0.26) нестандартны и непроверяемы** (из System Design §6.5,
     отличаются от канонического ImageNet) — риск тихой деградации при неверном препроцессинге.
  3. **Маппинг HEX_TO_CARS_COLOR (hex → имя цвета) непроверен** — взят из спеки датасета, не сверен с
     эталонной разметкой.
- Если docs противоречат — приоритет у docs; расхождений по входу/выходу/классам не выявлено.

## 2. Подготовка
Отдельная подготовка данных не требуется: используется как есть локальный дамп MAD-Cars
(`/home/mk/CarCV/data/external/ymad_cars`) — `images_index.jsonl` + каталог `images/`.
Чтение реализует `load_madcars_color_index(jsonl_path, dedup=True)` в
`deploy/evaluation/evaluate.py`: построчно парсит JSONL и делает dedup по `car_id` (1 view/машину).
GT-метка получается из `HEX_TO_CARS_COLOR[row.color]`; строки с hex без маппинга пропускаются
(`skipped`).

Команда прогона (запускает контроллер):
`.venv/bin/python deploy/scripts/run_local.py --models color`
с `model_path`/`data_dir` из `configs/local_paths.yaml`.

## 3. Измеренные метрики
Прогон: RTX 3090, **CUDAExecutionProvider** (`Loaded bae_model_f3.onnx on CUDAExecutionProvider`),
~47 с (16:44→16:45), ~44 img/s.
- **num_samples: 2000** (dedup по `car_id`, 1 view/машину). **skipped: 0** — все hex в локальном
  sample мапятся в HEX_TO_CARS_COLOR, все изображения прочитаны `cv2.imread`.
- **Покрытие классов: 13/15** (фактически, считается динамически). Отсутствуют `tan` (нет hex) и
  `pink` (hex `ffc0cb` в локальном sample не встретился) — подтверждает прогноз §1, а НЕ 14/15 из плана.
- **Top-1 accuracy (overall): 0.6205** — порог ≥ 0.80 → **FAIL**
- **Top-3 accuracy: 0.8395** (справочно; порога нет)
- **best_group_min (black/white/red/blue): 0.4793** — порог ≥ 0.90 → **FAIL**
  (минимум держит `blue` 0.479; остальные best заметно выше: black 0.897, white 0.794, red 0.730)
- **challenging_group_min (beige/tan/gold/silver): 0.0000** — порог ≥ 0.70 → **FAIL**
  (минимум держит `gold` 0.0; `tan` отсутствует в GT и в min не входит)
- Per-class accuracy (top-1), от лучшего к худшему:
  black 0.897, white 0.794, red 0.730, yellow 0.615, brown 0.506, orange 0.500, blue 0.479,
  silver 0.436, green 0.413, grey 0.401, beige 0.150, **gold 0.000, purple 0.000**;
  tan и pink — нет данных (вне покрытия).
- **Предсказания НЕ коллапсировали в один-два класса:** распределены по многим классам, при этом
  black/white/red узнаются уверенно (0.73–0.90), а top-3 (0.84) существенно выше top-1 (0.62) — это
  скорее путаница соседних цветов, чем сбитый маппинг целиком. Однако **gold=0.0 и purple=0.0**
  (предсказания этих классов не попадают в top-1 ни разу) — локальный сигнал, что отдельные индексы
  маппинга индекс→имя могут быть смещены (КАВЕАТ 1) либо классы просто трудные/малопредставленные.
- Артефакты: `results/color/metrics.json`, `results/color/per_class_metrics.csv`,
  `plots/color_confusion.png`, `plots/color_per_class.png`.

## 4. Вердикт
**FAIL** — все три порога не достигнуты: overall 0.6205 < 0.80; best_group_min 0.4793 < 0.90;
challenging_group_min 0.0000 < 0.70.

Окончательный валидный результат — **тюнинг не применялся.** Вердикт следует читать как **осторожный
(cautious)** в силу задокументированных кавеатов §1: (1) маппинг индекс→имя класса (COLOR_CLASSES,
алфавитный) непроверяем — если модель выдаёт классы в ином порядке, часть классов (наблюдаемые
`gold`/`purple` = 0.0) занижается искусственно; (2) нестандартные непроверяемые mean/std §6.5 могли
дать тихую деградацию; (3) маппинг HEX_TO_CARS_COLOR непроверен. При этом результат **не является
коллапсом** (black 0.897 / white 0.794 / red 0.730 узнаются уверенно, top-3 = 0.84), то есть базовая
конфигурация (вход 384×384, BGR→RGB, /255, softmax-постобработка) в целом рабочая, а низкая точность —
сочетание трудных/слитых цветов MAD-Cars (beige/gold/silver/grey) и возможного частичного смещения
маппинга меток. Для снятия неопределённости нужен оригинальный файл меток модели и сверка обучающих
mean/std (см. рекомендации §«Использование» спеки bae_model_f3.md).
