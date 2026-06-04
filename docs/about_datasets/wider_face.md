# WIDER FACE Dataset

## Общая информация

**WIDER FACE** — эталонный бенчмарк для детекции лиц, отобранный из публичного датасета WIDER. Опубликован в работе Yang, Luo, Loy, Tang «WIDER FACE: A Face Detection Benchmark» (CVPR 2016, MMLab @ CUHK). На момент публикации был в 10 раз крупнее существовавших датасетов детекции лиц.

Ключевые числа (официально): **32 203 изображения** и **393 703 размеченных лица** с высокой вариативностью масштаба, позы, окклюзии и освещения. Датасет организован по **61 категории событий** (event class; категории пронумерованы от `0--Parade` до `61--Street_Battle`, при этом номер `60` в нумерации пропущен — каталогов ровно 61). Внутри каждой категории данные случайно разбиты в пропорции **40% train / 10% val / 50% test** (12 880 / 3 226 / 16 097 изображений). По уровню сложности детекции выделяются три подмножества — **Easy / Medium / Hard**.

**Локальное расположение:** `/home/mk/Загрузки/DATASETS/Wider Face`

> Локально присутствуют только подмножества **val** и **test** плюс метаданные сплита. Каталога `WIDER_train` с изображениями на диске **нет** (хотя файл аннотаций train `wider_face_train_bbx_gt.txt` поставлен в составе сплита).

**Официальный сайт:** http://shuoyang1213.me/WIDERFACE/

**HuggingFace:** https://huggingface.co/datasets/CUHK-CSE/wider_face

**Статья (paper):** Yang S., Luo P., Loy C. C., Tang X. «WIDER FACE: A Face Detection Benchmark», CVPR 2016 — https://arxiv.org/abs/1511.06523

---

## Структура датасета

```
/home/mk/Загрузки/DATASETS/Wider Face/
├── WIDER_val/                          # Валидационный сплит (~357 MB)
│   ├── images/
│   │   ├── 0--Parade/*.jpg
│   │   ├── 14--Traffic/*.jpg
│   │   ├── 5--Car_Accident/*.jpg
│   │   ├── 42--Car_Racing/*.jpg
│   │   ├── 59--people--driving--car/*.jpg
│   │   └── ...                         # всего 61 каталог категорий событий
│   ├── wider_face_val_bbx_gt.txt       # bbox + атрибуты для val
│   └── wider_face_val.mat              # тот же ground truth в формате MATLAB
│
├── WIDER_test/                         # Тестовый сплит (~1.8 GB)
│   └── images/
│       ├── 0--Parade/*.jpg
│       └── ...                         # 61 каталог категорий, БЕЗ bbox-аннотаций
│
└── wider_face_split/                   # Метаданные сплита (~7 MB)
    ├── readme.txt                      # кодировка атрибутов и формат txt
    ├── wider_face_train_bbx_gt.txt     # ground truth train (изображений локально нет)
    ├── wider_face_test_filelist.txt    # список файлов test (без bbox)
    ├── wider_face_train.mat
    └── wider_face_test.mat
```

> Каталог `WIDER_train/images/` локально отсутствует. Для использования train-сплита изображения нужно докачать с официального источника (см. раздел «Получение»). Обратите внимание: в `wider_face_split/` присутствует `wider_face_train.mat`, но **нет** `wider_face_val.mat` — последний лежит рядом с изображениями в `WIDER_val/`.

---

## Статистика

| Компонент | Количество | Размер (прибл.) |
|-----------|-----------|-----------------|
| **Весь датасет** (официально) | 32 203 изображения / 393 703 лица | — |
| train (официально) | 12 880 изображений (40%) | локально только аннотации |
| **val (локально)** | 3 226 изображений, 61 категория | ~357 MB |
| **test (локально)** | 16 097 изображений, 61 категория | ~1.8 GB |
| `wider_face_split/` (локально) | метаданные + 5 файлов | ~7 MB |
| Категорий событий | 61 (`0--Parade` … `61--Street_Battle`, номер `60` пропущен) | — |
| Подмножества сложности | Easy / Medium / Hard | — |

Локальные числа измерены на диске (`find ... -name '*.jpg' | wc -l`) и совпадают с официальными размерами val (3 226) и test (16 097). Каталогов категорий в `WIDER_val/images/` и `WIDER_test/images/` — ровно 61 в каждом.

---

## Формат изображений

- Формат: **JPEG** (`.jpg`), RGB. Изображения собраны из веба, **разрешение и соотношение сторон варьируются** (фиксированного размера, как 1920×1080 в automotive-сценарии, нет).
- Организация по каталогам категорий событий: `images/<NN--EventName>/<NN_EventName_..._k>.jpg`.
- Именование файла: `<id>_<EventName>_<EventName>_<eventId>_<seq>.jpg`, например `0_Parade_marchingband_1_849.jpg`.
- Сильная вариативность: очень мелкие лица (вплоть до единиц пикселей по ширине/высоте), толпы, окклюзии, экстремальное освещение, нетипичные позы.

---

## Формат аннотаций

Ground truth хранится в текстовом файле (`wider_face_val_bbx_gt.txt`). Структура — блок на каждое изображение:

```
0--Parade/0_Parade_marchingband_1_465.jpg          <- строка 1: относительный путь категория/файл.jpg
126                                                 <- строка 2: N (число лиц)
345 211 4 4 2 0 0 0 2 0                              <- далее N строк по 10 чисел на лицо
331 126 3 3 0 0 0 1 0 0
...
```

Каждая строка лица содержит 10 целых чисел в порядке (см. `wider_face_split/readme.txt`):

```
x1  y1  w  h  blur  expression  illumination  invalid  occlusion  pose
```

> **Внимание:** в txt-файле порядок атрибутов `... illumination, invalid, occlusion, pose` (поле `invalid` стоит ПЕРЕД `occlusion`/`pose`). Это отличается от порядка полей в карточке HuggingFace, где `faces` отдаёт их как `blur, expression, illumination, occlusion, pose, invalid`. При парсинге локального txt используйте порядок из `readme.txt`.

| Поле | Тип | Семантика |
|------|-----|-----------|
| `x1`, `y1` | int (px) | координаты левого верхнего угла bbox |
| `w`, `h` | int (px) | ширина и высота bbox |
| `blur` | int | 0 — clear, 1 — normal blur, 2 — heavy blur |
| `expression` | int | 0 — typical, 1 — exaggerate |
| `illumination` | int | 0 — normal, 1 — extreme |
| `invalid` | int | 0 — valid, 1 — invalid (лицо помечено как непригодное — низкое разрешение / очень мелкий масштаб; конвенция «ignore» как в PASCAL VOC) |
| `occlusion` | int | 0 — no, 1 — partial, 2 — heavy |
| `pose` | int | 0 — typical, 1 — atypical |

Особенности:
- **bbox = `(x, y, w, h)`** — абсолютные пиксели, левый верхний угол + ширина/высота (тот же формат, что COCO `bbox`).
- **Изображения без лиц:** счётчик `N` равен `0`, после чего идёт **строка-заполнитель из десяти нулей** `0 0 0 0 0 0 0 0 0 0` (не пустая строка). В **локальном `wider_face_val_bbx_gt.txt` таких изображений НЕТ** — все 3 226 валидационных изображений содержат ≥1 лицо (проверено на диске). Случай `N=0` встречается в `wider_face_train_bbx_gt.txt` (на диске — 4 записи, например `0--Parade/0_Parade_Parade_0_452.jpg`), поэтому парсер обязан корректно обрабатывать его при работе с train-аннотациями.
- **test-сплит bbox НЕ содержит** — поставляется только `wider_face_test_filelist.txt` (список путей). Официальная оценка на test выполняется через evaluation-сервер / devkit MMLab.
- `.mat`-файлы (`wider_face_val.mat`, `wider_face_train.mat`, `wider_face_test.mat`) содержат тот же ground truth в формате MATLAB и используются официальным devkit для расчёта AP по Easy/Medium/Hard.

Пример парсинга аннотаций (учитывает строку-заполнитель при `N=0`):

```python
def parse_wider_gt(path):
    """Парсит wider_face_*_bbx_gt.txt -> {rel_path: [ {bbox, blur, ...}, ... ]}.

    Порядок чисел в строке лица: x1 y1 w h blur expression illumination
    invalid occlusion pose (как в wider_face_split/readme.txt).
    При N=0 после счётчика идёт одна строка-заполнитель '0 0 0 0 0 0 0 0 0 0'.
    """
    out = {}
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    while i < len(lines):
        name = lines[i]; i += 1
        n = int(lines[i].split()[0]); i += 1
        faces = []
        if n == 0:
            i += 1                # пропускаем строку-заполнитель из нулей
        else:
            for _ in range(n):
                x, y, w, h, blur, expr, illum, invalid, occl, pose = map(int, lines[i].split())
                faces.append({"bbox": [x, y, w, h], "blur": blur, "expression": expr,
                              "illumination": illum, "invalid": invalid,
                              "occlusion": occl, "pose": pose})
                i += 1
        out[name] = faces
    return out
```

---

## Использование для CarCV

В CARS WIDER FACE обслуживает прежде всего **валидацию детекции лиц** моделью **FaceDetect** (NVIDIA FaceNet, `pruned_quantized_v2.0.1`; локальный путь модели `home/mk/Загрузки/CarCVModels/facenet_pruned_quantized_v2.0.1`). Детекция лица в CARS питает требование **FR-06** (сохранение кропов лиц), поля БД `face_count` и `face_coords`, а также артефакты `face_images/{track_id}.bmp`.

> **Задача «Face embedding» — будущая.** В `models.md` для эмбеддинга лиц указано `None (Trainable)` — модели эмбеддинга в стеке CARS **пока нет**. Поэтому на текущем этапе WIDER FACE применяется только для валидации **детекции** лиц; пригодность для обучения/оценки эмбеддинга рассматривается как задел на будущее (датасет не содержит идентичностей/меток персон, для эмбеддинга он малопригоден).

### Применимость

- ✅ Валидация детектора лиц **FaceDetect** (bbox-аннотации val в формате `x,y,w,h`).
- ✅ Стандартная метрика бенчмарка — **Average Precision (AP)** на подмножествах **Easy / Medium / Hard**; альтернатива — Recall при фиксированном precision.
- ✅ Богатые атрибуты лица (blur, occlusion, pose, illumination, expression) → срез качества по сложным условиям, релевантным automotive POV (размытие при движении, частичная окклюзия, нетипичные позы).
- ✅ Наличие категорий, близких к транспортной сцене: `14--Traffic`, `5--Car_Accident`, `42--Car_Racing`, `59--people--driving--car`.

### Ограничения

- ❌ **Domain gap:** WIDER FACE — разнородные веб-фото (толпы, парады, спорт), а CARS — automotive POV: лицо через лобовое стекло, на дистанции 5–50 м, day/night/IR. Распределение сцен расходится.
- ❌ Локально **нет изображений train** (`WIDER_train/images/`) — только val + test + аннотации сплита.
- ❌ **test-сплит без bbox** — независимая локальная оценка на нём невозможна (нужен evaluation-сервер MMLab).
- ⚠️ Очень много **экстремально мелких лиц** (несколько пикселей), которых в CARS-сценарии практически не встретить; они тянут вниз Hard-AP и искажают вердикт под automotive-условия.
- ⚠️ Все фото — **RGB/дневной веб-контент**; **IR-режим** (FaceDetect работает по RGB/IR) датасетом не покрывается — ночные/ИК-сценарии CARS остаются непокрытыми.
- ⚠️ Нет меток идентичности персон → датасет **не подходит** для обучения/оценки будущей модели эмбеддинга лиц.

### Рекомендации

1. **Базовая валидация FaceDetect:** прогнать FaceNet `pruned_quantized_v2.0.1` по `WIDER_val` (3 226 изображений), рассчитать **AP отдельно по Easy / Medium / Hard** официальным devkit (`.mat` ground truth). Это эталонный режим бенчмарка для вердикта pass/fail детектора.
2. **Приближение к automotive-сценарию:** для среза, близкого к CARS, фильтровать val по категориям `14--Traffic`, `5--Car_Accident`, `59--people--driving--car` (и при необходимости `42--Car_Racing`) и считать метрики только на этом подмножестве — оно ближе к POV «лицо в/возле автомобиля».
3. **Отсечение нерелевантного масштаба:** при оценке под FR-06 отфильтровать боксы с малой стороной (например `h < 20–30 px`) и помеченные `invalid=1` — мелкие лица из толпы не отражают целевую дистанцию 5–50 м и занижают Recall несопоставимо со сценарием CARS.
4. **Интерпретация вердикта:** результат на WIDER FACE — это метрика детекции в условиях, лишь частично совпадающих с CARS (RGB, день; нет IR, нет «через стекло»). FAIL на Hard-подмножестве из-за микро-лиц — **валидный результат**, а не повод для тюнинга; вердикт по сценарию CARS опирать на отфильтрованный automotive-срез + отдельную IR/night-выборку вне WIDER FACE.
5. **Эмбеддинг лиц — на будущее:** для задачи «Face embedding» (`None (Trainable)`) WIDER FACE не использовать (нет идентичностей); понадобится отдельный датасет с метками персон, когда модель эмбеддинга появится.

---

## Получение

WIDER FACE распространяется через официальный сайт проекта и зеркало на HuggingFace.

Через HuggingFace `datasets`:

```python
from datasets import load_dataset

# Доступны сплиты: "train", "validation", "test"
ds = load_dataset("CUHK-CSE/wider_face", split="validation")
print(ds[0]["image"], ds[0]["faces"]["bbox"])
```

Прямое скачивание архивов (официальный источник — Google Drive ссылки на странице проекта):

```bash
# Страница со ссылками на WIDER_train.zip / WIDER_val.zip / WIDER_test.zip
# и архив аннотаций wider_face_split.zip:
#   http://shuoyang1213.me/WIDERFACE/
# Для докачки отсутствующего локально train-сплита нужен WIDER_train.zip
```

Структура после распаковки совпадает с разделом «Структура датасета» (`WIDER_train/`, `WIDER_val/`, `WIDER_test/`, `wider_face_split/`).

---

## Лицензия

**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** — указано в карточке датасета на HuggingFace (`CUHK-CSE/wider_face`). Изображения отобраны из публично доступного датасета WIDER (веб-источники); датасет позиционируется как бенчмарк для академических исследований.

Что разрешено / запрещено по CC BY-NC-ND 4.0:
- ✅ Использование и распространение **с указанием авторства** (Attribution).
- ❌ **NonCommercial** — коммерческое использование запрещено.
- ❌ **NoDerivatives** — распространение производных (модифицированных) версий запрещено.

**Вывод для CARS (коммерческий продукт):**
- ⚠️ Использование WIDER FACE для **внутренней валидации/бенчмаркинга** ML-стека (CP-prep, отчёты для команды) допустимо как research-активность, но формально лицензия **NonCommercial** не разрешает применение в составе коммерческого продукта или его процессов разработки. До любого внешнего использования (публикация метрик, маркетинг, поставка артефактов заказчику) **требуется legal review** и/или получение отдельного разрешения у правообладателя (MMLab @ CUHK).
- ❌ Запрещено включать изображения/производные WIDER FACE в дистрибутив продукта, обучающие пайплайны коммерческого назначения и публичные материалы без согласования.
- Рекомендуется рассматривать WIDER FACE как **внутренний research-бенчмарк детекции** с пометкой о некоммерческой лицензии в трекинге датасетов.

---

## Ссылки

- [Официальная страница проекта (MMLab @ CUHK / Shuo Yang)](http://shuoyang1213.me/WIDERFACE/)
- [Карточка датасета на HuggingFace (CUHK-CSE/wider_face)](https://huggingface.co/datasets/CUHK-CSE/wider_face)
- [Статья CVPR 2016 (arXiv:1511.06523)](https://arxiv.org/abs/1511.06523)
- [Результаты бенчмарка (WiderFace Results, MMLab)](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)
- [TensorFlow Datasets: wider_face](https://www.tensorflow.org/datasets/catalog/wider_face)

---

## История изменений

- **2026-06-04** — Создана документация в рамках валидационной кампании на обновлённом datasets.md.
