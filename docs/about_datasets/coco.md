# COCO 2017 Dataset (Vehicle Subset)

## Общая информация

**COCO (Common Objects in Context)** — масштабный датасет общего назначения для детекции, сегментации, keypoint estimation и captioning. В рамках CARS используется **vehicle subset** из набора детекции для кросс-валидации Primary Detector (TrafficCamNet).

**Локальное расположение:** _(скачивать при необходимости)_ — см. раздел «Получение».

**Официальный сайт:** https://cocodataset.org/

**Версия:** COCO 2017 (train/val/test split).

---

## Структура датасета

```
coco_2017/
├── annotations/
│   ├── instances_train2017.json   # 118,287 train annotations
│   ├── instances_val2017.json     #   5,000 val annotations
│   ├── image_info_test2017.json   # 40,670 test images (без публичных GT)
│   └── ... (captions, person_keypoints)
│
├── train2017/                      # 118,287 .jpg
├── val2017/                        #   5,000 .jpg
└── test2017/                       #  40,670 .jpg (без GT)
```

---

## Статистика

| Компонент | Количество | Размер (прибл.) |
|-----------|------------|------------------|
| train2017 | 118,287 изображений | ~18 GB |
| val2017 | 5,000 изображений | ~1 GB |
| test2017 | 40,670 изображений | ~6 GB (без публичных аннотаций) |
| Аннотации | JSON | ~810 MB (train), ~50 MB (val) |
| Всего классов | 80 объектных категорий | — |

---

## Vehicle subset (для CARS)

Из 80 категорий COCO к Primary Detector CARS (car/person/bike/sign) относятся следующие 6 (по `category_id`):

| `category_id` | name (COCO) | CARS-класс соответствие |
|---------------|-------------|--------------------------|
| 1 | person | `person` |
| 2 | bicycle | `bike` (часть) |
| 3 | car | `car` |
| 4 | motorcycle | `bike` (часть, объединить с bicycle при evaluation) |
| 6 | bus | `car` (или отдельный large-vehicle) |
| 8 | truck | `car` (или отдельный large-vehicle) |
| 13 | **stop sign** | ⚠️ только `stop sign`, не полный `traffic sign` |

> **Важно:** COCO **не имеет** прямого аналога `traffic sign` из §6.2 CARS. Это ограничение покрытия (3/4 целевых классов).

---

## Формат аннотаций

### Структура JSON

COCO использует единый формат с разделами `info`, `licenses`, `images`, `annotations`, `categories`:

```json
{
  "info": {...},
  "licenses": [...],
  "images": [
    {
      "id": 397133,
      "file_name": "000000397133.jpg",
      "width": 640,
      "height": 427,
      "license": 4,
      "date_captured": "2013-11-14 17:02:52",
      "flickr_url": "...",
      "coco_url": "..."
    }
  ],
  "annotations": [
    {
      "id": 1768,
      "image_id": 397133,
      "category_id": 3,
      "bbox": [388.66, 69.92, 109.41, 277.62],
      "area": 30371.0,
      "iscrowd": 0,
      "segmentation": [...]
    }
  ],
  "categories": [
    {"id": 3, "name": "car", "supercategory": "vehicle"}
  ]
}
```

### Формат bounding box

COCO bbox: `[x, y, w, h]` в абсолютных пикселях, где `(x, y)` — верхний левый угол.

- **Для YOLO:** конвертировать в `[cx, cy, w, h]` (нормализованные).
- **Для KITTI/BDD:** конвертировать в `[x1, y1, x2, y2]`.

### Атрибуты объекта

- `area` (float) — площадь объекта (для AP small/medium/large).
- `iscrowd` (0/1) — кластер объектов (RLE-сегментация вместо полигона).
- `segmentation` — polygon coords или RLE.

---

## Использование для CarCV

### Применимость

- ✅ **Кросс-валидация Primary Detector** на distribution, отличной от BDD100K → проверка robustness и overfit на dashcam-стиль.
- ✅ Лицензия CC BY 4.0 → безопасно для коммерческой публикации метрик.
- ✅ Большой val (5,000 кадров) — статистическая значимость mAP@0.5.

### Ограничения

- ❌ Нет `traffic sign` напрямую (только `stop sign`).
- ❌ Сцены — не automotive POV; распределение объектов и углов отличается от §5.1 CARS.
- ❌ Нет метаданных `weather × timeofday × scene` (как в BDD100K).
- ⚠️ Per-image Flickr licenses могут быть NC; для коммерческой публикации проверять `license` поле каждой картинки.

### Рекомендации для CP3

- Использовать **val2017** (5,000 кадров) как secondary benchmark.
- Фильтровать `annotations` по `category_id ∈ {1, 2, 3, 4, 6, 8}` для CARS-релевантных классов.
- При расчёте mAP@0.5 — объединить (2, 4) → `bike`, (3, 6, 8) → `car` (или вести три суб-метрики: car / bus / truck отдельно для интерпретации §6.2).

---

## Примеры использования

### Загрузка аннотаций через pycocotools

```python
from pycocotools.coco import COCO

coco = COCO('annotations/instances_val2017.json')

# CARS-релевантные категории
cars_cat_ids = coco.getCatIds(catNms=['car', 'bus', 'truck'])
person_cat_ids = coco.getCatIds(catNms=['person'])
bike_cat_ids = coco.getCatIds(catNms=['bicycle', 'motorcycle'])

# Все изображения с автомобилями
img_ids = coco.getImgIds(catIds=cars_cat_ids)
print(f"Images with vehicles: {len(img_ids)}")

# Аннотации для конкретного изображения
ann_ids = coco.getAnnIds(imgIds=img_ids[0], catIds=cars_cat_ids)
anns = coco.loadAnns(ann_ids)
```

### Получение датасета

```bash
# Annotations (~241 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Val images (~1 GB)
wget http://images.cocodataset.org/zips/val2017.zip

# (Опционально) Train images (~18 GB)
wget http://images.cocodataset.org/zips/train2017.zip
```

---

## Лицензия

- **Annotations:** Creative Commons Attribution 4.0 License (**CC BY 4.0**) — коммерческое использование разрешено с указанием авторов.
- **Images:** Flickr Terms of Use; каждое изображение имеет один из 8 лицензионных типов (поле `license` в JSON), некоторые подкатегории — NonCommercial.

**Рекомендация:** при использовании отдельных изображений в публичных материалах фильтровать по `licenses` JSON и оставлять только CC BY / CC BY-SA / CC0.

---

## Ссылки

- [Официальный сайт COCO](https://cocodataset.org/)
- [Документация по формату](https://cocodataset.org/#format-data)
- [pycocotools (GitHub)](https://github.com/cocodataset/cocoapi)
- [COCO 2017 Paper (Lin et al.)](https://arxiv.org/abs/1405.0312)

---

## История изменений

- **2026-05-15** — Создана документация в рамках research-датасетов для валидации ML-стека CARS (см. `_bmad-output/planning-artifacts/research-datasets-validation.md`).
