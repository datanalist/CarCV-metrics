# Car Plate OCR Dataset (RU) — `AY000554/Car_plate_OCR_dataset`

## Общая информация

**Car_plate_OCR_dataset** — корпус кропов российских номерных знаков (GOST-Р 50577) с OCR-аннотациями. Производный от данных проекта Nomeroff Net (RIA.com), очищен и приведён к единому формату. **Единственный найденный публичный RU LP датасет с явной коммерчески-совместимой лицензией.**

**Локальное расположение:** _(скачивать при необходимости)_ — см. раздел «Получение».

**HuggingFace:** https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset

**Связанный исходный проект:** https://nomeroff.net.ua/

---

## Структура датасета

```
Car_plate_OCR_dataset/
├── train/                # 37,775 .jpg (filename = plate text)
├── val/                  #  4,891 .jpg
└── test/                 #  2,845 .jpg
```

Имя файла кодирует ground-truth текст номера, например:
- `A129XY196.jpg`
- `K211PA69.jpg`
- `E353TA46.jpg`

---

## Статистика

| Компонент | Количество | Доля |
|-----------|------------|------|
| Train | 37,775 кропов | 83% |
| Val | 4,891 кропов | 10.7% |
| Test | 2,845 кропов | 6.3% |
| **Всего** | **45,514** | 100% |
| Размер архива | ~1.23 GB | — |

---

## Формат изображений

- **Тип контента:** crop номерного знака (не полный кадр).
- **Разрешение:** в HF datasheet не указано явно; типично варьируется (соответствует выходу детектора-источника).
- **Цветовое пространство:** RGB, JPEG.

---

## Алфавит (GOST-Р 50577)

```python
ALPHABET = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','E','K','M','H','O','P','C','T','Y','X']
```

22 символа: 10 цифр + 12 латинских букв, визуально идентичных кириллическим А/В/Е/К/М/Н/О/Р/С/Т/У/Х (стандарт GOST-Р 50577 для гражданских автомобильных номеров РФ).

**Сравнение с CARS §6.4 (23 символа):**

| CARS-алфавит | Car_plate_OCR_dataset | Покрытие |
|--------------|------------------------|----------|
| 0–9 | ✅ 0–9 | 10/10 |
| A,B,E,K,M,H,O,P,C,T,Y,X | ✅ ровно эти | 12/12 |
| `-` | ❌ отсутствует | 0/1 |

`-` в `LPR_STN_PRE_POST.onnx` — служебный токен (CTC blank / разделитель), не символ номера. На реальных RU GOST-Р 50577 plates символ `-` не встречается. Функционально покрытие **полное**.

---

## Формат аннотаций

OCR-only через filename. Нет bbox-разметки.

### Пример парсинга

```python
from pathlib import Path

def parse_filename(fname: str) -> str:
    """Извлекает plate text из имени файла, отбрасывая .jpg."""
    return Path(fname).stem

# A129XY196.jpg → "A129XY196"
```

### Совместимость с моделью CARS (`LPR_STN_PRE_POST.onnx`)

| Параметр | CARS модель | Датасет |
|----------|-------------|---------|
| Вход модели | 188×48×3 RGB | crop переменного размера |
| Выход | строка длины ≤ N | строка из filename |
| Алфавит | 22 + `-` (служ.) | 22 |

**Требуется на стороне валидатора:**
1. Resize crop → 188×48 (preserve aspect ratio, padding).
2. Pre-processing per LPR_STN_PRE_POST.onnx (normalization).
3. Inference → output string.
4. Сравнение с filename → Full Plate Accuracy + Character Accuracy.

---

## Использование для CarCV

### Применимость

- ✅ **CP4 (LP Accuracy >0.85)** — валидация Full Plate Accuracy на val+test = 7,736 примеров. Размер достаточен для статистически значимой оценки.
- ✅ **CP4 — Character Accuracy** (целевое >0.92 из §6.4) — agregated по позициям.
- ✅ Алфавит полностью совпадает с §6.4 CARS (22/22 содержательных символов).
- ✅ **Лицензия CC BY 4.0 — безусловно безопасна для коммерческой валидации и публикации метрик.**

### Ограничения

- ❌ **Нет bbox-аннотаций** → не валидирует LP Detection stage (LPDNet) из §6.4 → не покрывает «LP Detection Recall >0.80» из §3.3.
- ❌ Только кропы — нет полных кадров → нет real-world end-to-end pipeline test.
- ❌ Только GOST-Р 50577 (стандартные гражданские); специальные/прицепные/такси — отсутствуют.
- ⚠️ Distribution регионов РФ внутри датасета не задокументирован → возможны смещения.

### Рекомендации для CP4

1. Использовать `val + test` (7,736 plates) как primary benchmark Full Plate Accuracy.
2. Train (37,775) — **не использовать** для валидации (риск if dataset overlap с обучающим корпусом nomeroff-net derived models).
3. Считать метрику с учётом padding/blank token:
   ```python
   predicted = decode_ctc(model_output, blank_token='-')
   gt = filename.stem
   full_plate_match = (predicted == gt)
   char_acc = sum(p == g for p, g in zip(predicted, gt)) / max(len(gt), 1)
   ```
4. Сегментировать ошибки по длине номера (8 vs 9 символов — частный/региональный код).

### Для LP Detection (отдельно)

Использовать **Kaggle `c/car-plates-recognition`** competition data (RU plates с bbox) или собственную разметку 1000 CARS-собранных кадров (CP4 ground-truth comparison из §12).

---

## Получение

```python
from datasets import load_dataset

# HuggingFace Datasets API
ds = load_dataset("AY000554/Car_plate_OCR_dataset")

print(ds)
# DatasetDict({
#     train: Dataset({features: ['image'], num_rows: 37775})
#     val:   Dataset({features: ['image'], num_rows: 4891})
#     test:  Dataset({features: ['image'], num_rows: 2845})
# })
```

Прямая загрузка архива через HF CLI:

```bash
huggingface-cli download AY000554/Car_plate_OCR_dataset --repo-type dataset --local-dir ./data/car_plate_ocr_ru
```

---

## Лицензия

**CC BY 4.0** — Creative Commons Attribution 4.0 International.

Разрешено:
- ✅ Коммерческое использование.
- ✅ Модификация и распространение.
- ✅ Публикация метрик и использование в маркетинговых материалах.

Обязательно:
- Указание авторства датасета (`AY000554` на HuggingFace) и исходного проекта Nomeroff Net.

---

## Ссылки

- [HuggingFace dataset card](https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset)
- [Nomeroff Net (исходный проект)](https://nomeroff.net.ua/)
- [ria-com/nomeroff-net (GitHub)](https://github.com/ria-com/nomeroff-net)
- [ГОСТ-Р 50577 (Wikipedia)](https://ru.wikipedia.org/wiki/%D0%9D%D0%BE%D0%BC%D0%B5%D1%80%D0%BD%D0%BE%D0%B9_%D0%B7%D0%BD%D0%B0%D0%BA_%D1%82%D1%80%D0%B0%D0%BD%D1%81%D0%BF%D0%BE%D1%80%D1%82%D0%BD%D1%8B%D1%85_%D1%81%D1%80%D0%B5%D0%B4%D1%81%D1%82%D0%B2_%D0%A0%D0%BE%D1%81%D1%81%D0%B8%D0%B8)

---

## История изменений

- **2026-05-15** — Создана документация в рамках research-датасетов для валидации ML-стека CARS (см. `_bmad-output/planning-artifacts/research-datasets-validation.md`).
