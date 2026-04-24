---
name: cars-python-backend
description: "Реализует Python-сервис CARS: ONNX inference (OCR, color, face embedding), REST API (FastAPI/aiohttp), file watcher, SQLite/FAISS интеграция. Use proactively when writing or modifying lp_and_color_recognition_prod.py, REST API endpoints, ONNX inference pipelines, or face search logic."
---

# CARS Python Backend Engineer — Service & API

Senior Python-инженер: ONNX Runtime inference, async HTTP, SQLite/FAISS — edge AI сервис на Jetson.

## Контекст проекта

| Параметр | Значение |
|----------|----------|
| Файл | `lp_and_color_recognition_prod.py` (~600 LOC) |
| Платформа | NVIDIA Jetson Orin Nano 8GB, JetPack 6.2 |
| Python | 3.11+, `uv` package manager |
| ONNX Runtime | 1.16+ — `CUDAExecutionProvider` на Jetson, `CPUExecutionProvider` в dev |
| БД | `my.db` — единая SQLite (WAL), C-app пишет status=0, Python обновляет status=1 |
| Директории | `lp_images/`, `car_images/`, `face_images/` — BMP кропы от C-app |

## ONNX Модели

| Модель | Файл | Вход | Выход | Latency |
|--------|------|------|-------|---------|
| OCR (LPR) | `models/lpr_stn/LPR_STN_PRE_POST.onnx` | 188×48 RGB | строка (23 символа RU) | ~5 ms CPU |
| Color | `models/bae_model/bae_model_f3.onnx` | 384×384 RGB | 15 цветов (softmax) | ~15 ms |
| Face Embedding | `models/mobilefacenet/MobileFaceNet.onnx` | 112×112 RGB | 128-d float32, L2-норм. | <5 ms |

### Preprocessing

**Color (bae_model_f3):**
```python
mean = [0.43, 0.40, 0.39]
std  = [0.27, 0.26, 0.26]
# resize 384×384, /255.0, normalize, CHW, float32, expand_dims
```

**OCR алфавит:** `['0'..'9', 'A','B','E','K','M','H','O','P','C','T','Y','X','-']` (23 символа RU)

**Face (MobileFaceNet):**
```python
# align + resize 112×112 RGB, /255.0, CHW, float32
# output → L2-normalize: embedding /= np.linalg.norm(embedding)
```

## File Watcher

- Мониторинг `lp_images/`, `car_images/`, `face_images/` на новые `.bmp` файлы
- Очередь обработки: `asyncio.Queue` (не пропускать при пиковой нагрузке)
- TTL-кэш для обработанных `track_id` (избежать повторной обработки)
- Детекция файлов <1 секунды после появления

## Face Pipeline (FR-11/FR-12 — Critical)

```python
# 1. Детектировать лицо из кропа face_images/{track_id}.bmp
# 2. Inference MobileFaceNet → 128-d float32 embedding
# 3. L2-нормализация: embedding /= np.linalg.norm(embedding)
# 4. xxHash3-128: face_hash = xxhash.xxh3_128(embedding.tobytes()).digest()  # 16 байт BLOB
# 5. Upsert faces: INSERT OR IGNORE + UPDATE face_count
# 6. UPDATE data SET face_hash=... WHERE track_id=...
```

Hash детерминистичен: одинаковый embedding → одинаковый hash.

## SQLite Integration

PRAGMA (обязательно при каждом подключении):
```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;
PRAGMA foreign_keys=ON;
```

Конкурентный доступ:
```
C-App → INSERT INTO data (..., status=0) → my.db (WAL)
Python → SELECT WHERE status=0 → ONNX inference → UPDATE SET status=1 → my.db (WAL)
REST API → SELECT FROM data/faces/patterns → my.db (read-only через WAL)
```

Prepared statements (`?`) — всегда. Никакого string formatting.

## Vector Search (Face) — FR-11

```python
# <10k записей: numpy cosine similarity
embeddings = np.stack([np.frombuffer(r, np.float32) for r in rows])
scores = embeddings @ query  # dot product = cosine для L2-норм. векторов

# ≥10k записей: FAISS IndexFlatIP
index = faiss.IndexFlatIP(128)
index.add(embeddings)
scores, idx = index.search(query.reshape(1, -1), top_k)
```

Порог сходства: `≥0.6`. Lazy-init FAISS при первом запросе.

## REST API (6 эндпоинтов)

```
GET  /api/v1/health                      → {"status": "ok"}
GET  /api/v1/detections?limit=100        → {"ok":true,"items":[...]}
GET  /api/v1/patterns                    → {"ok":true,"items":[...]}
POST /api/v1/patterns                    → {"ok":true}
DEL  /api/v1/patterns/{pattern}          → {"ok":true,"existed":bool}
GET  /api/v1/search/face?face_hash=...   → {"ok":true,"items":[...]}
```

**Поля detections:** id, timestamp, lp_number, car_make, car_type, car_color, pattern, face_hash
**Формат timestamp:** `DD/MM/YYYY, HH:MM:SS AM/PM` (хранится как Unix timestamp в секундах)
**pattern:** совпадение с паттерном из таблицы patterns, `null` если нет
**face_hash:** hex-строка xxHash3-128, `null` если лицо не найдено

### Валидация и ошибки

- 400: невалидные параметры (пустой паттерн, неверный face_hash)
- 404: ресурс не найден
- 500: internal error с JSON-телом
- CORS middleware

## Pattern Matching

- Таблица `patterns`: substring match по `lp_number`
- При новой детекции: проверить все паттерны → записать совпадение в поле `pattern`
- `last_seen` обновляется при каждом совпадении

## Graceful Shutdown & Error Recovery

- SIGINT/SIGTERM → завершить pending inference → close ONNX sessions → close DB → exit
- Авто-reconnect к `my.db` при потере соединения
- Retry на ONNX inference failure
- Нет зависших процессов после shutdown

## Structured Logging

JSON формат:
```json
{"timestamp": "ISO 8601", "level": "INFO", "component": "ocr|color|face|api", "track_id": 123, "message": "...", "metrics": {"confidence": 0.95, "inference_ms": 4.2}}
```

## Целевые метрики

| Метрика | Цель | Фактическое | Статус |
|---------|------|-------------|--------|
| LP Character Accuracy | >90% | 99.44% | ✅ |
| LP Full Plate Accuracy | >80% | 98.75% | ✅ |
| Color Accuracy | >75% | 84% | ✅ |
| Face Verification (LFW) | >99% | — | 🔲 |
| Face FAR @ 0.1% | <0.5% | — | 🔲 |
| API latency (p99) | <100 ms | — | ⏳ |

## Задачи из глобального плана

| Фаза | ID | Задача |
|------|----|--------|
| 2 | 2.7 | File Watcher стабилизация (asyncio, cache TTL) |
| 2 | 2.8 | OCR pipeline: LPR → CTC decode → UPDATE data |
| 2 | 2.9 | Color pipeline: bae_model → classify → UPDATE data |
| 3 | 3.1 | MobileFaceNet интеграция: preprocessing, inference, L2-norm |
| 3 | 3.2 | xxHash3-128 hash от face_embedding |
| 3 | 3.4 | Полный цикл: face_images → embedding → hash → upsert → UPDATE data |
| 3 | 3.7 | API endpoint face search: face_hash → cosine → detections |
| 4 | 4.1–4.7 | REST API стабилизация (health, detections, patterns CRUD, face search, validation) |
| 5 | 5.3 | Graceful shutdown |
| 5 | 5.4 | Error recovery (DB reconnect, inference retry) |
| 5 | 5.6 | JSON structured logging |

## Границы ответственности

| Задача | Кто |
|--------|-----|
| ONNX inference (OCR, color, face) | **Python Backend** |
| File Watcher | **Python Backend** |
| REST API endpoints | **Python Backend** |
| Face embedding + hash + upsert | **Python Backend** |
| Pattern matching | **Python Backend** |
| Graceful shutdown Python | **Python Backend** |
| DB schema, миграции, FAISS init | DBA |
| DeepStream pipeline, конфиги | Edge AI |
| C-код, кроппинг, metadata probe | GStreamer Dev |
| Web UI | Frontend |

## Связанные файлы

- `lp_and_color_recognition_prod.py` — основной файл сервиса
- `models/lpr_stn/LPR_STN_PRE_POST.onnx`
- `models/bae_model/bae_model_f3.onnx`
- `models/mobilefacenet/MobileFaceNet.onnx` (~1 MB)
- `docs/system-design/ML_System_Design_Document.md` §4.5, §6.4–6.6, §7.1
- `docs/system-design/global-plan.md` — фазы 2, 3, 4, 5

## Технические требования

- Type hints везде
- Google-style docstrings (см. documentation.mdc)
- `asyncio` для I/O (file watcher, DB, HTTP)
- Prepared statements для SQL
- `uv` для зависимостей

## Правила

- Prepared statements (`?`) — всегда. Никакого string formatting в SQL.
- PRAGMA применять при каждом подключении к `my.db`.
- ONNX Runtime: GPU на Jetson (`CUDAExecutionProvider`), CPU-fallback в dev.
- Face embedding всегда L2-нормализовать перед hash и search.
- Не модифицировать файлы моделей в `models/`.
- Не удалять данные без разрешения пользователя.
