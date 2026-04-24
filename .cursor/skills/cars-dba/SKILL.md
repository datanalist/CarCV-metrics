---
name: cars-dba
description: SQLite schema design, PRAGMA tuning, FAISS vector search, face embedding storage, concurrent access, migrations, and data lifecycle for CARS project. Use proactively when working with database schema, SQLite queries, FAISS vector search, face embedding storage, or data migration scripts.
---

# CARS Database Administrator

Database engineer: SQLite (embedded, high-throughput write) + векторный поиск (FAISS/numpy) для face embeddings в проекте CARS.

## Контекст

- **СУБД:** SQLite 3.37.2
- **БД файл:** `my.db` — единый файл; C-app и Python-сервис используют один файл с WAL
- **Нагрузка:** ~80 записей/мин при 10 авто/мин (8ч/день = 4,800 записей)
- **Face embeddings:** 128-d float32 vectors (512 байт), xxHash3-128 (16 байт) как PK

## Схема БД

### Таблица `data`

```sql
CREATE TABLE data (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   INTEGER NOT NULL,     -- Unix timestamp (сек); вывод: DD/MM/YYYY HH:MM:SS AM/PM
    track_id    INTEGER NOT NULL,     -- ID от NvMultiObjectTracker
    cx          REAL,                 -- Центр bbox X (0.0–1.0)
    cy          REAL,                 -- Центр bbox Y (0.0–1.0)
    w           REAL,                 -- Ширина bbox (0.0–1.0)
    h           REAL,                 -- Высота bbox (0.0–1.0)
    car_color   TEXT,                 -- 15 классов: beige/black/blue/brown/gold/green/grey/orange/pink/purple/red/silver/tan/white/yellow
    car_type    TEXT,                 -- 6 классов: coupe/largevehicle/sedan/suv/truck/van
    car_make    TEXT,                 -- 105 классов (vmn_vmmrdb_ymad_105c)
    face_hash   BLOB,                -- FK → faces.hash; 16 байт xxHash3-128; NULL если лицо не найдено
    lp_number   TEXT,                -- OCR результат (RU алфавит, 23 символа)
    status      INTEGER DEFAULT 0,   -- 0=new (C-app), 1=processed (Python)
    FOREIGN KEY (face_hash) REFERENCES faces(hash)
);

CREATE INDEX idx_timestamp ON data(timestamp);
CREATE INDEX idx_track_id  ON data(track_id);
CREATE INDEX idx_status    ON data(status);
CREATE INDEX idx_lp_number ON data(lp_number);
CREATE INDEX idx_face_hash ON data(face_hash);
```

### Таблица `faces`

```sql
CREATE TABLE faces (
    hash           BLOB PRIMARY KEY,  -- xxHash3-128 от face_embedding (16 байт BLOB)
    face_count     INTEGER DEFAULT 0, -- Сколько раз лицо встречалось
    face_crop_path TEXT,              -- face_images/{hash_hex}.bmp
    face_embedding BLOB               -- 128 × float32 = 512 байт, L2-нормализован
);
```

### Таблица `patterns`

```sql
CREATE TABLE patterns (
    pattern   TEXT PRIMARY KEY,    -- Подстрока для поиска в lp_number
    last_seen INTEGER              -- Unix timestamp последнего совпадения; NULL если не было
);
```

## PRAGMA оптимизации (обязательно)

```sql
PRAGMA journal_mode=WAL;       -- Write-Ahead Logging (конкурентные чтения)
PRAGMA synchronous=NORMAL;     -- Баланс скорость/надёжность
PRAGMA cache_size=10000;       -- ~40MB page cache
PRAGMA temp_store=MEMORY;      -- Temp структуры в RAM
PRAGMA foreign_keys=ON;        -- Контроль FK face_hash → faces.hash
```

## Паттерн конкурентного доступа

```
C-App (deepstream-vehicle-analyzer)
  → INSERT INTO data (..., status=0)
  → my.db (WAL mode)

Python Service (lp_and_color_recognition_prod.py)
  → SELECT FROM data WHERE status=0
  → ONNX inference (OCR, color, face embedding)
  → UPDATE data SET lp_number, car_color, face_hash, status=1
  → INSERT OR IGNORE INTO faces; UPDATE faces SET face_count+1
  → my.db (тот же файл, WAL)

REST API (тот же Python-процесс)
  → SELECT FROM data/faces/patterns
  → my.db (read-only через WAL)
```

WAL mode: одновременные чтения + один writer без блокировок. C и Python безопасно работают с одним файлом.

## Face Embedding: Логика вставки (FR-12)

```python
embedding = mobilefacenet.run(face_crop)        # 128-d float32
embedding /= np.linalg.norm(embedding)           # L2-нормализация
raw_bytes = embedding.tobytes()                  # 512 байт
face_hash = xxhash.xxh3_128(raw_bytes).digest()  # 16 байт BLOB

conn.execute("""
    INSERT INTO faces(hash, face_count, face_crop_path, face_embedding)
    VALUES (?, 1, ?, ?)
    ON CONFLICT(hash) DO UPDATE SET face_count = face_count + 1
""", (face_hash, crop_path, raw_bytes))

conn.execute("UPDATE data SET face_hash=?, status=1 WHERE track_id=? AND status=0",
             (face_hash, track_id))
```

## Vector Search — Face (FR-11)

```python
def search_by_face(query_embedding: np.ndarray, threshold: float = 0.6,
                   top_k: int = 100) -> list[dict]:
    query = query_embedding / np.linalg.norm(query_embedding)
    rows = conn.execute("SELECT hash, face_embedding FROM faces").fetchall()

    if len(rows) < 10_000:
        embeddings = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
        scores = embeddings @ query  # cosine (уже L2-норм.)
        top_idx = np.argsort(scores)[::-1][:top_k]
        matches = [(rows[i][0], float(scores[i])) for i in top_idx if scores[i] >= threshold]
    else:
        index = faiss.IndexFlatIP(128)
        embeddings = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
        index.add(embeddings)
        scores, idx = index.search(query.reshape(1, -1), top_k)
        matches = [(rows[i][0], float(scores[0][j]))
                   for j, i in enumerate(idx[0]) if scores[0][j] >= threshold]

    result = []
    for face_hash, score in matches:
        rows_data = conn.execute(
            "SELECT * FROM data WHERE face_hash=?", (face_hash,)).fetchall()
        result.extend(rows_data)
    return result
```

## Maintenance Queries

```sql
-- Очистка старше 7 дней
DELETE FROM data  WHERE timestamp < strftime('%s','now') - 604800;
DELETE FROM faces WHERE hash NOT IN (SELECT DISTINCT face_hash FROM data WHERE face_hash IS NOT NULL);

-- VACUUM после очистки
VACUUM;

-- Статистика
SELECT COUNT(*) as total, AVG(CASE WHEN lp_number IS NOT NULL THEN 1.0 ELSE 0.0 END) as lp_rate
FROM data WHERE timestamp > strftime('%s','now') - 86400;
```

## Расчёт хранилища

```
Средняя интенсивность: 10 авто/мин × 60 мин × 8ч = 4,800 детекций/день
На детекцию: ~486 KB (БД + кропы)
За день: ~2.3 GB | За неделю: ~16 GB | За месяц: ~70 GB
Рекомендация: NVMe SSD 128-256 GB с авторотацией 7 дней
```

## Задачи из глобального плана

| Фаза | ID | Задача |
|------|----|--------|
| 0 | 0.6 | Инициализация `my.db`: таблицы + индексы + PRAGMA |
| 2 | 2.10 | Проверка concurrent writes: C-app + Python через WAL без конфликтов |
| 3 | 3.3 | Face upsert логика: INSERT OR UPDATE по hash |
| 3 | 3.5 | Векторный поиск: numpy cosine similarity для <10k |
| 3 | 3.6 | FAISS IndexFlatIP для ≥10k: lazy-init, rebuild |

## Связанные файлы

- `docs/system-design/ML_System_Design_Document.md` §4.5, §5.3
- `docs/system-design/global-plan.md` — фазы 0, 2, 3
- `deepstream-vehicle-analyzer.c` — C-app, INSERT INTO data
- `lp_and_color_recognition_prod.py` — Python, UPDATE/INSERT faces

## Правила

- Все запросы используют prepared statements (параметризация `?`).
- Схема БД — source of truth из этого скилла. Не модифицировать без согласования.
- PRAGMA применять при каждом подключении к `my.db`.
- Face upsert — всегда через `ON CONFLICT(hash) DO UPDATE`.
- Векторный поиск: numpy для <10k записей, FAISS для ≥10k.
- Не удалять данные из `my.db` без разрешения пользователя.
- Data rotation — 7 дней, с `VACUUM` после `DELETE`.
