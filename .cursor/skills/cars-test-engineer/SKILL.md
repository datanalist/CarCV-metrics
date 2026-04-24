---
name: cars-test-engineer
description: Пишет и запускает тесты для CARS — unit (pytest), integration (API + file→DB pipeline), performance (FPS, latency), edge cases, acceptance (CP1-CP6). Use proactively when writing tests, setting up test fixtures, verifying acceptance criteria, or running benchmarks.
---

# CARS Test Engineer — Skill

QA Engineer для embedded AI системы CARS на NVIDIA Jetson Orin Nano 8GB.

## Стек

- **pytest** — unit/integration тесты
- **requests** — API endpoint тесты
- **hypothesis** — property-based тесты (edge cases)
- **tegrastats** — hardware метрики (FPS, power, temperature)
- **time.perf_counter** — latency measurement
- **sqlite3** — DB schema и concurrent write тесты
- **numpy** — face embedding / vector search тесты

## Контекст проекта

- **C-приложение** (~2400 LOC): DeepStream pipeline → metadata → image crops → async SQLite writer
- **Python Service** (~600 LOC): ONNX inference (OCR, color, face embedding) + REST API + file watcher
- **БД:** `my.db` — SQLite WAL, таблицы `data`, `faces`, `patterns`
- **Платформа:** Jetson Orin Nano 8GB, JetPack 6.2, DeepStream 7.1

## Структура тестов

```
tests/
├── unit/
│   ├── test_ocr.py              # LPR inference (CTC decode, alphabet coverage)
│   ├── test_color.py            # Color recognition (15 классов)
│   ├── test_face_embedding.py   # MobileFaceNet pipeline (128-d, L2-norm)
│   ├── test_pattern_matching.py # Substring matching logic
│   ├── test_db_writer.py        # SQLite write/read, schema validation
│   └── test_face_search.py      # Vector search (numpy cosine + FAISS)
├── integration/
│   ├── test_api_endpoints.py    # 6 REST endpoints (health, detections, patterns CRUD, face search)
│   ├── test_file_watcher.py     # File → inference → DB pipeline
│   └── test_db_concurrent.py    # Concurrent C-app + Python writes (WAL)
├── performance/
│   ├── bench_fps.py             # CP1: FPS ≥30 stability
│   ├── bench_latency.py         # CP2: E2E latency <50ms
│   └── bench_throughput.py      # Peak load throughput
├── edge_cases/
│   └── test_edge_cases.py       # Empty images, no face, concurrent writes, invalid input
└── fixtures/
    ├── sample_lp.bmp            # 188×48 номерной знак
    ├── sample_car.bmp           # 384×384 автомобиль
    ├── sample_face.bmp          # Кроп лица
    └── test_db.sqlite           # Тестовая БД (schema: data + faces + patterns)
```

## Acceptance Criteria (CP1-CP6)

Источник: ML_System_Design_Document.md §12.2

| ID  | Критерий                   | Метод                                          | Порог             |
|-----|----------------------------|-------------------------------------------------|-------------------|
| CP1 | FPS стабильно              | tegrastats + fps_current, 1 час                 | ≥30 FPS avg       |
| CP2 | End-to-end latency         | timestamp diff probe в C-app, 1000 кадров       | <50ms p99         |
| CP3 | Detection Precision        | Ручная разметка 1000 кадров vs GT               | >90%              |
| CP4 | LP Accuracy (full plate)   | Ground truth comparison, 500 номеров            | >85%              |
| CP5 | Uptime                     | systemd + health endpoint polling, 72ч          | >99%              |
| CP6 | Power consumption          | INA3221 sensor (tegrastats), 1 час средняя      | <25W              |

## Целевые метрики ML

| Модель              | Метрика             | Порог   |
|---------------------|---------------------|---------|
| TrafficCamNet       | Precision           | >0.90   |
| TrafficCamNet       | Recall              | >0.85   |
| TrafficCamNet       | F1-Score            | >0.87   |
| VMN 105c            | Top-1 Accuracy      | >0.70   |
| VMN 105c            | Top-3 Accuracy      | >0.85   |
| VehicleTypeNet      | Accuracy            | >0.85   |
| LPR                 | Full Plate Accuracy | >0.80   |
| LPR                 | Char Accuracy       | >0.90   |
| Color (bae_model)   | Overall Accuracy    | >0.75   |
| MobileFaceNet       | Verification (LFW)  | >99%    |

## Паттерны тестов

### Unit: OCR (test_ocr.py)

```python
@pytest.fixture
def lpr_session():
    import onnxruntime as ort
    return ort.InferenceSession("models/lpr_stn/LPR_STN_PRE_POST.onnx")

def test_ocr_standard_plate(lpr_session):
    img = np.array(Image.open("tests/fixtures/A123BC77.bmp"))
    result = run_lpr_inference(lpr_session, img)
    assert result == "A123BC77"

def test_ocr_alphabet_coverage(lpr_session):
    ALPHABET = "0123456789ABEKMHOPCТYX-"
    # Тест на synth-изображениях с каждым символом

def test_ocr_inference_time(lpr_session):
    img = np.zeros((48, 188, 3), dtype=np.uint8)
    start = time.perf_counter()
    for _ in range(100):
        run_lpr_inference(lpr_session, img)
    avg_ms = (time.perf_counter() - start) / 100 * 1000
    assert avg_ms < 8.0, f"LPR inference {avg_ms:.1f}ms > 8ms"
```

### Unit: Face Search (test_face_search.py)

```python
def test_numpy_cosine_search():
    embeddings = [np.random.randn(128) for _ in range(1000)]
    embeddings = [e / np.linalg.norm(e) for e in embeddings]
    query = embeddings[42].copy()
    results = search_by_face_numpy(query, embeddings, threshold=0.6)
    assert any(r['rank'] == 0 for r in results)

def test_faiss_search_consistent_with_numpy():
    # >10k embeddings — оба метода дают одинаковые top-5

def test_face_hash_uniqueness():
    import xxhash
    e1 = np.random.randn(128).astype(np.float32)
    e2 = np.random.randn(128).astype(np.float32)
    h1 = xxhash.xxh3_128(e1.tobytes()).digest()
    h2 = xxhash.xxh3_128(e2.tobytes()).digest()
    assert h1 != h2
```

### Integration: API Endpoints (test_api_endpoints.py)

```python
BASE_URL = "http://localhost:8080"

def test_health():
    r = requests.get(f"{BASE_URL}/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_detections_returns_list():
    r = requests.get(f"{BASE_URL}/api/v1/detections?limit=10")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert isinstance(data["items"], list)
    assert len(data["items"]) <= 10

def test_pattern_crud():
    r = requests.post(f"{BASE_URL}/api/v1/patterns", json={"pattern": "TEST123"})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = requests.get(f"{BASE_URL}/api/v1/patterns")
    patterns = [p["pattern"] for p in r.json()["items"]]
    assert "TEST123" in patterns

    r = requests.delete(f"{BASE_URL}/api/v1/patterns/TEST123")
    assert r.json()["existed"] is True

def test_detections_fields():
    r = requests.get(f"{BASE_URL}/api/v1/detections?limit=1")
    items = r.json().get("items", [])
    if items:
        item = items[0]
        assert isinstance(item["timestamp"], int)
        for field in ("lp_number", "car_make", "car_type", "car_color", "pattern", "face_hash"):
            assert field in item

def test_face_search_endpoint():
    r = requests.get(f"{BASE_URL}/api/v1/search/face",
                     params={"face_hash": "nonexistent", "limit": 10})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["items"] == []
```

### Edge Cases (test_edge_cases.py)

```python
def test_empty_lp_image(lpr_session):
    img = np.zeros((48, 188, 3), dtype=np.uint8)
    result = run_lpr_inference(lpr_session, img)
    assert result is not None  # не падает

def test_no_face_detected(db_conn):
    row = db_conn.execute("SELECT face_hash FROM data WHERE id=?", (no_face_id,)).fetchone()
    assert row[0] is None

def test_database_concurrent_write():
    import threading, sqlite3, random, time
    errors = []
    def writer():
        try:
            conn = sqlite3.connect("test.db", timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("INSERT INTO data(timestamp,track_id) VALUES(?,?)",
                        (int(time.time()), random.randint(1, 10000)))
            conn.commit()
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=writer) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, f"Concurrent write errors: {errors}"
```

### Performance: FPS Benchmark (bench_fps.py, CP1)

```python
def test_fps_stability():
    fps_values = []
    for _ in range(300):  # 5 минут
        fps = get_current_fps()
        fps_values.append(fps)
        time.sleep(1)
    avg_fps = sum(fps_values) / len(fps_values)
    p5_fps = sorted(fps_values)[len(fps_values) // 20]
    assert avg_fps >= 30, f"Avg FPS {avg_fps:.1f} < 30"
    assert p5_fps >= 25, f"P5 FPS {p5_fps:.1f} < 25"
```

### 72-часовой Stress Test (CP5)

```bash
#!/bin/bash
START=$(date +%s)
ERRORS=0
while true; do
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START ))
    STATUS=$(curl -sf http://localhost:8080/api/v1/health | jq -r .status)
    if [ "$STATUS" != "ok" ]; then
        ERRORS=$((ERRORS + 1))
        echo "$(date): Health check FAILED (errors: $ERRORS)"
    fi
    if [ $ELAPSED -gt $((72 * 3600)) ]; then
        echo "Test complete. Total errors: $ERRORS"
        exit 0
    fi
    sleep 60
done
```

## Пункты глобального плана

Агент отвечает за следующие пункты `docs/system-design/global-plan.md`:

**Фаза 5 — Тестирование:**
- 5.11: Unit tests (pytest) — OCR, color, face embedding+hash, pattern matching, DB schema, face search. Coverage ≥80%
- 5.12: Integration tests — 6 API endpoints, file→inference→DB, concurrent writes (WAL)
- 5.13: Edge case tests — пустые/повреждённые изображения, отсутствие лица, concurrent DB, неполные lp_number

**Фаза 6 — Acceptance Testing:**
- 6.12: CP1 — FPS ≥30 (tegrastats, 1 час)
- 6.13: CP2 — Latency <50ms (1000 кадров)
- 6.14: CP3 — Detection Precision >90% (manual labeling 1000 frames)
- 6.15: CP4 — LP Accuracy >85% (ground truth, 500 номеров)
- 6.16: CP5 — Uptime >99% (72h stress test)
- 6.17: CP6 — Power <25W (tegrastats, 1 час)
- 6.18: Performance benchmarks report → `docs/experiments/`

## Связанные файлы

| Файл | Описание |
|------|----------|
| `docs/system-design/ML_System_Design_Document.md` §3.3, §12 | Метрики, acceptance criteria |
| `docs/system-design/global-plan.md` §5, §6 | Пункты плана |
| `tests/` | Все тестовые файлы |
| `scripts/` | Утилиты, benchmark-скрипты |
| `models/` | ONNX/TensorRT модели (READ-ONLY) |
| `my.db` | Production SQLite БД |

## Rules

- `tests/` — единственная директория для тестов.
- Fixtures в `tests/fixtures/`.
- Не модифицировать `models/` — только read.
- Тестовая БД изолирована: `tests/fixtures/test_db.sqlite` или `tmp_path`.
- Performance тесты только на Jetson (tegrastats недоступен в dev).
- pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.performance`, `@pytest.mark.slow`.
- Acceptance report → `docs/experiments/acceptance/`.
