# TrafficCamNet (UNPRUNED) — Evaluation на BDD100K

Этот документ описывает запуск evaluation для **unpruned** версии модели
TrafficCamNet на датасете BDD100K. Pipeline переиспользует существующие
утилиты пакета `utils/` и отличается от pruned-версии лишь способом загрузки
модели: входной формат — `.tlt` (TAO Toolkit checkpoint), который при первом
запуске экспортируется в ONNX и кэшируется.

---

## 1. Предварительные требования

| Артефакт | Путь по умолчанию | Источник |
|---|---|---|
| `.tlt` checkpoint | `models/baseline/resnet18_trafficcamnet.tlt` | NGC `nvidia/tao/trafficcamnet:unpruned_v1.0` |
| ONNX (кэш) | `models/baseline/resnet18_trafficcamnet_unprunned.onnx` | автогенерируется |
| BDD100K val labels | `data.local_ann_json` (см. YAML) | BDD100K release |
| BDD100K val images | `data.local_images_dir` | BDD100K release / Kaggle |

Для экспорта `.tlt → .onnx` нужен один из вариантов **TAO Toolkit CLI**
(пробуются по порядку — берётся первый, найденный в `PATH`):

1. `tao model detectnet_v2 export` (TAO 5.x)
2. `tao detectnet_v2 export` (TAO 4.x)
3. `tlt-export detectnet_v2` (legacy TLT 3.x)

Установка (один из вариантов):

```bash
pip install nvidia-tao        # TAO Launcher 5.x
# или используйте Docker image: nvcr.io/nvidia/tao/tao-toolkit:5.x.x-tf1.15.5
```

Ключ дешифрования NGC public-модели TrafficCamNet: `tlt_encode`
(переопределяется через `model.model_key` в YAML или env-переменные
`TLT_MODEL_KEY` / `NGC_MODEL_KEY`).

---

## 2. Конфигурация

| Файл | Назначение |
|---|---|
| `configs/experiment/trafficcamnet_unprunned_eval.yaml` | Полный прогон BDD100K val (~10 000 кадров), `batch_size: auto` |
| `configs/experiment/trafficcamnet_unprunned_eval_test.yaml` | Smoke-test на 10 кадрах, `batch_size: 4` |

Ключевые поля секции `model` (отличия от pruned-конфига):

```yaml
model:
  format: tlt                                            # tlt | onnx
  tlt_path: models/baseline/resnet18_trafficcamnet.tlt
  onnx_cache_path: models/baseline/resnet18_trafficcamnet_unprunned.onnx
  model_key: tlt_encode                                  # NGC public key
  force_reexport: false                                  # true → переэкспорт ONNX
  input_w: 960
  input_h: 544
  input_c: 3
  ...
```

Остальные секции (`data`, `evaluation`, `batch`, `artifacts`) идентичны
pruned-конфигу — поэтому количественные результаты сравнимы напрямую.

---

## 3. Запуск

### Smoke-test (10 кадров)

```bash
cd /home/user/webapp && python scripts/eval_trafficcamnet_unprunned_gpu.py \
    --config configs/experiment/trafficcamnet_unprunned_eval_test.yaml
```

### Полный прогон BDD100K val

```bash
cd /home/user/webapp && python scripts/eval_trafficcamnet_unprunned_gpu.py \
    --config configs/experiment/trafficcamnet_unprunned_eval.yaml
```

При первом запуске будет вызван TAO export `.tlt → .onnx`; результат
кэшируется в `model.onnx_cache_path`. Для повторной генерации ONNX (например,
после обновления `.tlt`) выставьте `model.force_reexport: true`.

### Ручной экспорт (если TAO CLI отсутствует на runtime-машине)

```bash
tao model detectnet_v2 export \
    -m models/baseline/resnet18_trafficcamnet.tlt \
    -k tlt_encode \
    --export_format onnx \
    -o models/baseline/resnet18_trafficcamnet_unprunned.onnx \
    --input_dims 3,544,960
```

Затем перезапустите скрипт — он подхватит готовый ONNX-кэш.

---

## 4. Что добавлено в репозиторий

| Файл | Что делает |
|---|---|
| `utils/model_loader.py` → `class TLTModelLoader(TrafficCamNetLoader)` | Прозрачно экспортирует `.tlt` → `.onnx` (с кэшем) и переиспользует ONNX-инференс |
| `configs/experiment/trafficcamnet_unprunned_eval.yaml` | Полный конфиг для unpruned-модели |
| `configs/experiment/trafficcamnet_unprunned_eval_test.yaml` | Smoke-test конфиг (10 кадров) |
| `scripts/eval_trafficcamnet_unprunned_gpu.py` | Точка входа: загрузка `.tlt`, ONNX-экспорт, batch-evaluation |

`utils/batch_data_loader.py`, `utils/batch_inference.py`, `utils/postprocess.py`,
`utils/metrics.py`, `utils/adaptive_batch_size.py`, `utils/gpu_memory.py`
переиспользуются без изменений.

---

## 5. Артефакты результатов

После запуска в `results/trafficcamnet_unprunned_eval/`:

| Файл | Содержимое |
|---|---|
| `results.json` | `model="TrafficCamNet (unpruned)"`, `model_source` (tlt/onnx пути), `config`, `metrics`, `latency_ms`, `dataset`, `target_met`, `confidence_stats` |
| `results.csv` | Плоский список метрик (`metric, value`) |

Структура полностью совместима со скриптом `scripts/visualize_results.py`,
поэтому результаты pruned и unpruned можно сравнивать в одном отчёте.

---

## 6. Чек-лист перед запуском

- [ ] Существует `models/baseline/resnet18_trafficcamnet.tlt`
- [ ] В `PATH` доступен `tao` / `tlt-export` (или ONNX уже экспортирован вручную)
- [ ] Корректный ключ: `tlt_encode` (или `TLT_MODEL_KEY` env)
- [ ] Существуют `data.local_ann_json` и `data.local_images_dir`
- [ ] `nvidia-smi` показывает GPU; `uv sync` выполнен
- [ ] Сначала прогнан smoke-test, затем полный прогон
