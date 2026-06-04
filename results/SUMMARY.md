# CARS — Локальная валидационная кампания (итоговая агрегация)

_Сгенерировано 2026-06-04 (Task 8: финальная агрегация) из `results/<model>/metrics.json` + `EXPERIMENT.md`._

**Окружение:** локальный прогон на **RTX 3090**, **CUDAExecutionProvider**, **onnxruntime 1.24** (1.24.4),
**Python 3.13** (3.13.11), `.venv`. Это **локальная** кампания — НЕ удалённый прогон qudata (его архив —
`results_collected/`, см. «Контекст и ограничения»).

**Итог: 0 PASS / 4 FAIL / 2 DEFERRED / 1 UNDEF** (7 пар модель×датасет).

## Сводная таблица (все 7 пар)

| Модель | Датасет | Ключевые метрики (измерено) | Пороги | Вердикт |
|---|---|---|---|:--:|
| `trafficcamnet` | BDD100K val (10 000 img) | car: P=0.3909, R=0.2584, F1=0.3111 (conf_thr=0.2) | P≥0.90, R≥0.85, F1≥0.87 | ❌ **FAIL** |
| `vehicletypenet` | Stanford Cars (TRAIN, 7577 img; 567 пропущено) | Top-1=0.3549, Top-3=0.6932 | Top-1≥0.85 | ❌ **FAIL** |
| `vehiclemakenet` | VMMRdb (243 519 img) | Top-1=0.4387, Top-3=0.6250 | Top-1≥0.70, Top-3≥0.85 | ❌ **FAIL** |
| `color` (bae_model_f3) | MAD-Cars (2000 dedup img; покрытие 13/15) | Top-1=0.6205, Top-3=0.8395; best_group_min=0.4793; challenging_group_min=0.0 | overall≥0.80, best_group_min≥0.90, challenging_group_min≥0.70 | ❌ **FAIL** |
| `nomeroff_lpd` | AUTO.RIA Numberplate (detection, val) | локально НЕ измерено | P≥0.70, R≥0.80 | ⏸️ **DEFERRED** |
| `nomeroff_ocr` | AUTO.RIA Numberplate (OCR) | локально НЕ измерено | CharAcc≥0.90, PlateAcc≥0.80 | ⏸️ **DEFERRED** |
| `facedetect` (FaceNet) | WIDER FACE val | модель недоступна (ONNX не получен) | AP@0.5≥0.50 (Easy≥0.80/Med≥0.70/Hard≥0.50) | ❓ **UNDEF** |

## Заметки по парам

### Измеренные FAIL (4)

- **`trafficcamnet` × BDD100K → FAIL.** Доменный разрыв: TrafficCamNet обучен на видах с дорожных камер
  (сверху/под углом), BDD100K — бортовая съёмка с уровня дороги. car P=0.3909 / R=0.2584 / F1=0.3111
  при conf_thr=0.2 — ниже порогов (P≥0.90, R≥0.85, F1≥0.87). Выше прежнего суррогата COCO val2017
  (там car F1≈0.0642), но порогов кампании не достигает. Окончательный валидный результат.

- **`vehicletypenet` × Stanford Cars (TRAIN-сплит) → FAIL.** Top-1=0.3549 < 0.85 (Top-3=0.6932 справочно).
  Из 8144 записей отобрано 7577, пропущено 567 (нет ключевого слова типа кузова). Причина — доменный
  разрыв + **шумные суррогатные метки**: Stanford размечен по make/model, тип кузова выводится по
  ключевым словам (`convertible→coupe`, `wagon→sedan` и т.п.); класс `largevehicle` на Stanford
  отсутствует. Лучший класс `truck` 0.6137, худший `sedan` 0.2867.

- **`vehiclemakenet` × VMMRdb → FAIL.** Top-1=0.4387 < 0.70 и Top-3=0.6250 < 0.85 на 243 519 изображениях
  (20 NGC-марок US/EU-рынка, OOD-каталоги отфильтрованы заранее, `skipped_ood=0`). При этом результат
  **существенно выше** прежнего RU-суррогата mad-cars (Top-1≈0.083 на удалённом прогоне): VMMRdb —
  US-домен, ближе к US/EU-обучению модели, поэтому массовые марки узнаются заметно лучше (bmw 0.705,
  mercedes 0.686), но на long-tail (kia 0.205, chrysler 0.195, subaru 0.233) до порога не дотягивает.

- **`color` (bae_model_f3) × MAD-Cars → FAIL.** Все три порога не достигнуты: overall Top-1=0.6205 < 0.80;
  best_group_min=0.4793 < 0.90 (минимум держит `blue`); challenging_group_min=0.0 < 0.70 (минимум держит
  `gold`=0.0). Top-3=0.8395 (справочно). 2000 изображений (dedup по `car_id`), покрытие 13/15 — отсутствуют
  `tan` (нет hex-маппинга) и `pink` (hex `ffc0cb` не встретился в локальном sample). **Кавеат:** маппинг
  индекс→имя класса (COLOR_CLASSES, алфавитный) непроверяем без оригинального файла меток модели — вердикт
  следует читать как **осторожный (cautious)**, не как коллапс (black 0.897 / white 0.794 / red 0.730
  узнаются уверенно). Подробности и кавеаты — `results/color/EXPERIMENT.md`.

### Отложенные DEFERRED (2)

- **`nomeroff_lpd` × AUTO.RIA detection → DEFERRED.** Прогон не выполнен. Блокер: `import nomeroff_net`
  падает — `ModuleNotFoundError: No module named 'modelhub_client'` (PyPI-имя `modelhub-client`
  отсутствует на PyPI; установка из произвольного git — вне рамок плана). Локальных чисел нет.
  Прежний **удалённый прогон qudata** (ssh9.qudata.ai): P=0.9056, R=0.9221, F1=0.9138 — _PASS на
  удалённом прогоне, локально НЕ воспроизведено_ (приводится только как удалённый ориентир).

- **`nomeroff_ocr` × AUTO.RIA OCR → DEFERRED.** Тот же блокер `modelhub_client`. Локальных чисел нет.
  Прежний **удалённый прогон qudata**: CharAcc=0.9995, PlateAcc=0.9978 — _PASS на удалённом прогоне,
  локально НЕ воспроизведено_.

### Неопределённый UNDEF (1)

- **`facedetect` (FaceNet) × WIDER FACE → UNDEF.** Не FAIL — измерение не проводилось, т.к. модель не
  удалось загрузить: FaceNet доступен локально только как зашифрованный `.etlt`, `tao_converter` не
  установлен, NGC deployable-ONNX URL → HTTP 404. При этом **код готов и данные на месте**:
  `parse_wider_gt` + `eval_facedetect` реализованы и протестированы (suite 13 passed), WIDER FACE val
  (GT + images, включая automotive-категории `14--Traffic`, `5--Car_Accident`, `59--people--driving--car`)
  присутствует. Прогон запускается без изменений кода, как только появится deployable FaceNet ONNX.

## Контекст и ограничения

- **Корневая причина FAIL — доменный разрыв.** Модели NGC TAO (TrafficCamNet, VehicleTypeNet,
  VehicleMakeNet, FaceNet) обучены на US/EU-данных и слабы на RU-рынке / разнородных датасетах:
  TrafficCamNet (дорожные камеры → бортовая съёмка), VehicleTypeNet/VehicleMakeNet (US-марки и
  суррогатные/шумные метки), Color (трудные/слитые цвета + непроверяемый маппинг меток). FAIL —
  валидные окончательные результаты, тюнинг не применялся.
- **Продакшн-стек RU (`nomeroff_lpd` / `nomeroff_ocr`) локально не воспроизведён** из-за упаковки:
  `nomeroff-net` 4.0.1 ставится, но не импортируется (отсутствует `modelhub-client` на PyPI). Это
  именно RU-ориентированная замена US-обученных LPDNet/LPRNet.
- **Прежние удалённые результаты** (включая nomeroff PASS) хранятся в `results_collected/` (хосты
  `qudata2`, `ssh1.qudata.ai`, `ssh9.qudata.ai`). В этой локальной кампании удалённые числа для
  nomeroff приводятся **исключительно как ориентир** и помечены «локально НЕ воспроизведено».
- Воспроизводимый ноутбук: `notebooks/local_validation_campaign.ipynb` (читает предвычисленные
  `results/<model>/metrics.json`, не перезапускает модели).
