# Задача: Валидация TrafficCamNet (Primary Detector)

## Что сделать
Написать и запустить evaluation pipeline для модели TrafficCamNet.

## Шаги выполнения
1. Проверь доступность GPU: `nvidia-smi`
2. Установи зависимости: `pip install numpy opencv-python-headless pycocotools`
3. Если модель в формате Caffe — конвертируй в ONNX через `caffe2onnx` или используй TensorRT напрямую через `trtexec`
4. Если уже есть .engine файл — используй TensorRT Python API
5. Напиши скрипт `eval_trafficcamnet.py`:
   - Загрузи модель
   - Пройди по всем изображениям из датасета
   - Для каждого: resize до 960×544, BGR, scale_factor=1.0
   - Запусти inference, собери предсказания
   - Отфильтруй только класс "car" (class_id=0)
   - Вычисли Precision, Recall, F1, mAP@0.5 через pycocotools
6. Замерь latency (1000 прогонов, warm-up 100)
7. Построй:
   - Precision-Recall кривую
   - Распределение confidence scores
   - Примеры TP, FP, FN (сохрани изображения)
8. Сохрани результаты в `validation/results/trafficcamnet/`

## Данные
- Данные валидации: скачиваются (https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)
- Модель: скачивается с официального репозитория NVIDIA
- Путь к изображениям: `/data/validation/images/` (1920×1080 JPEG)
- Путь к аннотациям: `/data/validation/annotations.json` (COCO format)

## Формат результатов
```json
{
  "model": "TrafficCamNet",
  "metrics": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "mAP_50": 0.0
  },
  "latency_ms": {
    "mean": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "p99": 0.0
  },
  "dataset": {
    "total_images": 0,
    "total_gt_boxes": 0,
    "total_predictions": 0
  },
  "target_met": {
    "precision": false,
    "recall": false,
    "mAP_50": false
  }
}
