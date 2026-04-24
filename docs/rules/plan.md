# План

## Цель
Оценить качество детекции TrafficCamNet на BDD100K (val, cars) минимум по 3 метрикам, без DeepStream.

## Шаги
- [x] Подготовить инференс код (ONNXRuntime + CUDA, декодирование, NMS)
- [x] Загрузить BDD100K аннотации и сопоставить GT по car
- [x] Реализовать метрики (mAP@0.5, mAP@0.5:0.95, precision/recall/F1, mean IoU)
- [x] Обновить notebook под BDD100K и результаты
- [x] Сохранить результаты в results/baseline/trafficcamnet_bdd100k
- [x] Создать notebook с 20 примерами детекции
- [x] Описать эксперимент в docs/experiments

## Данные и пути
- Датасет: `/home/mk/Загрузки/DATASETS/bdd100k`
- Модель: `models/trafficcamnet_pruned_onnx_v1.0.4`

## Результаты
- Notebook: `notebooks/trafficcamnet_bdd100k_evaluation.ipynb`
- Notebook (примеры детекции): `notebooks/trafficcamnet_bdd100k_detection_examples.ipynb`
- Результаты: `results/baseline/trafficcamnet_bdd100k/`
- Обработано изображений: 9,879 (BDD100K val)
- mAP@0.5: 11.39%, mAP@0.5:0.95: 3.52%
- Precision: 57.37%, Recall: 17.70%, F1: 27.06%
- Mean IoU: 69.64%, FPS: 31.10
