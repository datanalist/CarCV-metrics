# CARS Model Evaluation Project

## О проекте
Бортовая система видеоаналитики для распознавания транспортных средств
в реальном времени на NVIDIA Jetson Orin Nano 8GB.

## Архитектура ML-pipeline
Система состоит из 6 моделей, выполняемых последовательно:

1. **TrafficCamNet** (PGIE) → детекция ТС в кадре
   - ResNet-18 pruned, Caffe → TensorRT FP16
   - Input: 960×544×3 BGR, Output: bbox 4 классов (car, person, bike, sign)
2. **VehicleMakeNet** (SGIE1) → марка автомобиля (20 классов)
3. **VehicleTypeNet** (SGIE2) → тип кузова (6 классов: coupe, largevehicle, sedan, suv, truck, van)
4. **LPDNet** (SGIE3) → детекция номерного знака (bbox)
5. **FaceDetect** (SGIE4) → детекция лиц (bbox)
6. **LPR_STN_PRE_POST.onnx** (Python/ONNX) → OCR номерного знака
   - Input: 188×48×3 RGB, STN + BiLSTM + CTC
   - Алфавит: 0-9, A,B,E,K,M,H,O,P,C,T,Y,X,- (23 символа)
7. **bae_model_f3.onnx** (Python/ONNX) → цвет автомобиля (15 классов)
   - Input: 384×384×3 RGB, нормализация ImageNet
   - mean=[0.43, 0.40, 0.39], std=[0.27, 0.26, 0.26]

## Целевые метрики качества

| Модель | Метрика | Целевое значение |
|--------|---------|------------------|
| TrafficCamNet | Precision | >0.90 |
| TrafficCamNet | Recall | >0.85 |
| TrafficCamNet | mAP@0.5 | >0.89 |
| VehicleMakeNet | Top-1 Accuracy | >0.70 |
| VehicleMakeNet | Top-3 Accuracy | >0.85 |
| VehicleTypeNet | Accuracy | >0.85 |
| LPDNet | Recall | >0.80 |
| LPR_STN | Full Plate Accuracy | >0.85 |
| LPR_STN | Character Accuracy | >0.92 |
| bae_model_f3 | Accuracy | >0.80 |

**Применение:** контроль доступа на объекты, мониторинг парковки, патрульные автомобили, логистика.

## Окружение
- GPU: NVIDIA (проверить через `nvidia-smi`)
- Python: 3.10+, venv в ./venv
- Менеджер зависимостей: uv

## Текущая задача
- Описана в docs/tasks/task.md

## Правила
- Все результаты сохранять в results/ как JSON + CSV
- Графики сохранять в plots/ как PNG
- В конце каждого эксперимента создавать summary в results/SUMMARY.md
- Jupyter notebook с воспроизводимым кодом сохранять в notebooks/
- Логировать все эксперименты
