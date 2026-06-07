# Models URLs

Task | model name | URL | version | Local path | Spec
Detection | TrafficCamNet | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet?version=pruned_v1.0.3 | pruned_quantized_v2.0.1 | /home/mk/CarCV/models/trafficcamnet_pruned_onnx_v1.0.4 | [trafficcamnet.md](docs/about_models/trafficcamnet.md)
LP Detection | nomeroff_lpd | https://nomeroff.net.ua/models/object_detection/yolov11x-keypoints-2026-01-21.pt | None | /home/mk/CarCV/models/nomeroff_net/object_detection/yolov26x-keypoints-2026-01-21.pt| [nomeroff_lpd.md](docs/about_models/nomeroff_lpd.md)
Face Detection | FaceDetect | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet?version=pruned_quantized_v2.0.1 | pruned_quantized_v2.0.1 |  | [facedetect.md](docs/about_models/facedetect.md)
OCR | nomeroff_ocr | https://nomeroff.net.ua/models/ocr/ru/torch/model_v3.3/resnet18/anpr_ocr_ru_2023_02_01_resnet18.ckpt | None |  | [nomeroff_ocr.md](docs/about_models/nomeroff_ocr.md)
Color | bae_model_f3 |  |  | /home/mk/CarCV/models/bae_model_f3.onnx | [bae_model_f3.md](docs/about_models/bae_model_f3.md)
Make | VehicleMakeNet | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehiclemakenet?version=pruned_onnx_v1.1.0 | pruned_v1.0.2 |  | [vehiclemakenet.md](docs/about_models/vehiclemakenet.md)
Type | VehicleTypeNet | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehicletypenet?version=pruned_v1.0.2 | pruned_v1.0.2 |  | [vehicletypenet.md](docs/about_models/vehicletypenet.md)
Face Detector | FaceDetect | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet?version=pruned_quantized_v2.0.1 | pruned_quantized_v2.0.1 | home/mk/Загрузки/CarCVModels/facenet_pruned_quantized_v2.0.1 | [facedetect.md](docs/about_models/facedetect.md)
Face embedding | None | None | Trainable |  | [face_embedding.md](docs/about_models/face_embedding.md)

## Исследование открытых моделей-аналогов

Для каждой модели стека подобраны открытые предобученные **аналоги**, пригодные для пайплайна CARS на Jetson Orin Nano 8GB (ONNX→TensorRT/DeepStream либо Python-сервис). Лицензии перепроверены **адверсариально под коммерческий продукт** (AGPL/GPL/NC/EULA — блокеры; Apache-2.0/MIT/BSD — чистые), оценены edge-пригодность и домен-fit к бортовому POV (day/night/IR, RU/UA). См. также исследование [датасетов-аналогов](docs/dataset_research/00_SUMMARY.md).

- **Сводка + большая сравнительная таблица:** [docs/model_research/00_SUMMARY.md](docs/model_research/00_SUMMARY.md)
- Detection (TrafficCamNet): [01_detection_trafficcamnet.md](docs/model_research/01_detection_trafficcamnet.md) — DashCamNet, D-FINE, RF-DETR, PP-YOLOE+, YOLOX, RT-DETRv2, NanoDet…
- LP Detection (nomeroff_lpd): [02_lpd_nomeroff.md](docs/model_research/02_lpd_nomeroff.md) — open-image-models, we0091234, PaddleOCR det, WPOD-NET, LPDNet…
- Face Detection (FaceDetect): [03_facedetect_facenet.md](docs/model_research/03_facedetect_facenet.md) — YuNet, FaceDetectIR, ULFG-1MB, BlazeFace, SCRFD/RetinaFace…
- OCR (nomeroff_ocr): [04_ocr_nomeroff.md](docs/model_research/04_ocr_nomeroff.md) — PaddleOCR PP-OCRv5 (cyrillic), fast-plate-ocr, EasyOCR, PARSeq, LPRNet…
- Color (bae_model_f3): [05_color_bae.md](docs/model_research/05_color_bae.md) — timm EfficientNet-B3, OpenVINO barrier-0039/0042, PaddleClas PULC…
- Make (VehicleMakeNet): [06_make_vehiclemakenet.md](docs/model_research/06_make_vehiclemakenet.md) — timm transfer, Jordo23, dima806, Spectrico MMR…
- Type (VehicleTypeNet): [07_type_vehicletypenet.md](docs/model_research/07_type_vehicletypenet.md) — OpenVINO barrier-0042, PaddleClas PULC, NVIDIA VehicleTypeNet…
- Face embedding (None/Trainable): [08_faceembed.md](docs/model_research/08_faceembed.md) — SFace, dlib, GhostFaceNets, AdaFace, InsightFace…

> **Сквозной вывод:** готового коммерчески-чистого drop-in аналога нет почти ни по одной задаче. Лучший domain-fit (DashCamNet/FaceDetectIR) — под NVIDIA EULA; чистые Apache-2.0-веса обучены на web/COCO/WIDER (domain gap, нет RU/UA); по 5 из 8 задач (Color, Make, Type, LP-углы, Face embedding) рекомендуемый путь — **дообучение чистого backbone на собственном бортовом RU/UA-датасете**.
