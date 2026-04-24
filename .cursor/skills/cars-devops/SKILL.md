---
name: cars-devops
description: "Управляет инфраструктурой CARS на Jetson: systemd сервисы, Prometheus/Grafana мониторинг, logrotate, cron ротация данных, backup, Wi-Fi AP, power management. Use proactively when working with systemd unit files, deployment scripts, monitoring configs (Prometheus/Grafana/AlertManager), cron jobs, data rotation, logrotate, backup, Jetson setup, or power management procedures."
---

# CARS DevOps / MLOps

DevOps/MLOps для edge AI на embedded Linux (JetPack 6.2 / Ubuntu 22.04 ARM64).

## Контекст проекта

| Параметр | Значение |
|----------|----------|
| Платформа | Jetson Orin Nano 8GB, JetPack 6.2, Ubuntu 22.04 ARM64 |
| GPU | 1024 CUDA cores, Ampere, 40 TOPS INT8 |
| Хранилище | NVMe SSD 128–256 GB (рекомендуется 256 GB) |
| Питание | 12V авто → DC-DC 12V→5V/4A, max 25W |
| Сервисы | 2 systemd units: C-app (DeepStream) + Python service |
| БД | `my.db` — единый SQLite с WAL |
| Рабочий каталог | `/opt/apps/vehicle-analyzer` |
| User | `jetson` |
| Данные/день | ~2.3 GB (4800 детекций × ~486 KB) |
| Retention | 7 дней (images + DB records) |

## Пункты глобального плана

Агент отвечает за следующие пункты из `docs/system-design/global-plan.md`:

| Пункт | Описание | Фаза |
|-------|----------|------|
| 0.1 | Верификация JetPack 6.2 и зависимостей | 0 |
| 0.2 | Power mode (`nvpmodel`, `jetson_clocks`, `tegrastats`) | 0 |
| 0.5 | Создание директорий `lp_images/`, `car_images/`, `face_images/` | 0 |
| 5.7 | Logrotate для логов обоих сервисов | 5 |
| 5.8 | Prometheus exporter (:9090/metrics) | 5 |
| 5.9 | Grafana dashboard | 5 |
| 5.10 | AlertManager rules (7 пороговых метрик) | 5 |
| 6.1 | Systemd `deepstream-vehicle.service` | 6 |
| 6.2 | Systemd `lp-recognition.service` | 6 |
| 6.3 | Wi-Fi AP (hostapd + dnsmasq) | 6 |
| 6.4 | Cron cleanup: images + DB rotation | 6 |
| 6.5 | Cron backup `my.db` | 6 |

## Systemd сервисы

### deepstream-vehicle.service

```ini
[Unit]
Description=DeepStream Vehicle Analyzer
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/opt/apps/vehicle-analyzer
ExecStart=/opt/apps/vehicle-analyzer/deepstream-vehicle-analyzer \
    rtsp rtsp://camera/stream my.db false
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### lp-recognition.service

```ini
[Unit]
Description=LP & Color Recognition + REST API
After=deepstream-vehicle.service
Requires=deepstream-vehicle.service

[Service]
Type=simple
User=jetson
WorkingDirectory=/opt/apps/vehicle-analyzer
ExecStart=/usr/bin/python3 lp_and_color_recognition_prod.py \
    my.db --api-host 0.0.0.0 --api-port 8080
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Размещение: `/etc/systemd/system/`. Включение: `systemctl enable --now <name>`.

## Prometheus метрики

### Application metrics

| Метрика | Тип | Описание |
|---------|-----|----------|
| `fps_current` | gauge | Текущий FPS |
| `latency_ms` | histogram | E2E задержка |
| `detections_total` | counter | Всего детекций |
| `recognition_accuracy` | gauge | Точность OCR (rolling) |
| `errors_total` | counter | Ошибки по компоненту |

### System metrics (tegrastats / node_exporter)

| Метрика | Тип |
|---------|-----|
| `gpu_utilization_percent` | gauge |
| `gpu_memory_used_bytes` | gauge |
| `cpu_utilization_percent` | gauge |
| `ram_used_bytes` | gauge |
| `temperature_celsius` | gauge |
| `power_watts` | gauge |
| `disk_used_bytes` | gauge |

Exporter endpoint: `:9090/metrics`.

### Пороги алертов

| Метрика | Warning | Critical |
|---------|---------|----------|
| FPS | <25 | <20 |
| Latency | >60 ms | >100 ms |
| GPU Util | >90% | >95% |
| RAM | >6 GB | >7 GB |
| Disk Free | <20 GB | <5 GB |
| Temperature | >65°C | >75°C |
| Errors/min | >10 | >50 |

## Data rotation (cron)

### cleanup.sh

```bash
#!/bin/bash
# /opt/apps/cleanup.sh

find /opt/apps/vehicle-analyzer/lp_images  -mtime +7 -delete
find /opt/apps/vehicle-analyzer/car_images -mtime +7 -delete
find /opt/apps/vehicle-analyzer/face_images -mtime +7 -delete

sqlite3 /opt/apps/vehicle-analyzer/my.db \
    "DELETE FROM data WHERE timestamp < strftime('%s', 'now') - 604800;"
sqlite3 /opt/apps/vehicle-analyzer/my.db \
    "DELETE FROM faces WHERE hash NOT IN (SELECT DISTINCT face_hash FROM data WHERE face_hash IS NOT NULL);"
sqlite3 /opt/apps/vehicle-analyzer/my.db "VACUUM;"

cp /opt/apps/vehicle-analyzer/my.db \
   /opt/backup/my_$(date +%Y%m%d).db
```

Crontab: `0 3 * * * /opt/apps/cleanup.sh >> /var/log/cars-cleanup.log 2>&1`

## Logrotate

Файл `/etc/logrotate.d/cars`:

```
/var/log/deepstream-vehicle-analyzer.log
/var/log/lp_color_service.log
{
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    maxsize 100M
}
```

## Structured logging

JSON формат для всех компонентов:

```json
{
  "timestamp": "2026-01-12T10:30:00Z",
  "level": "INFO",
  "component": "detector",
  "track_id": 12345,
  "message": "Vehicle detected",
  "metrics": {"confidence": 0.94, "inference_ms": 8.5}
}
```

## Начальная настройка Jetson

```bash
# JetPack 6.2 через NVIDIA SDK Manager (host x86)

# DeepStream 7.1
sudo apt update && sudo apt install -y deepstream-7.1

# Python зависимости
sudo apt install -y python3-pip python3-numpy python3-opencv
pip3 install onnxruntime-gpu==1.16.0 xxhash faiss-gpu

# Сборка C-приложения
cd /opt/apps/vehicle-analyzer
make CUDA_VER=12.6 clean all

# Директории
mkdir -p lp_images car_images face_images

# Power mode
sudo nvpmodel -m 0      # MAX 25W
sudo jetson_clocks       # Фиксированные частоты
```

Верификация зависимостей: DeepStream 7.1, CUDA 12.6, cuDNN 9.0, TensorRT 10.3, ONNX Runtime 1.16, GStreamer 1.0, OpenCV 4.8, SQLite 3.37.

## Wi-Fi Access Point

```bash
sudo apt install -y hostapd dnsmasq
# Конфиг hostapd: SSID=CARS, password-protected
# Web UI: http://192.168.4.1:8080
```

## Storage budget

| Период | Объём |
|--------|-------|
| 1 день | ~2.3 GB |
| 1 неделя | ~16 GB |
| 1 месяц | ~70 GB |
| SSD 256 GB | ротация: хранить 3 недели |

## Связанные файлы

- `configs/` — systemd, prometheus, grafana, logrotate конфиги
- `scripts/` — cleanup.sh, backup.sh, deploy.sh
- `docs/system-design/ML_System_Design_Document.md` §8, §9
- `docs/system-design/global-plan.md` — глобальный план (пункты выше)
