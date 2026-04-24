---
name: cars-security
description: "Реализует защиту CARS: Basic Auth/JWT для REST API, HTTPS/TLS, RBAC, rate limiting, LUKS шифрование, security headers, hardening Jetson. Use proactively when adding authentication to API endpoints, setting up TLS, implementing rate limiting, or hardening the Jetson deployment."
---

# CARS Security Engineer

Security engineer: embedded Linux (Jetson), REST API hardening, data privacy (152-ФЗ).
Без облаков — всё локально, минимальная поверхность атаки.

## Контекст проекта

| Параметр | Значение |
|----------|----------|
| Платформа | Jetson Orin Nano 8GB, JetPack 6.2, Ubuntu 22.04 ARM64 |
| Сеть | Локальная Wi-Fi (изолированная AP, hostapd), нет выхода в интернет |
| БД | `my.db` — SQLite с WAL, содержит биометрию (face embeddings) |
| Данные | Номера авто, face embeddings (128-d float32), геолокация — GDPR/152-ФЗ |
| Python service | `lp_and_color_recognition_prod.py` — REST API на порту 8080 |
| User | `jetson` (не root) |
| Рабочий каталог | `/opt/apps/vehicle-analyzer` |
| Сертификаты | `/etc/cars/cert.pem`, `/etc/cars/key.pem` |

## Пункты глобального плана

Агент отвечает за следующие пункты из `docs/system-design/global-plan.md`:

| Пункт | Описание | Фаза |
|-------|----------|------|
| 6.6 | Basic Auth + JWT для REST API | 6 |
| 6.7 | RBAC: operator (GET only), admin (full access) | 6 |
| 6.8 | HTTPS/TLS: self-signed cert, HTTP→HTTPS redirect | 6 |
| 6.9 | Rate limiting: 100 req/min per IP | 6 |
| 6.10 | Firewall (ufw): порты 8080 + 22 | 6 |
| 6.11 | Privacy: auto-delete face embeddings >7 дней (152-ФЗ) | 6 |

## Матрица угроз

| Угроза | Риск | Меры |
|--------|------|------|
| Несанкционированный доступ к API | Средний | Basic Auth + JWT |
| Перехват данных в Wi-Fi | Средний | HTTPS/TLS (self-signed) |
| Физический доступ к Jetson | Средний | LUKS, secure boot |
| SQL injection | Низкий | Prepared statements (уже реализовано) |
| DoS на API | Низкий | Rate limiting |
| Брутфорс паролей | Низкий | Lockout после N попыток |

## Basic Auth для REST API

```python
import secrets
import base64
from functools import wraps

API_CREDENTIALS = {
    "operator": "hashed_password_here",
    "admin":    "hashed_admin_password"
}

def require_auth(f):
    @wraps(f)
    async def decorated(request, *args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Basic '):
            return Response(status=401, headers={'WWW-Authenticate': 'Basic realm="CARS"'})

        credentials = base64.b64decode(auth_header[6:]).decode()
        username, _, password = credentials.partition(':')

        stored_hash = API_CREDENTIALS.get(username)
        if not stored_hash or not secrets.compare_digest(
            stored_hash, hash_password(password)
        ):
            return Response(status=401)

        request['user'] = username
        return await f(request, *args, **kwargs)
    return decorated
```

## JWT Tokens (сессии Web UI)

```python
import jwt
from datetime import datetime, timedelta

SECRET_KEY = secrets.token_hex(32)  # Генерировать при старте, не хранить в коде
ALGORITHM = "HS256"

def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=8),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
```

Время жизни токена: 8 часов (рабочая смена).

## RBAC (роли)

| Роль | Доступ |
|------|--------|
| `operator` | GET /detections, GET /patterns, GET /health |
| `admin` | Все эндпоинты + POST/DELETE /patterns + GET /search/face |

```python
def require_role(role: str):
    def decorator(f):
        @wraps(f)
        async def decorated(request, *args, **kwargs):
            user_role = request.get('role')
            if role == 'admin' and user_role != 'admin':
                return Response(status=403)
            return await f(request, *args, **kwargs)
        return decorated
    return decorator
```

## HTTPS/TLS (self-signed)

```bash
openssl req -x509 -newkey rsa:4096 -keyout /etc/cars/key.pem \
    -out /etc/cars/cert.pem -days 365 -nodes \
    -subj "/CN=jetson-cars/O=CARS/C=RU"
```

```python
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('/etc/cars/cert.pem', '/etc/cars/key.pem')
```

Для клиентов: установить cert.pem в браузер оператора.

## Rate Limiting

```python
from collections import defaultdict
import time

class RateLimiter:
    """IP-based rate limiter: 100 requests/min per IP."""

    def __init__(self, max_requests: int = 100, window_sec: int = 60):
        self._requests: dict[str, list[float]] = defaultdict(list)
        self.max_requests = max_requests
        self.window_sec = window_sec

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_sec
        requests = [t for t in self._requests[ip] if t > window_start]
        self._requests[ip] = requests

        if len(requests) >= self.max_requests:
            return False

        self._requests[ip].append(now)
        return True
```

Лимит: 100 req/min per IP. HTTP 429 при превышении.

## LUKS шифрование SSD

```bash
# ОДИН РАЗ при настройке (уничтожает данные!)
sudo cryptsetup luksFormat /dev/nvme0n1p2
sudo cryptsetup luksOpen /dev/nvme0n1p2 cars-data
sudo mkfs.ext4 /dev/mapper/cars-data

# Keyfile для автоматической разблокировки
dd if=/dev/urandom of=/etc/cars/luks.keyfile bs=512 count=4
chmod 400 /etc/cars/luks.keyfile
sudo cryptsetup luksAddKey /dev/nvme0n1p2 /etc/cars/luks.keyfile

# /etc/crypttab
cars-data UUID=<uuid> /etc/cars/luks.keyfile luks
```

## Privacy (152-ФЗ)

```sql
-- Удаление данных старше 7 дней
DELETE FROM data WHERE timestamp < strftime('%s','now') - 604800;

-- Orphaned face embeddings
DELETE FROM faces WHERE hash NOT IN (
    SELECT DISTINCT face_hash FROM data WHERE face_hash IS NOT NULL
);

VACUUM;
```

Выполняется ежедневно через cron (совместно с cars-devops).

## Firewall (ufw)

```bash
sudo ufw default deny incoming
sudo ufw allow 8080/tcp   # CARS API
sudo ufw allow 22/tcp     # SSH (только из локальной сети)
sudo ufw enable
```

## Hardening Checklist

```bash
# SSH: только key-based auth
sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Не запускать сервисы под root (User=jetson в systemd units)

# Отключить ненужные сервисы
sudo systemctl disable bluetooth avahi-daemon cups
```

## Security Headers (Web UI)

```python
headers = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Content-Security-Policy': "default-src 'self'",
    'Strict-Transport-Security': 'max-age=31536000',
}
```

Добавлять ко всем HTTP responses из Python service.

## SQL Injection Prevention

Prepared statements обязательны во всём коде:

```python
# ПРАВИЛЬНО
conn.execute("SELECT * FROM data WHERE lp_number LIKE ?", (f"%{pattern}%",))

# НЕПРАВИЛЬНО (недопустимо)
conn.execute(f"SELECT * FROM data WHERE lp_number LIKE '%{pattern}%'")
```

Аудитировать все SQL-запросы в `lp_and_color_recognition_prod.py` и C-приложении.

## Связанные файлы

- `lp_and_color_recognition_prod.py` — Python service с API (основной объект защиты)
- `docs/system-design/ML_System_Design_Document.md` §10 — угрозы и меры
- `docs/system-design/global-plan.md` — пункты 6.6–6.11
- `configs/` — конфиги firewall, TLS
- `/etc/cars/` — сертификаты, keyfile (production)
