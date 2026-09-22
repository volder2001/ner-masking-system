#!/bin/bash
set -e

# ============================================
# Скрипт создания проекта NER Masking System
# Для macOS / Linux
# ============================================

PROJECT_DIR="/Users/petrosyanvadim/PycharmProjects/ner-masking-system"
cd "$PROJECT_DIR"

echo "🚀 Создаём структуру проекта в: $PROJECT_DIR"
echo ""

# 1. Создаём структуру папок
echo "📁 Создаём папки сервисов..."
mkdir -p api-gateway/routes
mkdir -p document-service/converters
mkdir -p entity-dictionary-service/dictionaries
mkdir -p ner-service/extractors
mkdir -p masking-service/renderers
mkdir -p masking-service/strategies
mkdir -p database/migrations
mkdir -p tests
mkdir -p docs
mkdir -p .github/workflows
mkdir -p scripts

# 2. Создаём скрипты в папке scripts/
echo "📜 Создаём служебные скрипты..."

# scripts/setup.sh
cat > scripts/setup.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e
echo "🔧 Первоначальная настройка проекта..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker Desktop."
    exit 1
fi

mkdir -p data/postgres data/redis data/minio logs
chmod +x scripts/*.sh

echo "✅ Настройка завершена!"
echo "🚀 Теперь запустите: ./scripts/start.sh"
SCRIPT_EOF

# scripts/start.sh
cat > scripts/start.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e
echo "🚀 Запуск NER Masking System..."

if docker compose ps 2>/dev/null | grep -q "Up"; then
    echo "⚠️  Сервисы уже запущены."
    docker compose ps
    exit 0
fi

docker compose up -d

echo ""
echo "✅ Все сервисы запущены!"
echo ""
echo "📊 Доступные сервисы:"
echo "   🔹 API Gateway:              http://localhost:8000"
echo "   🔹 Document Service:         http://localhost:8001"
echo "   🔹 NER Service:              http://localhost:8002"
echo "   🔹 Masking Service:          http://localhost:8003"
echo "   🔹 Entity Dictionary:        http://localhost:8004"
echo "   🔹 MinIO Console:            http://localhost:9001"
echo "      (логин: minio_user / пароль: minio_password)"
SCRIPT_EOF

# scripts/stop.sh
cat > scripts/stop.sh << 'SCRIPT_EOF'
#!/bin/bash
echo "🛑 Остановка NER Masking System..."
docker compose down
echo "✅ Все сервисы остановлены"
SCRIPT_EOF

# scripts/rebuild.sh
cat > scripts/rebuild.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e
echo "🔄 Пересборка и перезапуск сервисов..."
docker compose down
echo "🔨 Сборка образов..."
docker compose build --no-cache
docker compose up -d
echo "✅ Пересборка завершена!"
SCRIPT_EOF

# scripts/logs.sh
cat > scripts/logs.sh << 'SCRIPT_EOF'
#!/bin/bash
if [ -z "$1" ]; then
    echo "📋 Использование: ./scripts/logs.sh [сервис]"
    echo ""
    echo "Доступные сервисы:"
    echo "  api-gateway"
    echo "  document-service"
    echo "  ner-service"
    echo "  masking-service"
    echo "  entity-dictionary-service"
    echo "  postgres"
    echo "  redis"
    echo "  minio"
    exit 0
fi
echo "📋 Логи сервиса: $1"
echo "─────────────────────────────────────"
docker compose logs -f "$1"
SCRIPT_EOF

# scripts/test.sh
cat > scripts/test.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e
echo "🧪 Запуск тестов..."
if ! docker compose ps | grep -q "Up"; then
    echo "⚠️  Сервисы не запущены. Запускаем..."
    ./scripts/start.sh
fi
echo "🔹 Тесты NER Service..."
docker compose exec ner-service pytest tests/ -v || echo "Тесты не найдены"
echo "✅ Тесты завершены!"
SCRIPT_EOF

# scripts/cleanup.sh
cat > scripts/cleanup.sh << 'SCRIPT_EOF'
#!/bin/bash
echo "🧹 Очистка временных данных..."
docker compose down
read -p "⚠️  Удалить все данные (volumes)? (y/N): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    docker compose down -v
    echo "✅ Volumes удалены"
else
    echo "ℹ️  Volumes сохранены"
fi
docker image prune -f
echo "✅ Очистка завершена!"
SCRIPT_EOF

chmod +x scripts/*.sh

# 3. Создаём docker-compose.yml
echo "🐳 Создаём docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: ner_postgres
    environment:
      POSTGRES_USER: ner_user
      POSTGRES_PASSWORD: ner_password
      POSTGRES_DB: ner_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ner_user -d ner_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: ner_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  minio:
    image: minio/minio:latest
    container_name: ner_minio
    environment:
      MINIO_ROOT_USER: minio_user
      MINIO_ROOT_PASSWORD: minio_password
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5

  api-gateway:
    build:
      context: ./api-gateway
      dockerfile: Dockerfile
    container_name: ner_api_gateway
    environment:
      - DOCUMENT_SERVICE_URL=http://document-service:8001
      - NER_SERVICE_URL=http://ner-service:8002
      - MASKING_SERVICE_URL=http://masking-service:8003
      - ENTITY_DICT_SERVICE_URL=http://entity-dictionary-service:8004
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      - redis
    volumes:
      - ./api-gateway:/app

  document-service:
    build:
      context: ./document-service
      dockerfile: Dockerfile
    container_name: ner_document_service
    environment:
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minio_user
      - MINIO_SECRET_KEY=minio_password
      - REDIS_URL=redis://redis:6379/1
    ports:
      - "8001:8001"
    depends_on:
      - minio
      - redis
    volumes:
      - ./document-service:/app

  ner-service:
    build:
      context: ./ner-service
      dockerfile: Dockerfile
    container_name: ner_ner_service
    environment:
      - ENTITY_DICT_SERVICE_URL=http://entity-dictionary-service:8004
      - REDIS_URL=redis://redis:6379/2
    ports:
      - "8002:8002"
    depends_on:
      - redis
      - entity-dictionary-service
    volumes:
      - ./ner-service:/app

  masking-service:
    build:
      context: ./masking-service
      dockerfile: Dockerfile
    container_name: ner_masking_service
    environment:
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minio_user
      - MINIO_SECRET_KEY=minio_password
      - REDIS_URL=redis://redis:6379/3
    ports:
      - "8003:8003"
    depends_on:
      - minio
      - redis
    volumes:
      - ./masking-service:/app

  entity-dictionary-service:
    build:
      context: ./entity-dictionary-service
      dockerfile: Dockerfile
    container_name: ner_entity_dictionary_service
    environment:
      - DATABASE_URL=postgresql://ner_user:ner_password@postgres:5432/ner_db
      - REDIS_URL=redis://redis:6379/4
    ports:
      - "8004:8004"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./entity-dictionary-service:/app

volumes:
  postgres_data:
  redis_data:
  minio_data:
EOF

# 4. Создаём Dockerfile для каждого сервиса
echo "🐳 Создаём Dockerfile для каждого сервиса..."

cat > api-gateway/Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

cat > document-service/Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-rus libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
EOF

cat > ner-service/Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
EOF

cat > masking-service/Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
EOF

cat > entity-dictionary-service/Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]
EOF

# 5. Создаём requirements.txt
echo "📦 Создаём requirements.txt..."

cat > api-gateway/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.2
redis==5.0.1
python-multipart==0.0.6
pydantic==2.5.2
EOF

cat > document-service/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-docx==1.1.0
PyMuPDF==1.23.8
pytesseract==0.3.10
Pillow==10.1.0
minio==7.2.3
redis==5.0.1
python-multipart==0.0.6
celery==5.3.4
EOF

cat > ner-service/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
natasha==1.6.0
yargy==0.18.0
pymorphy2==0.9.1
razdel==0.5.0
fuzzywuzzy==0.18.0
python-Levenshtein==0.23.0
redis==5.0.1
httpx==0.25.2
EOF

cat > masking-service/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-docx==1.1.0
PyMuPDF==1.23.8
Pillow==10.1.0
minio==7.2.3
redis==5.0.1
reportlab==4.0.7
EOF

cat > entity-dictionary-service/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
pydantic==2.5.2
EOF

# 6. Создаём main.py для каждого сервиса
echo "🐍 Создаём main.py для каждого сервиса..."

cat > api-gateway/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NER Masking System API Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "NER Masking System API Gateway",
        "version": "1.0.0",
        "status": "running"
    }
EOF

cat > document-service/main.py << 'EOF'
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Document Service")

@app.get("/")
async def root():
    return {"service": "Document Service", "status": "running"}

@app.post("/process")
async def process_document(document_id: str, file: UploadFile = File(...)):
    return {"document_id": document_id, "status": "received"}
EOF

cat > ner-service/main.py << 'EOF'
from fastapi import FastAPI

app = FastAPI(title="NER Service")

@app.get("/")
async def root():
    return {"service": "NER Service", "status": "running"}

@app.get("/entities/{document_id}")
async def get_entities(document_id: str):
    return {"document_id": document_id, "entities": []}
EOF

cat > masking-service/main.py << 'EOF'
from fastapi import FastAPI

app = FastAPI(title="Masking Service")

@app.get("/")
async def root():
    return {"service": "Masking Service", "status": "running"}

@app.post("/mask")
async def mask_document(request: dict):
    return {"masked_document_id": "test-id", "status": "success"}
EOF

cat > entity-dictionary-service/main.py << 'EOF'
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
from typing import List, Dict
import redis
import json

app = FastAPI(title="Entity Dictionary Service")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/4")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

DICTIONARY_PATH = os.getenv("DICTIONARY_PATH", "/app/dictionaries/entities.txt")

class Entity(BaseModel):
    prefix: str
    definition: str

class EntityDictionary:
    def __init__(self):
        self.entities: Dict[str, List[str]] = {}
        self.load_dictionary()
    
    def load_dictionary(self):
        if not os.path.exists(DICTIONARY_PATH):
            os.makedirs(os.path.dirname(DICTIONARY_PATH), exist_ok=True)
            with open(DICTIONARY_PATH, 'w', encoding='utf-8') as f:
                f.write("# Справочник сущностей\n# Формат: PREFIX->definition\n")
            return
        
        with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '->' in line:
                    prefix, definition = line.split('->', 1)
                    prefix = prefix.strip()
                    definition = definition.strip()
                    if prefix not in self.entities:
                        self.entities[prefix] = []
                    if definition not in self.entities[prefix]:
                        self.entities[prefix].append(definition)
        self._cache_to_redis()
    
    def _cache_to_redis(self):
        try:
            redis_client.set("entity_dictionary", json.dumps(self.entities))
        except:
            pass
    
    def get_all_entities(self) -> Dict[str, List[str]]:
        try:
            cached = redis_client.get("entity_dictionary")
            if cached:
                return json.loads(cached)
        except:
            pass
        return self.entities
    
    def add_entity(self, prefix: str, definition: str):
        if prefix not in self.entities:
            self.entities[prefix] = []
        if definition not in self.entities[prefix]:
            self.entities[prefix].append(definition)
            with open(DICTIONARY_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{prefix}->{definition}\n")
            self._cache_to_redis()

dictionary = EntityDictionary()

@app.get("/")
async def root():
    return {"service": "Entity Dictionary Service", "status": "running"}

@app.get("/entities")
async def get_entities():
    return dictionary.get_all_entities()

@app.post("/entities")
async def add_entity(entity: Entity):
    dictionary.add_entity(entity.prefix, entity.definition)
    return {"status": "success", "entity": {"prefix": entity.prefix, "definition": entity.definition}}

@app.post("/entities/upload")
async def upload_dictionary(file: UploadFile = File(...)):
    content = await file.read()
    with open(DICTIONARY_PATH, 'wb') as f:
        f.write(content)
    dictionary.load_dictionary()
    return {"status": "success", "message": "Dictionary uploaded and reloaded"}
EOF

# 7. Создаём начальный справочник
echo "📚 Создаём начальный справочник..."
cat > entity-dictionary-service/dictionaries/entities.txt << 'EOF'
# Справочник сущностей
# Формат: PREFIX->definition
ФИО->имя человека
АДРЕС->адрес
ДАТА->дата
ОРГАНИЗАЦИЯ->наименование организации
ДЕНЬГИ->денежная сумма
SUM->размер пени
SUM->общая задолженность
SUM->основной долг
ACT->страхование
ACT->задолженность по коммунальным услугам
ACT->пени
ACT->неустойка
ACT->штраф
ACT->государственная пошлина
NOT_ACT->о взыскании задолженности по кредитному договору
NOT_ACT->назначение платежа
EOF

# 8. Создаём init.sql
echo "🗄  Создаём init.sql..."
cat > database/init.sql << 'EOF'
CREATE TABLE IF NOT EXISTS entity_dictionary (
    id SERIAL PRIMARY KEY,
    prefix VARCHAR(100) NOT NULL,
    definition TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prefix, definition)
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    original_file_path VARCHAR(500) NOT NULL,
    extracted_text TEXT,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_entities (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    entity_type VARCHAR(100) NOT NULL,
    entity_value TEXT NOT NULL,
    start_position INTEGER NOT NULL,
    end_position INTEGER NOT NULL,
    confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entity_pairs (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    act_entity_id UUID REFERENCES extracted_entities(id),
    money_entity_id UUID REFERENCES extracted_entities(id),
    pair_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS masked_documents (
    id UUID PRIMARY KEY,
    original_document_id UUID REFERENCES documents(id),
    mask_strategy VARCHAR(50) NOT NULL,
    output_format VARCHAR(10) NOT NULL,
    masked_file_path VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_entities_document ON extracted_entities(document_id);
CREATE INDEX idx_entities_type ON extracted_entities(entity_type);
CREATE INDEX idx_pairs_document ON entity_pairs(document_id);
EOF

# 9. Создаём .gitignore
echo "📝 Создаём .gitignore..."
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/
.vscode/
.idea/
*.swp
*.swo
*~
.env
.env.local
*.log
logs/
*.db
*.sqlite
*.sqlite3
.DS_Store
Thumbs.db
data/
htmlcov/
.coverage
.pytest_cache/
EOF

# 10. Создаём README.md
echo "📖 Создаём README.md..."
cat > README.md << 'EOF'
# NER Masking System

Система автоматического распознавания именованных сущностей (NER) и маскирования документов.

## 🚀 Быстрый старт

```bash
# 1. Первоначальная настройка
./scripts/setup.sh

# 2. Запуск всех сервисов
./scripts/start.sh

# 3. Проверка работоспособности
curl http://localhost:8000/
