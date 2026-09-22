FROM python:3.12-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Запускаем сервис
EXPOSE 8001
CMD ["python", "ner_service.py"]FROM python:3.12-slim

WORKDIR /app

# Устанавливаем системные зависимости (если понадобятся для natasha/yargy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код сервиса
COPY . .

# Открываем правильный порт (как в docker-compose.yml)
EXPOSE 8002

# Запускаем FastAPI через uvicorn на порту 8002
CMD ["uvicorn", "ner_service:app", "--host", "0.0.0.0", "--port", "8002"]