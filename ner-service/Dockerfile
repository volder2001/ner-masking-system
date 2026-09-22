FROM python:3.12-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Открываем правильный порт
EXPOSE 8002

# ВАЖНО: запускаем app из файла main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]