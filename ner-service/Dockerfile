FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости для Tesseract и PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-rus \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip install --no-cache-dir --upgrade pip "setuptools<70.0.0"

# Копируем и устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код сервиса
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]