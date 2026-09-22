FROM python:3.12-slim

WORKDIR /app

# 1. ГАРАНТИРОВАННО обновляем pip и ставим setuptools ПЕРВЫМ ДЕЛОМ
RUN pip install --no-cache-dir --upgrade pip setuptools

# 2. Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем код
COPY . .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]