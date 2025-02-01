# Dockerfile (stt)
FROM python:3.9

WORKDIR /app
# 1) Системные зависимости
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 git
# 2) Копируем requirements и устанавливаем Python-зависимости
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 3) Скачиваем модель spaCy (ru_core_news_sm)
RUN python -m spacy download ru_core_news_sm

# 4) Копируем код
COPY . .

# 5) Открываем порт
EXPOSE 8000

# 6) Запускаем сервер
CMD ["uvicorn", "speech:app", "--host", "0.0.0.0", "--port", "8000"]
