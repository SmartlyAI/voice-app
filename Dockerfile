FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 5001

CMD ["gunicorn", "--workers=3", "--bind", "0.0.0.0:5001", "app:app"]