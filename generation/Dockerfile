FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN adduser --disabled-password --gecos "" gptuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R gptuser /app
USER gptuser

EXPOSE 8000

CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8000", "wsgi:app"]