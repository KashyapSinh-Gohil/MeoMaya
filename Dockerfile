FROM python:3.12-slim

WORKDIR /app

COPY meomaya /app/meomaya
COPY meomaya/requirements.txt /app/meomaya/requirements.txt

RUN pip install --no-cache-dir -r /app/meomaya/requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "meomaya.api.server:app", "--host", "0.0.0.0", "--port", "8000"]


