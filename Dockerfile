
FROM python:3.11-slim

WORKDIR /app

COPY requirements-serve.txt .

RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app.py .
COPY preprocessing.py .

COPY model/model.joblib model/model.joblib
COPY static/ static/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
