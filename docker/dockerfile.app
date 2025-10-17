FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
COPY . /app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_EXPERIMENT_NAME=telco-churn
EXPOSE 8000 8501
CMD ["bash", "-lc", "uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 & streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0"]