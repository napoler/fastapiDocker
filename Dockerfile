FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./app /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir transformers pytorch-lightning==1.1.6 pytorch-crf torch


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]