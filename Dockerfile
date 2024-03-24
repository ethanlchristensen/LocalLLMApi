FROM python:3.11-slim

WORKDIR /usr/src/app

RUN pip install poetry

COPY pyproject.toml poetry.lock* /usr/src/app/

RUN poetry config virtualenvs.create false \
  && poetry install --no-dev

COPY . /usr/src/app

EXPOSE 8000
EXPOSE 11434

CMD ["uvicorn", "--app-dir", "app", "app:app", "--host", "0.0.0.0", "--port", "8000"]
