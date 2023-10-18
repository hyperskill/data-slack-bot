FROM python:3.11-alpine
LABEL authors="Hyperskill Team"

RUN apk add --no-cache bash \
  && python -m pip install --upgrade poetry==1.5.1

COPY slack_bot/pyproject.toml slack_bot/poetry.lock ./
RUN poetry config virtualenvs.create false \
  && poetry install --only main --no-interaction --no-ansi --no-cache

COPY slack_bot ./

CMD ["python", "app.py"]
