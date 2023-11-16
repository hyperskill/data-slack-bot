FROM python:3.11-alpine
LABEL authors="Hyperskill Team"

RUN apk add --no-cache bash \
  && python -m pip install --upgrade poetry==1.6.1

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
  && poetry install --only main --no-interaction --no-ansi --no-cache

COPY slack_bot slack_bot/
COPY app.py ./

CMD ["python", "app.py"]
