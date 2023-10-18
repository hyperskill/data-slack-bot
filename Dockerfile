FROM python:3.11-alpine
LABEL authors="Hyperskill Team"

COPY slack_bot ./

CMD ["python", "app.py"]
