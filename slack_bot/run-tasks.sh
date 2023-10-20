#!/bin/bash

cd slack_bot || exit

echo "Formatting code..."
black .

echo "Checking types..."
mypy .

echo "Running ruff command..."
ruff check . --fix
