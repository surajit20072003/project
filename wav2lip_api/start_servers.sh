#!/bin/bash
set -e

# Use the full path to the conda environment's Python
CONDA_PYTHON_EXEC="/venv/chatterbox/bin/python"

echo "--- Starting Chatterbox API Service in the background... ---"

# Run the API script using the full path
$CONDA_PYTHON_EXEC chatterbox_api.py &
CHATTERBOX_PID=$!
echo "Chatterbox API started with PID: $CHATTERBOX_PID"

# Give the model time to load
sleep 20

echo ""
echo "--- Starting Django Server in the foreground... ---"
source ../venv/bin/activate
python manage.py runserver 0.0.0.0:8000