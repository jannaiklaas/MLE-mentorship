#!/bin/bash
# entrypoint.sh

# Fail on any error
set -e

# Debug log for seeing the experiment number
echo "Running experiment number: $EXPERIMENT_NUMBER"

# Execute Python script
exec python "/app/experiment_$EXPERIMENT_NUMBER/train.py"
