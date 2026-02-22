#!/bin/bash
# Setup script for SageMaker container.
# Run this as part of the container build or as a startup script.

set -e

echo "=== Installing NLP dependencies ==="

# Install spaCy model
python -m spacy download en_core_web_sm 2>/dev/null || echo "spaCy model download failed (may already exist)"

# Download NLTK data
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print('NLTK data downloaded successfully')
"

echo "=== Setup complete ==="
