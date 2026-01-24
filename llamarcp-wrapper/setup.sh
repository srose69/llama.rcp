#!/bin/bash
set -euo pipefail

echo "=== llamarcp-wrapper setup script ==="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create venv"
        exit 1
    fi
    echo "✓ venv created"
else
    echo "✓ venv already exists"
fi

# Activate venv
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate venv"
    exit 1
fi
echo "✓ venv activated"

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi
echo "✓ requirements installed"

# Install package in editable mode
echo ""
echo "Installing llamarcp package..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install llamarcp"
    exit 1
fi
echo "✓ llamarcp installed"

# Test import
echo ""
echo "Testing import..."
python -c "from llamarcp import LLAMA_FLASH_ATTN_TYPE_AUTO, LLAMA_MODEL_META_KEY_SAMPLING_TOP_K; print(f'Flash attn: {LLAMA_FLASH_ATTN_TYPE_AUTO}, Meta key: {LLAMA_MODEL_META_KEY_SAMPLING_TOP_K}')"
if [ $? -ne 0 ]; then
    echo "ERROR: Import test failed"
    exit 1
fi
echo "✓ Import test passed"

echo ""
echo "=== Setup complete ==="
