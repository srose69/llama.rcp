#!/bin/bash
set -euo pipefail

#==============================================================================
# llamarcp-wrapper Setup Script
#==============================================================================
# Usage:
#   ./setup.sh           - Dev mode: create symlinks to build/bin/
#   ./setup.sh pipinstall - Distribution mode: copy binaries for pip package
#==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BIN_DIR="${SCRIPT_DIR}/../build/bin"
LIB_DIR="${SCRIPT_DIR}/llamarcp/lib"
PIPINSTALL_MODE=false

# Parse arguments
if [ "${1:-}" = "pipinstall" ]; then
    PIPINSTALL_MODE=true
fi

echo "=== llamarcp-wrapper setup script ==="
echo "Mode: $([ "${PIPINSTALL_MODE}" = true ] && echo "pipinstall (distribution)" || echo "dev (symlinks)")"
echo ""

#==============================================================================
# Setup library files (symlinks or copies)
#==============================================================================
setup_libraries() {
    echo "Setting up library files..."
    
    # Check if build directory exists
    if [ ! -d "${BUILD_BIN_DIR}" ]; then
        echo ""
        echo "ERROR: Build directory not found: ${BUILD_BIN_DIR}"
        echo ""
        echo "Please build the main project first:"
        echo "  cd ${SCRIPT_DIR}/.."
        echo "  ./setup.sh           # Interactive build"
        echo "  # or:"
        echo "  ./setup.sh imlazy <model.gguf>  # One-click setup"
        echo ""
        exit 1
    fi
    
    # Check if any .so files exist in build
    if ! ls "${BUILD_BIN_DIR}"/*.so* >/dev/null 2>&1; then
        echo ""
        echo "ERROR: No .so files found in ${BUILD_BIN_DIR}"
        echo "Please build the project first: cd .. && ./setup.sh"
        echo ""
        exit 1
    fi
    
    # Create lib directory if it doesn't exist
    mkdir -p "${LIB_DIR}"
    
    if [ "${PIPINSTALL_MODE}" = true ]; then
        # Distribution mode: copy real files
        echo "Copying library files from build/bin/..."
        
        # Remove existing files/symlinks
        rm -f "${LIB_DIR}"/*.so*
        
        # Copy all .so files
        cp -v "${BUILD_BIN_DIR}"/*.so* "${LIB_DIR}/"
        
        echo "✓ Libraries copied ($(du -sh "${LIB_DIR}" | cut -f1))"
    else
        # Dev mode: create symlinks
        echo "Creating symlinks to build/bin/..."
        
        # Remove existing files/symlinks
        rm -f "${LIB_DIR}"/*.so*
        
        # Create symlinks
        cd "${LIB_DIR}" || exit 1
        ln -s ../../../build/bin/*.so* .
        cd "${SCRIPT_DIR}" || exit 1
        
        echo "✓ Symlinks created ($(du -sh "${LIB_DIR}" | cut -f1))"
    fi
    
    echo "✓ Library setup complete"
    echo ""
}

# Setup libraries first
setup_libraries

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
