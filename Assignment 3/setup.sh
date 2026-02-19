#!/bin/bash

# Hand Biometrics Assignment - Setup Script
# This script prepares the environment for running the hand biometrics solution

echo "=========================================="
echo "Hand Biometrics Assignment - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Found: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found. Please install Python 3.7 or higher."
    exit 1
fi

# Create hand_images folder if it doesn't exist
echo ""
echo "Creating directories..."
if [ ! -d "hand_images" ]; then
    mkdir hand_images
    echo "✓ Created hand_images/ folder"
else
    echo "✓ hand_images/ folder already exists"
fi

# Create outputs folder if it doesn't exist
if [ ! -d "outputs" ]; then
    mkdir outputs
    echo "✓ Created outputs/ folder"
else
    echo "✓ outputs/ folder already exists"
fi

# Check if virtual environment exists
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo ""
echo "Installing dependencies..."
source venv/bin/activate

pip install --upgrade pip > /dev/null 2>&1
pip install numpy opencv-python matplotlib

echo ""
echo "Checking installed packages..."
python3 -c "import numpy; print('✓ numpy version:', numpy.__version__)"
python3 -c "import cv2; print('✓ opencv-python version:', cv2.__version__)"
python3 -c "import matplotlib; print('✓ matplotlib version:', matplotlib.__version__)"

# Check for images
echo ""
echo "Checking for hand images..."
IMAGE_COUNT=$(find hand_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l | tr -d ' ')

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "⚠ No images found in hand_images/ folder"
    echo "  Please add at least 5 hand images before running the script."
elif [ "$IMAGE_COUNT" -lt 5 ]; then
    echo "⚠ Found only $IMAGE_COUNT images (need 5)"
    echo "  Please add more hand images to hand_images/ folder."
else
    echo "✓ Found $IMAGE_COUNT images in hand_images/ folder"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Ensure you have 5 hand images in hand_images/ folder"
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "3. Run the script:"
echo "   python hand_biometrics.py"
echo ""
echo "For detailed instructions, see README_HAND_BIOMETRICS.md"
echo "=========================================="
