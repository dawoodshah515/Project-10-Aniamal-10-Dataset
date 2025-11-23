#!/bin/bash

echo "========================================"
echo "Animals-10 Classifier - Quick Start"
echo "========================================"
echo ""

echo "[1/3] Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found! Please install Python 3.8 or higher."
    exit 1
fi
echo ""

echo "[2/3] Installing dependencies..."
echo "Installing main requirements..."
pip3 install -r requirements.txt
echo ""

echo "Installing backend requirements..."
cd backend
pip3 install -r requirements.txt
cd ..
echo ""

echo "[3/3] Setup complete!"
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""
echo "1. Train the model:"
echo "   python3 train_model.py"
echo ""
echo "2. Start the backend API:"
echo "   cd backend"
echo "   python3 app.py"
echo ""
echo "3. Open frontend/index.html in your browser"
echo ""
echo "For detailed instructions, see README.md"
echo "========================================"
