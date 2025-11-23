@echo off
echo ========================================
echo Animals-10 Classifier - Quick Start
echo ========================================
echo.

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)
echo.

echo [2/3] Installing dependencies...
echo Installing main requirements...
pip install -r requirements.txt
echo.

echo Installing backend requirements...
cd backend
pip install -r requirements.txt
cd ..
echo.

echo [3/3] Setup complete!
echo.
echo ========================================
echo Next Steps:
echo ========================================
echo.
echo 1. Train the model:
echo    python train_model.py
echo.
echo 2. Start the backend API:
echo    cd backend
echo    python app.py
echo.
echo 3. Open frontend/index.html in your browser
echo.
echo For detailed instructions, see README.md
echo ========================================
pause
