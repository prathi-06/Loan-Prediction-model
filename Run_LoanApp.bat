@echo off
echo ===============================================
echo  LoanCheck India — Starting Flask Server...
echo ===============================================
cd /d "%~dp0"
pip install -q flask joblib numpy pandas scikit-learn xgboost 2>nul
echo.
echo  Server starting on http://localhost:5000
echo  Open this URL in your browser.
echo  Press CTRL+C to stop.
echo.
python server.py
pause
