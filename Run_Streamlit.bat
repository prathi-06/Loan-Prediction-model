@echo off
echo ===============================================
echo  LoanCheck India - Starting Streamlit App...
echo ===============================================
cd /d "%~dp0"
echo.
echo Starting Streamlit on http://localhost:8501
echo Open this URL in your browser if it doesn't open automatically.
echo Press CTRL+C to stop.
echo.
streamlit run app.py
pause
