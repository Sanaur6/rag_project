@echo off
setlocal
cd /d "%~dp0"

if not exist "myenv\Scripts\python.exe" (
    echo Project Python environment was not found at myenv\Scripts\python.exe.
    pause
    exit /b 1
)

"myenv\Scripts\python.exe" -c "import extra_streamlit_components" >nul 2>&1
if errorlevel 1 (
    echo Required package is missing from myenv.
    echo Run: myenv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

"myenv\Scripts\python.exe" -m streamlit run app.py -- --streamlit