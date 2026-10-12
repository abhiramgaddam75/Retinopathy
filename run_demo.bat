@echo off
rem Starts the DR-XAI demo web app and opens it in the browser. No internet is needed.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The Python environment .venv was not found in %cd%
    pause
    exit /b 1
)
set NO_ALBUMENTATIONS_UPDATE=1
echo Starting the DR-XAI demo. The browser opens at http://localhost:8501
echo Leave this window open while presenting. Close it to stop the app.
".venv\Scripts\python.exe" -m streamlit run app\demo_app.py --server.port 8501 --browser.gatherUsageStats false
pause
