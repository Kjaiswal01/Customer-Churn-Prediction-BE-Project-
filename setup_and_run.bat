@echo off
setlocal EnableExtensions
cls

echo ============================================================
echo   Customer Churn Prediction - Setup and Run
echo ============================================================
echo.

cd /d "%~dp0"

set "PYTHON_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.11"
)

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.12 --version >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=py -3.12"
    )
)

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo Python is not installed or not added to PATH.
    echo Please install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo During installation, tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

echo Using Python:
%PYTHON_CMD% --version
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version >nul 2>nul
    if errorlevel 1 (
        echo Existing virtual environment is not usable on this PC.
        echo Recreating .venv...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo Installing project dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    echo Check internet connection, then run this file again.
    echo Recommended Python version: 3.11 or 3.12.
    pause
    exit /b 1
)

echo.
echo Starting Streamlit app...
echo Browser URL: http://localhost:8501
echo.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501

echo.
echo App stopped.
pause
