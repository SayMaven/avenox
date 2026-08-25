@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0python_embeded\python.exe" (
    echo [ERROR] python_embeded belum diinisialisasi. Jalankan scripts\setup_embedded_env.bat terlebih dahulu.
    pause
    exit /b 1
)

set PATH=%~dp0python_embeded;%~dp0python_embeded\Scripts;%PATH%

echo [AVENOX] Menjalankan pip install ke python_embeded: %*
"%~dp0python_embeded\python.exe" -m pip install %*
pause
