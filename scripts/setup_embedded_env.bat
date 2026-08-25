@echo off
setlocal
title Avenox - Setup Embedded Environment

cd /d "%~dp0\.."

echo [AVENOX] Memulai inisialisasi lingkungan Embedded Python terisolasi...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_embedded_env.ps1"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Terjadi kesalahan saat menginisialisasi lingkungan.
    pause
    exit /b 1
)

echo.
pause
