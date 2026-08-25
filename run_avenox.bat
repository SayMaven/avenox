@echo off
setlocal enabledelayedexpansion
title Avenox Audio Studio

cd /d "%~dp0"

if not exist "%~dp0python_embeded\python.exe" (
    echo =======================================================================
    echo [AVENOX] Lingkungan python_embeded belum ditemukan!
    echo Menjalankan inisialisasi otomatis via scripts\setup_embedded_env.bat...
    echo =======================================================================
    call "%~dp0scripts\setup_embedded_env.bat"
    if not exist "%~dp0python_embeded\python.exe" (
        echo [ERROR] Gagal menyiapkan python_embeded. Silakan periksa koneksi internet Anda.
        pause
        exit /b 1
    )
)

set PATH=%~dp0python_embeded;%~dp0python_embeded\Scripts;%PATH%
set PYTHONHOME=%~dp0python_embeded
set PYTHONPATH=%~dp0;%~dp0src

echo [AVENOX] Memulai Avenox Audio Studio Desktop...
"%~dp0python_embeded\python.exe" src\main.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [AVENOX] Aplikasi terhenti dengan kode error %ERRORLEVEL%.
    pause
)
