@echo off
setlocal enabledelayedexpansion
title Avenox Audio Studio - CLI Mode

cd /d "%~dp0"

if not exist "%~dp0python_embeded\python.exe" (
    echo [AVENOX] Lingkungan python_embeded belum ditemukan!
    echo Menjalankan setup scripts\setup_embedded_env.bat...
    call "%~dp0scripts\setup_embedded_env.bat"
)

set PATH=%~dp0python_embeded;%~dp0python_embeded\Scripts;%PATH%
set PYTHONHOME=%~dp0python_embeded
set PYTHONPATH=%~dp0;%~dp0src

"%~dp0python_embeded\python.exe" src\pipeline.py %*
