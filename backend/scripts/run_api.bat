@echo off
setlocal enabledelayedexpansion

REM === Descobrir paths de forma robusta ===
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.."
set "BACKEND_DIR=%CD%"
popd
pushd "%BACKEND_DIR%\.."
set "PROJECT_ROOT=%CD%"
popd

REM === Ativar venv (tenta em backend/.venv e depois na raiz/.venv) ===
if exist "%BACKEND_DIR%\.venv\Scripts\activate.bat" call "%BACKEND_DIR%\.venv\Scripts\activate.bat"
if not defined VIRTUAL_ENV if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"

REM === Garantir pastas geradas em runtime ===
if not exist "%BACKEND_DIR%\db" mkdir "%BACKEND_DIR%\db"
if not exist "%BACKEND_DIR%\results" mkdir "%BACKEND_DIR%\results"

REM === Rodar Uvicorn a partir da RAIZ p/ evitar problemas de import ===
pushd "%PROJECT_ROOT%"
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
popd

endlocal
