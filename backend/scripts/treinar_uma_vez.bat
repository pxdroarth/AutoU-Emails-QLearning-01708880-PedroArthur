@echo off
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\..\.." >nul

rem =========================
rem Config rápido (opcional)
rem =========================
set TRAIN_CSV=backend\data\train.csv
set RESULTS_DIR=backend\results
set DB_PATH=backend\db\qtable.sqlite

rem =====================================
rem 0) venv (se existir) e pastas básicas
rem =====================================
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo [WARN] .venv nao encontrado; usando Python do sistema...
)

if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "backend\db" mkdir "backend\db"
if not exist "backend\__init__.py" type NUL > "backend\__init__.py"

rem ===============================
rem 1) Treino Q-Learning (SQLite)
rem ===============================
echo ================================================
echo [INFO] Iniciando treino Q-Learning
echo ================================================
python -m backend.qlearning_sqlite ^
  --train_csv "%TRAIN_CSV%" ^
  --results_dir "%RESULTS_DIR%" ^
  --db "%DB_PATH%" ^
  --episodes 200 ^
  --epsilon_start 1.0 ^
  --epsilon_min 0.10 ^
  --epsilon_decay 0.995 ^
  --shuffle ^
  --verbose
if errorlevel 1 goto :err

rem =========================================================
rem 2) Gerar predicoes.csv a partir da Q-Table (sem usar API)
rem    - usa backend/data/test.csv; se nao existir, usa train.csv
rem =========================================================
echo.
echo ================================================
echo [INFO] Gerando predicoes.csv a partir da Q-Table
echo ================================================
python -m backend.tools.make_predicoes --results_dir "%RESULTS_DIR%" --db "%DB_PATH%"
if errorlevel 1 (
  echo [WARN] Nao foi possivel gerar predicoes.csv. Prosseguindo apenas com curvas de treino...
)

rem ==========================================
rem 3) Relatorio: curvas e metricas/figuras
rem ==========================================
echo.
echo ================================================
echo [INFO] Gerando metricas e graficos pos-treino
echo ================================================
python backend\tools\make_results_report.py --results "%RESULTS_DIR%"
if errorlevel 1 goto :err

rem =======================
rem 4) Resumo das saidas
rem =======================
echo.
echo [OK] Treino e metricas concluidos com sucesso.
echo Resultados em: %RESULTS_DIR%
echo -----------------------------------------------
dir /b "%RESULTS_DIR%"
echo -----------------------------------------------
pause
popd & endlocal & goto :eof

:err
echo [ERRO] Falha em uma das etapas. Veja mensagens acima.
popd & endlocal & exit /b 1
