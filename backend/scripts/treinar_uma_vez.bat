@echo off
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\..\.." >nul

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo [WARN] .venv não encontrado; usando Python do sistema...
)

if not exist "backend\results" mkdir "backend\results"
if not exist "backend\db" mkdir "backend\db"
if not exist "backend\__init__.py" type NUL > "backend\__init__.py"

echo [INFO] Treinando Q-Learning...
python -m backend.qlearning_sqlite ^
  --train_csv "backend\data\train.csv" ^
  --results_dir "backend\results" ^
  --db "backend\db\qtable.sqlite" ^
  --episodes 200 ^
  --epsilon_start 1.0 ^
  --epsilon_min 0.10 ^
  --epsilon_decay 0.995 ^
  --shuffle ^
  --verbose
if errorlevel 1 goto :err

echo [INFO] Gerando métricas...
python -m backend.metrics_eval ^
  --test_csv "backend\data\test.csv" ^
  --results_dir "backend\results"
if errorlevel 1 goto :err

echo [OK] Concluído. Saídas em backend\results
popd & endlocal & goto :eof

:err
echo [ERRO] Falha em uma das etapas. Veja mensagens acima.
popd & endlocal & exit /b 1
