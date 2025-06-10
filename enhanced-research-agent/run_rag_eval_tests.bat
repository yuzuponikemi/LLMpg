@echo off
echo Running RAG Evaluation Tests
echo ===========================
echo.

REM Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo WARNING: Virtual environment not found at venv\Scripts\activate.bat
    echo Continuing without activating a virtual environment...
)

echo.
echo Starting RAG evaluation tests...
echo.

REM Run the tests
python test\test_rag_eval_mode.py %*

REM Check the exit code
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Tests failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo RAG evaluation tests completed successfully!
echo Check the test\test_results directory for detailed results.
echo.

REM Deactivate the virtual environment if it was activated
if exist venv\Scripts\activate.bat (
    call deactivate
    echo Virtual environment deactivated
)

pause
