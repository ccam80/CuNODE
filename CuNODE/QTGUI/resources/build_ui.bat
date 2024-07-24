@echo off
REM Check if a directory argument is provided
if "%~1"=="" (
    REM No argument provided, use the current directory
    set DIRECTORY=%cd%
) else (
    REM Use the provided argument as the directory
    set DIRECTORY=%~1%
)

REM Run the Python script with the specified directory
python build_ui.py "%DIRECTORY%"
pause
