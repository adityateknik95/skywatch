@echo off
REM Double-click this to cleanly start Docker Desktop + the SkyWatch database.
REM Works around the Docker Desktop startup crash caused by zombie unix sockets
REM (the space in the Windows username). Safe to run anytime.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-docker-clean.ps1"
echo.
pause
