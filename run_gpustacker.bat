@echo off
cd /d "%~dp0"
py -3 run_gpustacker.py
if errorlevel 1 pause
