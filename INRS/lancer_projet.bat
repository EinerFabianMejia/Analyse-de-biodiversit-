@echo off
cd /d "%~dp0"
start "" py -3.11 server.py
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8001
