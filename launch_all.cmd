@echo off
setlocal
if "%HF_HOME%"=="" set "HF_HOME=%~dp0hf_cache"
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" > "%~dp0outputs\logs\suite_master.log" 2>&1
