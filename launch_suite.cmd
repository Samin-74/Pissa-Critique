@echo off
setlocal
if "%HF_HOME%"=="" set "HF_HOME=%~dp0hf_cache"
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_suite.ps1" %*
