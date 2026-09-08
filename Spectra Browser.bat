@echo off
rem Double-click to start the local spectra browser (uses scripts\spectra-browser.ps1).
powershell -NoLogo -ExecutionPolicy Bypass -File "%~dp0scripts\spectra-browser.ps1" %*
if errorlevel 1 pause
