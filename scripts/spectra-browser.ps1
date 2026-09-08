<#
Launch the local spectra browser on the instrument computer using a micromamba environment.

Usage (PowerShell, from anywhere):
    .\scripts\spectra-browser.ps1            # start the app
    .\scripts\spectra-browser.ps1 -Install   # install/update packages first, then start

Environment name defaults to py314; override with -Env or $env:SPECTRA_ENV.
The app is bound to 127.0.0.1 and only reads/writes local files.
#>
param(
    [string]$Env = $(if ($env:SPECTRA_ENV) { $env:SPECTRA_ENV } else { 'py314' }),
    [switch]$Install
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Find micromamba: on PATH, via $MAMBA_EXE, or in the default per-user install location.
$mamba = Get-Command micromamba -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $mamba -and $env:MAMBA_EXE) { $mamba = $env:MAMBA_EXE }
if (-not $mamba) { $mamba = Join-Path $env:LOCALAPPDATA 'micromamba\micromamba.exe' }
if (-not (Test-Path $mamba)) {
    throw "micromamba not found. Install it or set `$env:MAMBA_EXE to micromamba.exe."
}

if ($Install) {
    & $mamba run -n $Env python -m pip install -r spectra-browser-requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
}

& $mamba run -n $Env python -m streamlit run pchemapps\spectra_browser.py `
    --server.address 127.0.0.1 --server.headless false --browser.gatherUsageStats false
