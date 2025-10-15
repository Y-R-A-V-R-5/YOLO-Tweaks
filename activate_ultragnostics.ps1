<#
.SYNOPSIS
    Activates the 'ultragnostics' Conda environment.

.DESCRIPTION
    This script is designed to activate a specific Conda environment named
    'ultragnostics' located at a non-standard path. This is useful for
    ensuring the correct environment is loaded for YOLOv8 projects.

.NOTES
    Requires Conda to be initialized in your PowerShell profile.
    The path is hardcoded based on the user's request.
#>

# --- Configuration ---
$CondaEnvName = "ultragnostics"
$CondaEnvPath = "C:\Users\saket\anaconda3\envs\ultragnostics"

# 1. Check if the environment directory exists
if (-not (Test-Path -Path $CondaEnvPath -PathType Container)) {
    Write-Error "Conda environment directory not found at: $CondaEnvPath"
    Write-Error "Please ensure the 'ultragnostics' environment is created."
    exit 1
}

# 2. Check for the 'conda' command. If Conda is not initialized, 'conda' might not be recognized.
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Warning "The 'conda' command was not found."
    Write-Warning "Attempting to initialize the Conda base environment."
    
    # Attempt to use the Conda path you provided to run the base activation script
    $CondaInitScript = "C:\Users\saket\anaconda3\Scripts\conda.exe"
    if (Test-Path -Path $CondaInitScript) {
        # This initializes the Conda functionality in the current session
        & "C:\Users\saket\anaconda3\Scripts\conda.exe" "init" "powershell"
    } else {
        Write-Error "Cannot find conda.exe. Please run 'conda init powershell' manually in your terminal first."
        exit 1
    }
}

# 3. Activate the Conda environment by its name
Write-Host "Activating Conda environment: $CondaEnvName at $CondaEnvPath..."

# Use the standard 'conda activate' command. Conda is smart enough to find the environment
# using its name, which is safer than manipulating environment variables directly.
conda activate $CondaEnvName

# 4. Verification (Optional)
if ($env:CONDA_DEFAULT_ENV -eq $CondaEnvName) {
    Write-Host "Successfully activated $CondaEnvName. You are ready to start the ultragnostics timepass!" -ForegroundColor Green
} else {
    Write-Error "Activation failed. Current environment: $($env:CONDA_DEFAULT_ENV)"
    exit 1
}

# NOTE: When running this as a script in VS Code or PowerShell ISE, the changes to the
# environment usually only apply to the script's scope. To see the environment activated
# in your *current* terminal session, you should typically use the dot-sourcing method:
# . .\activate_ultragnostics.ps1