# Windows equivalent of setup.sh.
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"

$VenvDir = Join-Path $HOME ".venvs\codebase_qna_rag"

Write-Host "==> Creating virtual environment at $VenvDir"
python -m venv $VenvDir

Write-Host "==> Activating virtual environment"
& (Join-Path $VenvDir "Scripts\Activate.ps1")

Write-Host "==> Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "==> Creating .env from .env.example"
        Copy-Item ".env.example" ".env"
    } else {
        Write-Host "==> Creating .env"
        "NEBIUS_API_KEY=`nGITHUB_TOKEN=" | Out-File -FilePath ".env" -Encoding utf8
    }
} else {
    Write-Host "==> .env already exists, leaving it untouched"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "1. Fill in your secrets in .env"
Write-Host "2. Activate the venv:  $VenvDir\Scripts\Activate.ps1"
Write-Host "3. Run the app:        streamlit run main.py"
Write-Host ""
Write-Host "(You can also skip .env entirely and paste both keys into the app sidebar.)"
