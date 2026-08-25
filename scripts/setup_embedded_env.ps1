param(
    [string]$TargetDir = "$PSScriptRoot\..\python_embeded",
    [string]$PythonVersion = "3.11.9",
    [string]$CudaVersion = "cu121"
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RootDir = Split-Path -Parent $PSScriptRoot
$EmbedZipUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$ZipFile = "$RootDir\python_embed_temp.zip"
$GetPipFile = "$RootDir\get-pip.py"

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  AVENOX AUDIO STUDIO - ISOLATED EMBEDDED PYTHON ENVIRONMENT SETUP     " -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Lokasi Target: $TargetDir" -ForegroundColor Yellow

if (Test-Path "$TargetDir\python.exe") {
    Write-Host "[OK] python_embeded sudah terpasang. Memeriksa dependensi..." -ForegroundColor Green
} else {
    Write-Host "[1/5] Mengunduh Python $PythonVersion Embedded x64..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $EmbedZipUrl -OutFile $ZipFile -UseBasicParsing

    Write-Host "[2/5] Mengekstrak ke $TargetDir..." -ForegroundColor Yellow
    if (!(Test-Path $TargetDir)) {
        New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    }
    Expand-Archive -Path $ZipFile -DestinationPath $TargetDir -Force
    Remove-Item -Path $ZipFile -Force -ErrorAction SilentlyContinue

    Write-Host "[3/5] Mengonfigurasi python311._pth untuk mengaktifkan import site..." -ForegroundColor Yellow
    $pthContent = @"
python311.zip
.
..
../src
Lib/site-packages
import site
"@
    Set-Content -Path "$TargetDir\python311._pth" -Value $pthContent -Encoding ASCII

    Write-Host "[4/5] Mengunduh & Memasang pip..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipFile -UseBasicParsing
    & "$TargetDir\python.exe" $GetPipFile --no-warn-script-location
    Remove-Item -Path $GetPipFile -Force -ErrorAction SilentlyContinue
}

Write-Host "[5/5] Memasang PyTorch CUDA ($CudaVersion) & Library Audio DSP..." -ForegroundColor Yellow
$PythonExe = "$TargetDir\python.exe"

# Upgrade pip & wheel
& $PythonExe -m pip install --upgrade pip setuptools wheel --no-warn-script-location

# Install PyTorch with CUDA
Write-Host "Memasang PyTorch CUDA 12.1..." -ForegroundColor Yellow
& $PythonExe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/$CudaVersion --no-warn-script-location

# Install requirements
Write-Host "Memasang dependensi requirements.txt..." -ForegroundColor Yellow
& $PythonExe -m pip install -r "$RootDir\requirements.txt" --no-warn-script-location

Write-Host ""
Write-Host "=======================================================================" -ForegroundColor Green
Write-Host "  SETUP BERHASIL! Avenox Embedded Environment siap digunakan.          " -ForegroundColor Green
Write-Host "  Jalankan 'run_avenox.bat' untuk membuka aplikasi Desktop.            " -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Green
