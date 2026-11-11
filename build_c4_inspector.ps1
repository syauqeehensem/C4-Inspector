param(
    [string]$Python = (Join-Path $PSScriptRoot '.c4-runtime-venv\Scripts\python.exe'),
    [string]$DistRoot = (Join-Path $PSScriptRoot 'release'),
    [string]$ModelDirectory = (Join-Path $PSScriptRoot 'models'),
    [string]$VCRuntimeDirectory = (Join-Path $env:WINDIR 'System32')
)
$ErrorActionPreference = 'Stop'
$destination = Join-Path $DistRoot 'C4Inspector'
if (Test-Path $destination) { throw "Build destination already exists: $destination. Choose a new DistRoot." }
if (-not (Test-Path $Python)) { throw 'Create the CPU-only environment described in C4_INSPECTOR_README.md first.' }
if (-not (Test-Path (Join-Path $ModelDirectory 'products.json'))) { throw 'Supply the verified multi-product model bundle first.' }
$ModelDirectory = (Resolve-Path $ModelDirectory).Path
Push-Location $PSScriptRoot
try {
    & $Python -c @'
import sys
from pathlib import Path
import cv2
import torch
import zxingcpp
from c4_runtime import ProductPredictor
assert torch.version.cuda is None, 'Build requires CPU-only PyTorch'
bundle = Path(sys.argv[1])
predictor = ProductPredictor(bundle)
referenced = {'products.json'}
for product in predictor.metadata['products']:
    referenced.add(product['alignment'])
    for area in product['areas']:
        referenced.update({area['model'] + '/patchcore.pt', area['model'] + '/metadata.json'})
actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob('*') if path.is_file()}
assert actual == referenced, 'Unexpected or missing bundle files: ' + ', '.join(sorted(actual ^ referenced))
'@ $ModelDirectory
} finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw 'CPU environment check failed.' }
$runtimeFiles = foreach ($runtimeName in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcruntime140_threads.dll')) {
    $runtimePath = Join-Path $VCRuntimeDirectory $runtimeName
    if (-not (Test-Path $runtimePath -PathType Leaf)) { throw "Missing build-time runtime: $runtimePath. Install the current Microsoft Visual C++ x64 redistributable on the build PC." }
    $signature = Get-AuthenticodeSignature $runtimePath
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw "Runtime must have a valid Microsoft signature: $runtimePath"
    }
    $runtimePath
}
$work = Join-Path $PSScriptRoot '.c4-build'
New-Item -ItemType Directory -Force -Path $work | Out-Null
& $Python -m PyInstaller --noconfirm --clean --windowed --onedir --name C4Inspector --noupx `
    --distpath $DistRoot --workpath (Join-Path $work 'work') --specpath $work `
    --exclude-module anomalib --exclude-module torchvision --exclude-module lightning `
    --exclude-module matplotlib --exclude-module pytest --exclude-module IPython `
    (Join-Path $PSScriptRoot 'C4Inspector.py')
if ($LASTEXITCODE -ne 0) { throw 'Executable build failed.' }
Copy-Item -Path $runtimeFiles -Destination (Join-Path $destination '_internal') -Force
Copy-Item -Recurse $ModelDirectory (Join-Path $destination 'models')
Copy-Item (Join-Path $PSScriptRoot 'C4_INSPECTOR_README.md') $destination
& $Python -m pip freeze | Set-Content -Encoding UTF8 (Join-Path $destination 'runtime-installed.txt')
if ($LASTEXITCODE -ne 0) { throw 'Environment inventory failed.' }
$manifest = Get-ChildItem $destination -File -Recurse | ForEach-Object {
    [PSCustomObject]@{
        path = $_.FullName.Substring($destination.Length + 1).Replace('\', '/')
        bytes = $_.Length
        sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $destination 'MANIFEST.json')
Compress-Archive -Path $destination -DestinationPath ($destination + '.zip') -CompressionLevel Optimal
Write-Output "Portable folder: $destination"
Write-Output "Archive: $destination.zip"