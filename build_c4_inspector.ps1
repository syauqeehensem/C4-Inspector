param(
    [string]$Python = (Join-Path $PSScriptRoot '.c4-runtime-venv\Scripts\python.exe'),
    [string]$DistRoot = (Join-Path $PSScriptRoot 'release'),
    [string]$VCRuntimeDirectory = (Join-Path $env:WINDIR 'System32')
)
$ErrorActionPreference = 'Stop'
$destination = Join-Path $DistRoot 'C4Inspector'
if (Test-Path $destination) { throw "Build destination already exists: $destination. Choose a new DistRoot." }
if (-not (Test-Path $Python)) { throw 'Create the CPU-only environment described in C4_INSPECTOR_README.md first.' }
if (-not (Test-Path (Join-Path $PSScriptRoot 'c4_model\metadata.json'))) { throw 'Export and verify the C4 model first.' }
& $Python -c "import torch; assert torch.version.cuda is None, 'Build requires CPU-only PyTorch'"
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
Copy-Item -Recurse (Join-Path $PSScriptRoot 'c4_model') (Join-Path $destination 'c4_model')
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