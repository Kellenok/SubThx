# Build the release: two executables, subthx (CLI and server) and subthx-gui (the desktop app).
# usage: powershell -ExecutionPolicy Bypass -File tools/package.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

cargo build --release --bin subthx
if ($LASTEXITCODE) { throw "cargo build failed" }

Push-Location app
npx tauri build --no-bundle
if ($LASTEXITCODE) { throw "tauri build failed" }
Pop-Location

$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force $dist | Out-Null
Copy-Item (Join-Path $root "target/release/subthx.exe") (Join-Path $dist "subthx.exe") -Force
Copy-Item (Join-Path $root "target/release/subthx-app.exe") (Join-Path $dist "subthx-gui.exe") -Force
# ONNX Runtime loads its execution providers as separate DLLs next to the executables
# (CUDA and DirectML are used; TensorRT is not)
foreach ($dll in "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll", "DirectML.dll") {
    Copy-Item (Join-Path $root "target/release/$dll") (Join-Path $dist $dll) -Force
}

$sums = Get-ChildItem (Join-Path $dist "*") -Include "*.exe", "*.dll" | ForEach-Object {
    "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name
}
$sums | Set-Content (Join-Path $dist "SHA256SUMS.txt") -Encoding ascii

# one archive with everything: the two executables, the ONNX Runtime DLLs and the sums
$version = (Select-String -Path (Join-Path $root "Cargo.toml") -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
$zip = Join-Path $dist "subthx-$version-windows-x64.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $dist "subthx.exe"), (Join-Path $dist "subthx-gui.exe"), (Join-Path $dist "*.dll"), (Join-Path $dist "SHA256SUMS.txt") -DestinationPath $zip
Get-ChildItem $dist | Select-Object Name, Length | Format-Table -AutoSize
$sums
Write-Output "archive: $zip"
