$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $PSScriptRoot
$server = Join-Path $project ".runtime\llama.cpp\llama-server.exe"
$python = Join-Path $project ".venv\Scripts\python.exe"
$modelScript = Join-Path $PSScriptRoot "download_model.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "The virtual environment is missing. Follow the README setup steps first."
}

$model = & $python $modelScript --print-path
if ($LASTEXITCODE -ne 0) {
    throw "Could not resolve the configured model path."
}

if (-not (Test-Path -LiteralPath $server)) {
    throw "llama-server.exe is missing from $server"
}

if (-not (Test-Path -LiteralPath $model)) {
    throw "The configured model is missing at $model. Run the model download script."
}

& $server `
    -m $model `
    --host 127.0.0.1 `
    --port 8080 `
    -c 8192 `
    -ngl all `
    -fa auto `
    -ctk q8_0 `
    -ctv q8_0 `
    --parallel 1 `
    --metrics

exit $LASTEXITCODE
