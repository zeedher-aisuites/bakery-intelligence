
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot ".env"
$settingsPath = Join-Path $env:USERPROFILE ".config\muse\settings.json"

if (-not (Test-Path $envPath)) {
    throw "Project .env file not found."
}

if (-not (Test-Path $settingsPath)) {
    throw "Muse Code settings.json not found."
}

# Read the Baseten API key without displaying it.
$keyLine = Get-Content $envPath |
    Where-Object { $_ -match '^\s*BASETEN_API_KEY\s*=' } |
    Select-Object -First 1

if (-not $keyLine) {
    throw "BASETEN_API_KEY not found in .env."
}

$apiKey = ($keyLine -split '=', 2)[1].Trim().Trim('"', "'")

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "BASETEN_API_KEY is empty."
}

# Preserve the original configuration and environment.
$originalSettings = [System.IO.File]::ReadAllBytes($settingsPath)
$previousApiKey = [Environment]::GetEnvironmentVariable("META_API_KEY", "Process")

try {
    $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json

    $settings.model = "meta-models/muse-spark-open-1026"

    $settings | Add-Member -NotePropertyName endpoint_transport -NotePropertyValue @{
        base_url = "https://inference.baseten.co/v1"
        auth = "bearer"
    } -Force

    $json = $settings | ConvertTo-Json -Depth 20

    [System.IO.File]::WriteAllText(
        $settingsPath,
        $json,
        [System.Text.UTF8Encoding]::new($false)
    )

    $env:META_API_KEY = $apiKey

    Write-Host "Starting Muse Code with Baseten..."

    Push-Location $projectRoot
    try {
        muse --approval-mode on-request
    }
    finally {
        Pop-Location
    }
}
finally {
    # Restore the exact original settings bytes.
    [System.IO.File]::WriteAllBytes($settingsPath, $originalSettings)

    if ($null -eq $previousApiKey) {
        Remove-Item Env:META_API_KEY -ErrorAction SilentlyContinue
    }
    else {
        $env:META_API_KEY = $previousApiKey
    }

    Write-Host "Original Muse Code settings restored."
}
