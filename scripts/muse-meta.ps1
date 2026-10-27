
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot ".env"

if (-not (Test-Path $envPath)) {
    throw "Project .env file not found."
}

# Read the Meta API key without displaying it.
$keyLine = Get-Content $envPath |
    Where-Object { $_ -match '^\s*MODEL_API_KEY\s*=' } |
    Select-Object -First 1

if (-not $keyLine) {
    throw "MODEL_API_KEY not found in .env."
}

$apiKey = ($keyLine -split '=', 2)[1].Trim().Trim('"', "'")

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "MODEL_API_KEY is empty."
}

$previousApiKey = [Environment]::GetEnvironmentVariable("META_API_KEY", "Process")

try {
    $env:META_API_KEY = $apiKey

    Write-Host "Starting Muse Code with Meta..."

    Push-Location $projectRoot
    try {
        muse --provider meta --model muse-spark-1.3-contributor --approval-mode on-request
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $previousApiKey) {
        Remove-Item Env:META_API_KEY -ErrorAction SilentlyContinue
    }
    else {
        $env:META_API_KEY = $previousApiKey
    }

    Write-Host "Original environment restored."
}
