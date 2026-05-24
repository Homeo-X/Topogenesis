param(
    [string]$GodotPath = "C:\Users\rsijr\Downloads\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64.exe",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [switch]$NoWait,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$GodotProject = Join-Path $RepoRoot "godot"
$HealthUrl = "http://$HostName`:$Port/health"

function Test-BridgeReady {
    try {
        $response = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 1
        return [bool]$response.ok
    }
    catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $GodotPath)) {
    throw "Godot executable was not found at: $GodotPath"
}

if (-not (Test-Path -LiteralPath $GodotProject)) {
    throw "Godot project folder was not found at: $GodotProject"
}

if ($ValidateOnly) {
    Write-Host "[Topogenesis] Launcher validation passed."
    exit 0
}

$bridgeProcess = $null
$startedBridge = $false

if (Test-BridgeReady) {
    Write-Host "[Topogenesis] Bridge already running at $HealthUrl"
}
else {
    Write-Host "[Topogenesis] Starting Python bridge..."
    $bridgeProcess = Start-Process -FilePath "python" `
        -ArgumentList @("-m", "topogenesis.game_bridge.server", "--host", $HostName, "--port", "$Port") `
        -WorkingDirectory $RepoRoot `
        -PassThru `
        -WindowStyle Minimized
    $startedBridge = $true

    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 350
        if (Test-BridgeReady) {
            $ready = $true
            break
        }
        if ($bridgeProcess.HasExited) {
            throw "Python bridge exited before it became ready."
        }
    }
    if (-not $ready) {
        throw "Python bridge did not become ready at $HealthUrl"
    }
}

Write-Host "[Topogenesis] Opening Godot..."
$godotProcess = Start-Process -FilePath $GodotPath `
    -ArgumentList @("--path", $GodotProject) `
    -WorkingDirectory $RepoRoot `
    -PassThru

if ($NoWait) {
    Write-Host "[Topogenesis] Godot launched. Bridge remains running in the background."
    exit 0
}

try {
    Wait-Process -Id $godotProcess.Id
}
finally {
    if ($startedBridge -and $bridgeProcess -ne $null -and -not $bridgeProcess.HasExited) {
        Write-Host "[Topogenesis] Stopping Python bridge..."
        Stop-Process -Id $bridgeProcess.Id -Force
    }
}
