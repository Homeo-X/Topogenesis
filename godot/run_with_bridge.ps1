param(
    [string]$GodotPath = "C:\Users\rsijr\Downloads\Godot_v4.6.2-stable_win64.exe\Godot_v4.6.2-stable_win64_console.exe",
    [int]$BridgePort = 8765
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[topogenesis] starting Python cognition bridge on port $BridgePort"
$bridge = Start-Process `
    -FilePath "python" `
    -ArgumentList @("-m", "topogenesis.game_bridge.server", "--port", "$BridgePort") `
    -WorkingDirectory $repoRoot `
    -PassThru `
    -WindowStyle Hidden

try {
    Start-Sleep -Milliseconds 700
    Write-Host "[topogenesis] launching Godot with bridge process $($bridge.Id)"
    & $GodotPath --path $PSScriptRoot
}
finally {
    if ($bridge -and -not $bridge.HasExited) {
        Write-Host "[topogenesis] stopping Python cognition bridge"
        Stop-Process -Id $bridge.Id
    }
}
