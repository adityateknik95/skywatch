<#
  Cleanly start Docker Desktop on this machine.

  Works around a Docker Desktop bug that crashes startup when the Windows
  username contains a space (here "ADITYA S G"): its services leave behind
  AF_UNIX socket files that Windows can no longer access, and those zombies
  break the *next* boot. This script quarantines the socket directories so
  Docker recreates them clean, then launches it and brings up the SkyWatch DB.

  Safe to run anytime: if Docker is already healthy it does nothing but make
  sure the database container is up.

  Usage:  right-click -> Run with PowerShell, or run the .bat next to the repo,
          or:  powershell -ExecutionPolicy Bypass -File scripts\start-docker-clean.ps1
#>
[CmdletBinding()]
param(
    [switch]$NoCompose  # skip starting the SkyWatch database
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot          # scripts\ -> repo root
$composeFile = Join-Path $projectRoot 'docker-compose.yml'
$dockerExe   = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
$localDocker = Join-Path $env:LOCALAPPDATA 'Docker'

function Test-DockerUp {
    try { docker info *> $null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

if (Test-DockerUp) {
    Write-Host "Docker engine is already running." -ForegroundColor Green
}
else {
    Write-Host "Docker not responding - performing a clean start..." -ForegroundColor Yellow

    # 1) Stop any half-dead Docker processes from a previous/failed session.
    Get-Process -Name 'Docker Desktop','com.docker.backend','com.docker.build',
                      'com.docker.dev-envs','com.docker.extensions','vpnkit' `
        -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    cmd /c "wsl --terminate docker-desktop" 2>$null | Out-Null

    # 2) Quarantine socket directories that still contain leftover (zombie) files
    #    so Docker recreates them clean. Renaming only touches the directory
    #    entry, not the un-deletable children.
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $socketDirs = @(
        (Join-Path $localDocker 'run'),
        (Join-Path $env:LOCALAPPDATA 'docker-secrets-engine')
    )
    foreach ($d in $socketDirs) {
        if (Test-Path -LiteralPath $d) {
            $leftover = Get-ChildItem -LiteralPath $d -Force -ErrorAction SilentlyContinue
            if ($leftover) {
                try {
                    Rename-Item -LiteralPath $d -NewName ("$(Split-Path $d -Leaf).broken-$stamp") -ErrorAction Stop
                    Write-Host "  quarantined $(Split-Path $d -Leaf) (had leftover sockets)"
                } catch {
                    Write-Warning "  could not move $d aside: $($_.Exception.Message)"
                }
            }
        }
    }

    # 3) Best-effort prune of quarantine dirs older than 7 days (zombie files
    #    inside may refuse deletion - that's fine, just skip them).
    foreach ($base in @($localDocker, $env:LOCALAPPDATA)) {
        Get-ChildItem -LiteralPath $base -Directory -Filter '*.broken-*' -ErrorAction SilentlyContinue |
            Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
            ForEach-Object { try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop } catch {} }
    }

    # 4) Launch Docker Desktop and wait for the engine.
    if (-not (Test-Path -LiteralPath $dockerExe)) {
        Write-Error "Docker Desktop not found at: $dockerExe"; exit 1
    }
    Write-Host "Launching Docker Desktop..."
    Start-Process -FilePath $dockerExe

    $ready = $false
    for ($i = 0; $i -lt 75; $i++) {
        if (Test-DockerUp) { $ready = $true; break }
        Start-Sleep -Seconds 4
    }
    if (-not $ready) {
        Write-Error "Docker engine did not come up in time. Open Docker Desktop and check for an error dialog."
        exit 1
    }
    Write-Host "Docker engine is up." -ForegroundColor Green
}

# Bring up the SkyWatch database (idempotent).
if (-not $NoCompose -and (Test-Path -LiteralPath $composeFile)) {
    Write-Host "Ensuring the SkyWatch database is up..."
    docker compose -f $composeFile up -d | Out-Host
    docker compose -f $composeFile ps | Out-Host
}

Write-Host "`nReady. You can now run the collector." -ForegroundColor Green
