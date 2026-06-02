# ============================================================
# PATH Cleanup Script — Deduplicate User vs System PATH
# ============================================================

$ErrorActionPreference = "Stop"

# ---- Helper: normalize a path entry ----
function Normalize-PathEntry($entry) {
    $e = $entry.Trim()
    $e = $e -replace '[/\\]+$', ''    # remove trailing slashes
    $e = $e -replace '/', '\'          # normalize to backslash
    return $e
}

# ---- Read current PATH ----
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " READING CURRENT PATH FROM REGISTRY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$userPathRaw  = [Environment]::GetEnvironmentVariable('PATH', 'User')
$sysPathRaw   = [Environment]::GetEnvironmentVariable('PATH', 'Machine')

$userEntries  = $userPathRaw -split ';' | Where-Object { $_.Trim() -ne '' }
$sysEntries   = $sysPathRaw -split ';' | Where-Object { $_.Trim() -ne '' }

$origUserCount = $userEntries.Count
$origSysCount  = $sysEntries.Count
$origUserLen   = $userPathRaw.Length
$origSysLen    = $sysPathRaw.Length

Write-Host "User  PATH: $origUserCount entries, $origUserLen chars"
Write-Host "System PATH: $origSysCount entries, $origSysLen chars"
Write-Host "Combined:   $($origUserLen + $origSysLen + 1) chars"

# ---- Build normalized lookup sets ----
# For System PATH: normalized -> original form mapping (keep first occurrence)
$sysNormSet = [ordered]@{}
foreach ($entry in $sysEntries) {
    $norm = Normalize-PathEntry $entry
    if (-not $sysNormSet.Contains($norm)) {
        $sysNormSet[$norm] = $entry
    }
}

# ---- Analysis ----
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " ANALYSIS: ENTRIES TO REMOVE FROM USER PATH" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "(These already exist in System PATH)`n" -ForegroundColor Yellow

$toRemove = @()
$toKeep   = @()
$removalMap = @{}  # user entry -> matching system entry

foreach ($entry in $userEntries) {
    $norm = Normalize-PathEntry $entry
    if ($sysNormSet.Contains($norm)) {
        $toRemove += $entry
        $removalMap[$entry] = $sysNormSet[$norm]
        Write-Host "  [REMOVE] $entry" -ForegroundColor Red
        Write-Host "       -> already in System: $($sysNormSet[$norm])" -ForegroundColor DarkGray
    } else {
        $toKeep += $entry
    }
}

Write-Host "`nEntries to remove: $($toRemove.Count)"
Write-Host "Entries to keep:    $($toKeep.Count)"

# ---- Also check for self-duplicates in User PATH ----
$seen = @{}
$userSelfDupes = @()
$userCleaned = @()
foreach ($entry in $toKeep) {
    $norm = Normalize-PathEntry $entry
    if (-not $seen.ContainsKey($norm)) {
        $seen[$norm] = $entry
        $userCleaned += $entry
    } else {
        $userSelfDupes += $entry
        Write-Host "  [SELF-DUPE] $entry" -ForegroundColor Magenta
    }
}

if ($userSelfDupes.Count -gt 0) {
    Write-Host "`nAdditional self-duplicates within User PATH: $($userSelfDupes.Count)" -ForegroundColor Magenta
    $toKeep = $userCleaned
}

# ---- New PATH ----
$newUserPath = ($toKeep -join ';')
$newUserLen  = $newUserPath.Length
$newCombined = $newUserLen + $origSysLen + 1

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "User  PATH: $origUserCount -> $($toKeep.Count) entries  (-$($toRemove.Count + $userSelfDupes.Count))"
Write-Host "User  PATH: $origUserLen -> $newUserLen chars  (-$($origUserLen - $newUserLen))"
Write-Host "System PATH: $origSysCount entries (unchanged)"
Write-Host "Combined:   $($origUserLen + $origSysLen + 1) -> $newCombined chars"
Write-Host ""
if ($newCombined -le 2047) {
    Write-Host "*** Combined PATH within 2047 limit ***" -ForegroundColor Green
} else {
    Write-Host "*** WARNING: Combined PATH still over 2047 ($newCombined chars) ***" -ForegroundColor Yellow
}

# ---- Confirmation ----
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " APPLY CHANGES?" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "This will permanently modify the USER PATH in the Windows registry."
Write-Host "A backup will be saved to: $env:USERPROFILE\Desktop\PATH_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
$choice = Read-Host "`nType YES to apply changes, anything else to abort"

if ($choice -ne "YES") {
    Write-Host "Aborted. No changes made." -ForegroundColor Yellow
    exit 0
}

# ---- Backup ----
$backupFile = "$env:USERPROFILE\Desktop\PATH_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
@"
PATH Backup — $(Get-Date)
==========================

--- OLD User PATH ---
$userPathRaw

--- OLD System PATH ---
$sysPathRaw

--- Removed from User PATH (already in System) ---
$($toRemove -join "`n")
"@ | Out-File -FilePath $backupFile -Encoding UTF8
Write-Host "`nBackup saved to: $backupFile" -ForegroundColor Green

# ---- Apply ----
[Environment]::SetEnvironmentVariable('PATH', $newUserPath, 'User')

# ---- Verify ----
$verifyPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
$verifyEntries = $verifyPath -split ';' | Where-Object { $_.Trim() -ne '' }
Write-Host "`nVerification: User PATH now has $($verifyEntries.Count) entries, $($verifyPath.Length) chars" -ForegroundColor Green
Write-Host "Done! You should restart your terminal for changes to fully take effect." -ForegroundColor Green
