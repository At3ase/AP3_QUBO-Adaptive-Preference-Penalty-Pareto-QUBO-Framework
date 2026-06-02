# Check PATH script - read only, no changes
Write-Host "===== USER PATH =====" -ForegroundColor Cyan
$userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
$userEntries = $userPath -split ';' | Where-Object { $_ -ne '' }
$i = 1
foreach ($entry in $userEntries) {
    Write-Host "$i. $entry"
    $i++
}
Write-Host "`nUser PATH count: $($userEntries.Count)" -ForegroundColor Yellow
Write-Host "User PATH length: $($userPath.Length) chars" -ForegroundColor Yellow

Write-Host "`n===== SYSTEM PATH =====" -ForegroundColor Cyan
$sysPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
$sysEntries = $sysPath -split ';' | Where-Object { $_ -ne '' }
$j = 1
foreach ($entry in $sysEntries) {
    Write-Host "$j. $entry"
    $j++
}
Write-Host "`nSystem PATH count: $($sysEntries.Count)" -ForegroundColor Yellow
Write-Host "System PATH length: $($sysPath.Length) chars" -ForegroundColor Yellow

Write-Host "`n===== COMBINED =====" -ForegroundColor Cyan
$fullPath = "$userPath;$sysPath"
Write-Host "Combined length: $($fullPath.Length) chars" -ForegroundColor Magenta

# Show duplicates
$allEntries = @($userEntries) + @($sysEntries)
$dupes = $allEntries | Group-Object | Where-Object { $_.Count -gt 1 }
if ($dupes.Count -gt 0) {
    Write-Host "`n===== DUPLICATE ENTRIES =====" -ForegroundColor Red
    foreach ($d in $dupes) {
        Write-Host "Appears $($d.Count)x: $($d.Name)"
    }
} else {
    Write-Host "`nNo duplicates found." -ForegroundColor Green
}
