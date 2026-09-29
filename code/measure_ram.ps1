param(
    [string]$Checkpoint = 'runs/student-deep-16000/checkpoint.pt'
)

$ErrorActionPreference = 'Stop'
$pythonExe = (Get-Command python).Source
$arguments = @('evaluate.py', '--checkpoint', $Checkpoint, '--split', 'test',
               '--device', 'cpu', '--precision', 'fp32',
               '--output', 'runs/student-deep-16000/test_ram_measurement.json')
$runner = Start-Process -FilePath $pythonExe -ArgumentList $arguments -PassThru -NoNewWindow
$peakBytes = [long]0
$peakDetails = @()
$seenChild = $false

while ($true) {
    $all = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'")
    $ids = @([int]$runner.Id)
    # Repeatedly add descendants: venv redirectors can launch another Python.
    do {
        $oldCount = $ids.Count
        foreach ($proc in $all) {
            if (($ids -contains [int]$proc.ParentProcessId) -and
                ($ids -notcontains [int]$proc.ProcessId)) {
                $ids += [int]$proc.ProcessId
            }
        }
    } while ($ids.Count -gt $oldCount)

    $family = @($all | Where-Object { $ids -contains [int]$_.ProcessId })
    if ($family.Count -gt 1) { $seenChild = $true }
    $currentBytes = [long]0
    foreach ($proc in $family) { $currentBytes += [long]$proc.WorkingSetSize }
    if ($currentBytes -gt $peakBytes) {
        $peakBytes = $currentBytes
        $peakDetails = @($family | ForEach-Object {
            'PID {0}: {1:N1} MiB' -f $_.ProcessId, ([long]$_.WorkingSetSize / 1MB)
        })
    }
    $runner.Refresh()
    if ($runner.HasExited) { break }
    Start-Sleep -Milliseconds 150
}

'Peak process-tree RAM: {0:N3} GiB ({1} bytes)' -f ($peakBytes / 1GB), $peakBytes
'Processes at peak: ' + ($peakDetails -join '; ')
if ($peakBytes -lt 64MB) {
    'WARNING: Measurement is implausibly low; do not use it in the report.'
}
if ($runner.ExitCode -ne 0) { throw "Evaluator exited with code $($runner.ExitCode)" }
