$source = "dataset/trig_log"
$maxFiles = 2000

Get-ChildItem -Directory $source | ForEach-Object {
    $subfolder = $_.FullName
    $files = Get-ChildItem -Path $subfolder -File | Sort-Object { Get-Random }  

    if ($files.Count -gt $maxFiles) {
        $filesToDelete = $files | Select-Object -Skip $maxFiles
        $filesToDelete | ForEach-Object {
            Remove-Item -Path $_.FullName
        }
        Write-Output "Deleted $($filesToDelete.Count) files from $subfolder."
    } else {
        Write-Output "No deletion needed for $subfolder, it has $($files.Count) files."
    }
}
