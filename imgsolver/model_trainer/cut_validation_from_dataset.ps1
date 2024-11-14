$sourcePath = Read-Host -Prompt "Enter the source folder path"
$targetPath = Read-Host -Prompt "Enter the target folder path"

function Copy-FolderStructureWithTrimming {
    param (
        [Parameter(Mandatory=$true)]
        [string]$SourcePath,
        [Parameter(Mandatory=$true)]
        [string]$TargetPath
    )

    # Create target folder structure
    $targetFolderName = Split-Path $SourcePath -Leaf
    $targetPath = Join-Path $TargetPath $targetFolderName
    if (!(Test-Path -Path $targetPath)) {
        New-Item -ItemType Directory -Path $targetPath | Out-Null
    }

    # Process files in the current folder
    $files = Get-ChildItem -Path $SourcePath -File
    $trimCount = [int]($files.Count * 0.15)
    if ($trimCount -gt 0) {
        $filesToCopy = $files | Select-Object -First ($files.Count - $trimCount)
        foreach ($file in $filesToCopy) {
            $sourceFilePath = Join-Path $SourcePath $file.Name
            $targetFilePath = Join-Path $targetPath $file.Name
            Copy-Item -Path $sourceFilePath -Destination $targetFilePath
        }
    }
    else {
        Copy-Item -Path (Join-Path $SourcePath "*") -Destination $targetPath -Recurse
    }

    # Process subfolders recursively
    $subFolders = Get-ChildItem -Path $SourcePath -Directory
    foreach ($subfolder in $subFolders) {
        $subSourcePath = Join-Path $SourcePath $subfolder.Name
        $subTargetPath = Join-Path $targetPath $subfolder.Name
        Copy-FolderStructureWithTrimming -SourcePath $subSourcePath -TargetPath $targetPath
    }
}

Copy-FolderStructureWithTrimming -SourcePath $sourcePath -TargetPath $targetPath