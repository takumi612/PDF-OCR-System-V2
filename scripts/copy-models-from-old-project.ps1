param(
  [string]$OldProject = "D:\KeySoft\OCR-System"
)
$ErrorActionPreference = "Stop"
$source = Join-Path $OldProject "models\vietocr\vgg_seq2seq.pth"
$destination = Join-Path (Get-Location) "models\vietocr\vgg_seq2seq.pth"
$destinationDirectory = Split-Path -Parent $destination
New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
if (-not (Test-Path $source)) {
  throw "Không tìm thấy $source"
}
Copy-Item $source $destination -Force
Write-Host "Đã sao chép trọng số tới $destination"
