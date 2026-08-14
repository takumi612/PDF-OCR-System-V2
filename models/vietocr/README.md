# VietOCR assets

`vgg_seq2seq.yml` đã được kèm theo. Trọng số `vgg_seq2seq.pth` không được đóng gói
trong ZIP vì kích thước lớn. Sao chép file từ dự án gốc:

```powershell
Copy-Item "D:\KeySoft\OCR-System\models\vietocr\vgg_seq2seq.pth" `
  ".\models\vietocr\vgg_seq2seq.pth"
```

Có thể dùng `scripts/copy-models-from-old-project.ps1`.
