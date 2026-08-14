const fileInput = document.querySelector("#pdfFile");
const extractButton = document.querySelector("#extractButton");
const downloadButton = document.querySelector("#downloadButton");
const aiSafeDownloadButton = document.querySelector("#aiSafeDownloadButton");
const auditDownloadButton = document.querySelector("#auditDownloadButton");
const pdfViewer = document.querySelector("#pdfViewer");
const resultText = document.querySelector("#resultText");
const status = document.querySelector("#status");
const summary = document.querySelector("#summary");
const riskReview = document.querySelector("#riskReview");
const riskList = document.querySelector("#riskList");
const riskMessage = document.querySelector("#riskMessage");
const tabs = [...document.querySelectorAll(".tab")];

let pdfUrl = null;
let result = null;
let format = "text";

function showResult() {
  resultText.value = result ? (result[format] || "") : "";
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.format === format));
}

const reasonLabels = {
  tesseract_numeric_disagreement: "Hai OCR đọc khác số",
  tesseract_diacritic_disagreement: "Hai OCR đọc khác dấu tiếng Việt",
  tesseract_material_disagreement: "Hai OCR đọc khác nội dung",
  tesseract_line_unmatched: "Tesseract không ghép được dòng",
  tesseract_low_confidence: "Tesseract có độ tin cậy thấp",
  primary_recognition_risk: "OCR chính có dấu hiệu nhận dạng lỗi",
  secondary_indicates_primary_omission: "OCR phụ phát hiện OCR chính có thể thiếu từ",
};

function clearRiskReview() {
  riskList.replaceChildren();
  riskMessage.textContent = "";
  riskReview.hidden = true;
}

function renderRiskReview(payload) {
  clearRiskReview();
  const riskyLines = (payload.pages || []).flatMap((page) =>
    (page.line_results || [])
      .filter((line) => line.semantic_risk === "high")
      .map((line) => ({ page, line }))
  );
  const riskTotal = riskyLines.length || Number(payload.semantic_risk_count || 0);
  document.querySelector("#riskCount").textContent = `${riskTotal} dòng rủi ro cao`;
  document.querySelector("#aiReady").textContent = payload.ai_ready
    ? "AI-ready: CÓ"
    : "AI-ready: KHÔNG";
  if (!riskyLines.length) return;

  riskReview.hidden = false;
  riskMessage.textContent =
    `${riskyLines.length} dòng chưa đủ bằng chứng để tự sửa. ` +
    "Hệ thống giữ nguyên text và yêu cầu đối chiếu PDF thay vì đoán.";
  riskyLines.forEach(({ page, line }) => {
    const item = document.createElement("article");
    item.className = "risk-item";

    const heading = document.createElement("h3");
    heading.textContent = `Trang ${page.page_number || page.page_index + 1} · dòng ${line.line_index + 1}`;
    item.appendChild(heading);

    const reasons = document.createElement("p");
    reasons.className = "risk-reasons";
    reasons.textContent = (line.semantic_reasons || [])
      .map((reason) => reasonLabels[reason] || reason)
      .join(" · ");
    item.appendChild(reasons);

    const primary = document.createElement("div");
    primary.className = "evidence";
    primary.textContent = `OCR chính (${Number(line.confidence || 0).toFixed(3)}): ${line.text || "[trống]"}`;
    item.appendChild(primary);

    const verifier = document.createElement("div");
    verifier.className = "evidence verifier";
    verifier.textContent = line.verifier_text
      ? `Tesseract (${Number(line.verifier_confidence || 0).toFixed(3)}): ${line.verifier_text}`
      : "Tesseract: không có dòng đối chiếu";
    item.appendChild(verifier);
    riskList.appendChild(item);
  });
}

function downloadPayload(kind) {
  if (!result) return;
  const stem = result.filename.replace(/\.pdf$/i, "");
  let content;
  let mime;
  let filename;
  if (kind === "audit-json") {
    content = JSON.stringify(result, null, 2);
    mime = "application/json;charset=utf-8";
    filename = `${stem}.audit.json`;
  } else if (kind === "ai-safe") {
    content = result.ai_safe_text || "";
    mime = "text/plain;charset=utf-8";
    filename = `${stem}.ai-safe.txt`;
  } else {
    content = result.text || "";
    mime = "text/plain;charset=utf-8";
    filename = `${stem}.txt`;
  }
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (pdfUrl) URL.revokeObjectURL(pdfUrl);
  result = null;
  showResult();
  summary.hidden = true;
  clearRiskReview();
  downloadButton.disabled = true;
  aiSafeDownloadButton.disabled = true;
  auditDownloadButton.disabled = true;
  extractButton.disabled = !file;
  if (!file) {
    pdfViewer.removeAttribute("src");
    status.textContent = "Chưa chọn tệp.";
    return;
  }
  pdfUrl = URL.createObjectURL(file);
  pdfViewer.src = pdfUrl;
  status.textContent = `${file.name} — ${(file.size / 1024 / 1024).toFixed(2)} MB`;
});

tabs.forEach((tab) => tab.addEventListener("click", () => {
  format = tab.dataset.format;
  showResult();
}));

extractButton.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  extractButton.disabled = true;
  downloadButton.disabled = true;
  aiSafeDownloadButton.disabled = true;
  auditDownloadButton.disabled = true;
  status.textContent = "Đang bóc tách. Trang scan có thể mất nhiều thời gian...";
  const body = new FormData();
  body.append("file", file, file.name);
  try {
    const response = await fetch("/api/extract", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload.detail || payload;
      throw new Error(detail.message_vi || detail.error_code || "Bóc tách thất bại");
    }
    result = payload;
    format = "text";
    showResult();
    document.querySelector("#pageCount").textContent = `${payload.page_count} trang`;
    document.querySelector("#nativeCount").textContent = `${payload.native_page_count} native`;
    document.querySelector("#ocrCount").textContent = `${payload.ocr_page_count} OCR`;
    document.querySelector("#elapsed").textContent = `${(payload.processing_time_ms / 1000).toFixed(2)} giây`;
    summary.hidden = false;
    renderRiskReview(payload);
    downloadButton.disabled = false;
    aiSafeDownloadButton.disabled = false;
    auditDownloadButton.disabled = false;
    status.textContent = payload.status === "complete"
      ? "Hoàn tất. Không còn dòng rủi ro cao."
      : "Hoàn tất một phần. Xem danh sách dòng cần đối chiếu bên dưới.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    extractButton.disabled = false;
  }
});

downloadButton.addEventListener("click", () => downloadPayload("raw"));
aiSafeDownloadButton.addEventListener("click", () => downloadPayload("ai-safe"));
auditDownloadButton.addEventListener("click", () => downloadPayload("audit-json"));
