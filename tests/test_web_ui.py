from pathlib import Path


STATIC = Path(__file__).parents[1] / "src" / "government_ocr_text_api" / "static"


def test_web_ui_exposes_partial_review_and_all_safe_downloads():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="riskReview"' in html
    assert 'id="riskList"' in html
    assert 'id="aiSafeDownloadButton"' in html
    assert 'id="auditDownloadButton"' in html
    assert 'data-format="ai_safe_text"' in html
    assert "function renderRiskReview" in script
    assert 'line.semantic_risk === "high"' in script
    assert 'downloadPayload("ai-safe")' in script
    assert 'downloadPayload("audit-json")' in script
    assert "JSON.stringify(result, null, 2)" in script
