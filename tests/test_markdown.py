from government_ocr_text_api.extractor import markdown_to_text


def test_markdown_to_text_preserves_content_and_tables():
    source = "# Tiêu đề\n\n| Cột A | Cột B |\n|---|---|\n| Một | Hai |"
    assert markdown_to_text(source) == "Tiêu đề\n\nCột A\tCột B\nMột\tHai"
