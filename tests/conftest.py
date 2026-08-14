import pytest


@pytest.fixture(autouse=True)
def disable_external_secondary_model_by_default(monkeypatch):
    monkeypatch.setenv("GOVERNMENT_OCR_SECONDARY_RECOGNIZER_ENABLED", "false")
