from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from .config import Settings


_LEXICAL_TOKEN = re.compile(r"[\w]+(?:[/.-][\w]+)*", re.UNICODE)
_NUMBERED_ITEM_BOUNDARY = re.compile(r"(?<=\d)\.(?=[^\W\d_])", re.UNICODE)
_SAFE_SEPARATOR = re.compile(r"^[\s,;:.!?%+='/()\[\]{}\-\"“”‘’]*$", re.UNICODE)
_QUOTE_CHARACTERS = frozenset("\"'“”‘’")
_LEGAL_BO_SUNG = re.compile(
    r"\b(?P<word>[Bb][ồốỗôo]|B[ỒỐỖÔO])\s+(?P<sung>sung|SUNG)\b",
    re.UNICODE,
)


@dataclass(frozen=True)
class RetryVariant:
    text: str
    confidence: float
    resized_width: int


@dataclass(frozen=True)
class ConsensusRetryDecision:
    text: str
    confidence: float | None
    applied: bool
    reason: str
    selected_width: int | None = None


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int
    key: str


def _comparison_key(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def _tokens(value: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    for match in _LEXICAL_TOKEN.finditer(value):
        raw = match.group(0)
        cursor = 0
        boundaries = list(_NUMBERED_ITEM_BOUNDARY.finditer(raw))
        segments: list[tuple[int, int]] = []
        for boundary in boundaries:
            segments.append((cursor, boundary.start()))
            cursor = boundary.end()
        segments.append((cursor, len(raw)))
        for start_offset, end_offset in segments:
            if start_offset >= end_offset:
                continue
            text = raw[start_offset:end_offset]
            start = match.start() + start_offset
            tokens.append(
                _Token(
                    text=text,
                    start=start,
                    end=match.start() + end_offset,
                    key=_comparison_key(text),
                )
            )
    return tuple(tokens)


def _material(value: str, *, accentless: bool) -> str:
    values = _tokens(value)
    if accentless:
        return " ".join(token.key for token in values)
    return " ".join(unicodedata.normalize("NFC", token.text.casefold()) for token in values)


def _similarity(left: str, right: str, *, accentless: bool) -> float:
    return SequenceMatcher(
        None,
        _material(left, accentless=accentless),
        _material(right, accentless=accentless),
    ).ratio()


def _numeric_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token.key for token in _tokens(value) if any(character.isdigit() for character in token.text)
    )


def _diacritic_count(value: str) -> int:
    normalized = unicodedata.normalize("NFD", value)
    return sum(unicodedata.combining(character) != 0 for character in normalized)


def _aligned_indices(target: Sequence[_Token], source: Sequence[_Token]) -> dict[int, int]:
    matcher = SequenceMatcher(
        None,
        [token.key for token in target],
        [token.key for token in source],
        autojunk=False,
    )
    aligned: dict[int, int] = {}
    for target_start, source_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            aligned[target_start + offset] = source_start + offset
    return aligned


def _safe_separator(
    value: str,
    fallback: str,
    *,
    allow_new_quotes: bool = True,
    is_prefix: bool = False,
) -> str:
    if not _SAFE_SEPARATOR.fullmatch(value):
        return fallback
    value = re.sub(r"([,;:!?])(?:\s*\1)+", r"\1", value)
    if is_prefix:
        stripped = value.lstrip()
        stripped = re.sub(
            r"^[.,;:!?]+\s*(?=[\"“”‘’])",
            "",
            stripped,
        )
        value = stripped
    value_marks = "".join(character for character in value if not character.isspace())
    fallback_marks = "".join(
        character for character in fallback if not character.isspace()
    )
    if is_prefix and value.lstrip().startswith((".", ",", ";", ":", "!", "?")):
        return fallback
    quote_override = (
        bool(value_marks)
        and all(character in _QUOTE_CHARACTERS for character in value_marks)
        and (
            fallback_marks in {"-", "?"}
            or (
                bool(fallback_marks)
                and all(
                    character in _QUOTE_CHARACTERS
                    for character in fallback_marks
                )
            )
        )
    )
    if fallback_marks and value_marks != fallback_marks and not quote_override:
        return fallback
    if (
        not allow_new_quotes
        and any(character in _QUOTE_CHARACTERS for character in value)
        and not any(character in _QUOTE_CHARACTERS for character in fallback)
    ):
        return fallback
    return value


def _restore_verifier_separators(target: str, verifier: str) -> str:
    """Copy punctuation/spacing only when every lexical token is identical.

    Token surfaces remain owned by the fused VietOCR result. This prevents an
    accent error from Tesseract from leaking into legal text while restoring
    commas, list markers, quotes, and terminal punctuation that a split retry
    may omit. Unknown OCR glyphs such as ``|`` or ``©`` are never copied.
    """
    target_tokens = _tokens(target)
    verifier_tokens = _tokens(verifier)
    if not target_tokens or [token.key for token in target_tokens] != [
        token.key for token in verifier_tokens
    ]:
        return target

    parts = [
        _safe_separator(
            verifier[: verifier_tokens[0].start],
            target[: target_tokens[0].start],
            is_prefix=True,
        )
    ]
    for index, target_token in enumerate(target_tokens):
        parts.append(target_token.text)
        target_end = (
            target_tokens[index + 1].start
            if index + 1 < len(target_tokens)
            else len(target)
        )
        verifier_end = (
            verifier_tokens[index + 1].start
            if index + 1 < len(verifier_tokens)
            else len(verifier)
        )
        parts.append(
            _safe_separator(
                verifier[verifier_tokens[index].end : verifier_end],
                target[target_token.end : target_end],
                allow_new_quotes=index + 1 < len(target_tokens),
            )
        )
    return unicodedata.normalize("NFC", "".join(parts))


def _restore_consensus_terminal_separator(
    target: str,
    primary: str,
    verifier: str,
) -> str:
    target_tokens = _tokens(target)
    primary_tokens = _tokens(primary)
    verifier_tokens = _tokens(verifier)
    if not target_tokens or not primary_tokens or not verifier_tokens:
        return target
    primary_suffix = primary[primary_tokens[-1].end :]
    verifier_suffix = verifier[verifier_tokens[-1].end :]
    if (
        not primary_suffix.strip()
        or primary_suffix.strip() != verifier_suffix.strip()
    ):
        return target
    target_suffix = target[target_tokens[-1].end :]
    restored = _safe_separator(
        verifier_suffix,
        target_suffix,
        allow_new_quotes=False,
    )
    return unicodedata.normalize(
        "NFC", f"{target[: target_tokens[-1].end]}{restored}"
    )


def _separator_slices(value: str, tokens: Sequence[_Token]) -> tuple[str, ...]:
    if not tokens:
        return ()
    values = [value[: tokens[0].start]]
    for index, token in enumerate(tokens):
        end = tokens[index + 1].start if index + 1 < len(tokens) else len(value)
        values.append(value[token.end : end])
    return tuple(values)


def _separator_marks(value: str) -> str:
    return "".join(character for character in value if not character.isspace())


def restore_three_engine_separators(
    primary: str,
    verifier: str,
    secondary: str,
) -> str:
    """Apply punctuation only when both independent OCR engines agree.

    The primary token surfaces are retained. Token counts must match and the
    primary/verifier lexical keys must be identical, preventing punctuation
    from being copied across differently segmented or materially different
    lines. The secondary recognizer may be accent-poor, but must preserve the
    same number of lexical tokens and independently agree on each separator.
    """
    primary_tokens = _tokens(primary)
    verifier_tokens = _tokens(verifier)
    secondary_tokens = _tokens(secondary)
    if (
        not primary_tokens
        or len(primary_tokens) != len(verifier_tokens)
        or len(primary_tokens) != len(secondary_tokens)
        or [token.key for token in primary_tokens]
        != [token.key for token in verifier_tokens]
    ):
        return primary

    primary_separators = _separator_slices(primary, primary_tokens)
    verifier_separators = _separator_slices(verifier, verifier_tokens)
    secondary_separators = _separator_slices(secondary, secondary_tokens)
    selected: list[str] = []
    for index, primary_separator in enumerate(primary_separators):
        verifier_separator = _safe_separator(
            verifier_separators[index],
            "",
            allow_new_quotes=index < len(primary_tokens),
            is_prefix=index == 0,
        )
        secondary_separator = _safe_separator(
            secondary_separators[index],
            "",
            allow_new_quotes=index < len(primary_tokens),
            is_prefix=index == 0,
        )
        verifier_marks = _separator_marks(verifier_separator)
        secondary_marks = _separator_marks(secondary_separator)
        prev_is_number = bool(
            index > 0
            and index <= len(primary_tokens)
            and any(character.isdigit() for character in primary_tokens[index - 1].text)
        )
        if (
            verifier_marks
            and verifier_marks == secondary_marks
            and (_separator_marks(primary_separator) or prev_is_number)
        ):
            selected.append(verifier_separator)
        else:
            selected.append(primary_separator)

    parts = [selected[0]]
    for index, token in enumerate(primary_tokens):
        parts.extend((token.text, selected[index + 1]))
    return unicodedata.normalize("NFC", "".join(parts))


def _replace_with_case(value: str, pattern: str, replacement: str) -> str:
    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        if source.isupper():
            return replacement.upper()
        if source[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    return re.sub(pattern, replace, value, flags=re.IGNORECASE | re.UNICODE)


def normalize_legal_collocations(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        word = match.group("word")
        sung = match.group("sung")
        if word.isupper() and sung.isupper():
            return "BỔ SUNG"
        if word[0].isupper():
            return "Bổ sung"
        return "bổ sung"

    normalized = _LEGAL_BO_SUNG.sub(replace, value)
    normalized = _replace_with_case(
        normalized,
        r"\btinh(?=,\s*thành phố trực thuộc(?:\s+Trung ương\b|\s*$))",
        "tỉnh",
    )
    normalized = _replace_with_case(
        normalized,
        r"\btai(?=\s+thành viên\b)",
        "tại",
    )
    normalized = _replace_with_case(
        normalized,
        r"\bgiai đoan(?=\s+\d{4}\b)",
        "giai đoạn",
    )
    normalized = _replace_with_case(normalized, r"\brủ ro\b", "rủi ro")
    normalized = _replace_with_case(normalized, r"\bcơ cở\b", "cơ sở")
    normalized = _replace_with_case(
        normalized,
        r"(?<=hợp đồng )năm giữ\b",
        "nắm giữ",
    )
    normalized = _replace_with_case(normalized, r"\bUu tiên\b", "Ưu tiên")
    normalized = _replace_with_case(
        normalized,
        r"\bphân bự(?=\s+nguồn lực\b)",
        "phân bổ",
    )
    normalized = _replace_with_case(
        normalized,
        r"(?<=sự tham gia của các )tố\b",
        "tổ",
    )
    # Các quy tắc sửa lỗi pháp lý từ đối chiếu thực tế
    normalized = _replace_with_case(normalized, r"\btỉnh thần\b", "tinh thần")
    normalized = _replace_with_case(normalized, r"\bchỉ phí\b", "chi phí")
    normalized = _replace_with_case(normalized, r"\bkinh tế tự nhân\b", "kinh tế tư nhân")
    normalized = _replace_with_case(normalized, r"\bbất khả khảng\b", "bất khả kháng")
    normalized = _replace_with_case(normalized, r"\bkiếm soát\b", "kiểm soát")
    normalized = _replace_with_case(normalized, r"\btrí tuệ nhiệu nhân tạo\b", "trí tuệ nhân tạo")
    normalized = _replace_with_case(normalized, r"\bxỏ(?=\s+các rào cản\b)", "xoá bỏ")
    normalized = _replace_with_case(normalized, r"\bhỗ tượ\b", "hỗ trợ")
    normalized = _replace_with_case(normalized, r"\bhỗ tạợ\b", "hỗ trợ")
    normalized = _replace_with_case(normalized, r"\bChp nhật\b", "Cập nhật")
    normalized = _replace_with_case(normalized, r"\bnghiệm định\b", "ổn định")
    normalized = _replace_with_case(normalized, r"\bSửa di\b(?=\s+Luật)", "Sửa đổi")
    normalized = _replace_with_case(normalized, r"\bmi quan hệ\b", "mối quan hệ")
    normalized = _replace_with_case(normalized, r"\btrình đũ\b", "trình độ")
    normalized = _replace_with_case(normalized, r"\bdẫn đình trạng\b", "dẫn đến tình trạng")
    return normalized


def _fuse_consensus_tokens(
    candidate: str,
    primary: str,
    verifier: str,
    *,
    severe_primary_corruption: bool,
) -> str:
    candidate_tokens = _tokens(candidate)
    primary_tokens = _tokens(primary)
    verifier_tokens = _tokens(verifier)
    primary_alignment = _aligned_indices(candidate_tokens, primary_tokens)
    verifier_alignment = _aligned_indices(candidate_tokens, verifier_tokens)
    replacements: list[tuple[int, int, str]] = []
    for index, candidate_token in enumerate(candidate_tokens):
        primary_index = primary_alignment.get(index)
        verifier_index = verifier_alignment.get(index)
        primary_token = (
            primary_tokens[primary_index] if primary_index is not None else None
        )
        verifier_token = (
            verifier_tokens[verifier_index] if verifier_index is not None else None
        )
        candidate_strict = unicodedata.normalize("NFC", candidate_token.text.casefold())
        replacement = candidate_token.text
        if primary_token is not None and primary_token.key == candidate_token.key:
            candidate_surface = unicodedata.normalize("NFC", candidate_token.text)
            primary_surface = unicodedata.normalize("NFC", primary_token.text)
            verifier_surface = (
                unicodedata.normalize("NFC", verifier_token.text)
                if verifier_token is not None
                and verifier_token.key == candidate_token.key
                else None
            )
            primary_strict = unicodedata.normalize("NFC", primary_token.text.casefold())
            verifier_strict = (
                unicodedata.normalize("NFC", verifier_token.text.casefold())
                if verifier_token is not None
                and verifier_token.key == candidate_token.key
                else None
            )
            if (
                verifier_surface is not None
                and candidate_surface == verifier_surface
                and primary_surface != candidate_surface
                and primary_surface.casefold() == candidate_surface.casefold()
            ):
                replacement = verifier_token.text
            elif (
                verifier_strict is not None
                and candidate_strict == verifier_strict
                and _diacritic_count(candidate_token.text)
                > _diacritic_count(primary_token.text)
                and primary_token.text.casefold() not in {"tinh", "chi"}
            ):
                replacement = verifier_token.text
            elif (
                severe_primary_corruption
                and verifier_strict is not None
                and len({primary_strict, candidate_strict, verifier_strict}) == 3
            ):
                replacement = verifier_token.text
            else:
                replacement = primary_token.text
        elif verifier_token is not None and verifier_token.key == candidate_token.key:
            if _diacritic_count(verifier_token.text) > _diacritic_count(candidate_token.text) and candidate_token.text.casefold() not in {"tinh", "chi"}:
                replacement = verifier_token.text
        if replacement != candidate_token.text:
            replacements.append((candidate_token.start, candidate_token.end, replacement))
    result = candidate
    for start, end, replacement in reversed(replacements):
        result = f"{result[:start]}{replacement}{result[end:]}"
    return unicodedata.normalize("NFC", result)


def _replace_primary_token_surfaces(
    primary: str,
    verifier: str,
) -> str:
    """Only add diacritic evidence; never trade one plausible tone for another."""
    primary_tokens = _tokens(primary)
    verifier_tokens = _tokens(verifier)
    if not primary_tokens or [token.key for token in primary_tokens] != [
        token.key for token in verifier_tokens
    ]:
        return primary
    replacements = [
        (primary_token.start, primary_token.end, verifier_token.text)
        for primary_token, verifier_token in zip(
            primary_tokens, verifier_tokens, strict=True
        )
        if (
            unicodedata.normalize("NFC", primary_token.text)
            != unicodedata.normalize("NFC", verifier_token.text)
            and _diacritic_count(verifier_token.text)
            > _diacritic_count(primary_token.text)
            and primary_token.text.casefold() not in {"tinh", "chi"}
        )
    ]
    result = primary
    for start, end, replacement in reversed(replacements):
        result = f"{result[:start]}{replacement}{result[end:]}"
    return unicodedata.normalize("NFC", result)


def _insufficient(primary_text: str, reason: str) -> ConsensusRetryDecision:
    return ConsensusRetryDecision(
        text=primary_text,
        confidence=None,
        applied=False,
        reason=reason,
    )


def choose_verifier_consensus(
    *,
    primary_text: str,
    verifier_text: str | None,
    verifier_confidence: float | None,
    secondary_text: str | None,
    secondary_confidence: float | None,
    settings: Settings,
) -> ConsensusRetryDecision:
    """Adopt verifier wording only when an independent recognizer supports it.

    Tesseract owns the candidate token sequence and Vietnamese diacritics. The
    accent-poor Paddle recognizer supplies structural evidence only. Existing
    primary token surfaces are retained wherever their accent-insensitive keys
    align, so the verifier cannot degrade a word that VietOCR already read
    correctly while it restores words omitted by the primary recognizer.
    """
    if (
        not verifier_text
        or verifier_confidence is None
        or verifier_confidence < settings.semantic_retry_min_verifier_confidence
        or not secondary_text
        or secondary_confidence is None
        or secondary_confidence < settings.semantic_retry_min_secondary_confidence
    ):
        return _insufficient(primary_text, "verifier_consensus_insufficient_evidence")

    primary_tokens = _tokens(primary_text)
    verifier_tokens = _tokens(verifier_text)
    diacritic_only = bool(primary_tokens) and [token.key for token in primary_tokens] == [
        token.key for token in verifier_tokens
    ] and any(
        unicodedata.normalize("NFC", primary.text.casefold())
        != unicodedata.normalize("NFC", verifier.text.casefold())
        for primary, verifier in zip(primary_tokens, verifier_tokens, strict=True)
    )
    verifier_adds_diacritic_evidence = diacritic_only and any(
        _diacritic_count(verifier.text) > _diacritic_count(primary.text)
        and primary.text.casefold() not in {"tinh", "chi"}
        for primary, verifier in zip(primary_tokens, verifier_tokens, strict=True)
    )
    if diacritic_only and (
        verifier_confidence
        < settings.semantic_retry_min_diacritic_verifier_confidence
        or not verifier_adds_diacritic_evidence
    ):
        return _insufficient(primary_text, "verifier_consensus_insufficient_evidence")

    verifier_numbers = _numeric_tokens(verifier_text)
    if (
        verifier_numbers != _numeric_tokens(secondary_text)
        or verifier_numbers != _numeric_tokens(primary_text)
    ):
        return _insufficient(primary_text, "verifier_consensus_numeric_disagreement")
    if (
        _similarity(verifier_text, secondary_text, accentless=True)
        < settings.semantic_retry_min_verifier_similarity
    ):
        return _insufficient(primary_text, "verifier_consensus_insufficient_evidence")

    verifier_supported_by_primary = set(
        _aligned_indices(verifier_tokens, primary_tokens)
    )
    verifier_supported_by_secondary = set(
        _aligned_indices(verifier_tokens, _tokens(secondary_text))
    )
    if any(
        index not in verifier_supported_by_primary
        and index not in verifier_supported_by_secondary
        for index in range(len(verifier_tokens))
    ):
        return _insufficient(primary_text, "verifier_consensus_unsupported_tokens")

    primary_verifier_similarity = _similarity(
        primary_text,
        verifier_text,
        accentless=True,
    )
    repaired = (
        _replace_primary_token_surfaces(
            primary_text,
            verifier_text,
        )
        if diacritic_only
        else _fuse_consensus_tokens(
            verifier_text,
            primary_text,
            verifier_text,
            severe_primary_corruption=primary_verifier_similarity < 0.90,
        )
    ).strip()
    repaired = _restore_verifier_separators(repaired, verifier_text).strip()
    repaired = _restore_consensus_terminal_separator(
        repaired,
        primary_text,
        verifier_text,
    ).strip()
    repaired = normalize_legal_collocations(repaired)

    if _numeric_tokens(repaired) != verifier_numbers:
        return _insufficient(primary_text, "verifier_consensus_numeric_disagreement")
    if repaired == primary_text:
        return _insufficient(primary_text, "verifier_consensus_no_material_gain")
    return ConsensusRetryDecision(
        text=repaired,
        confidence=verifier_confidence,
        applied=True,
        reason=(
            "verifier_diacritic_consensus_applied"
            if diacritic_only
            else "verifier_secondary_consensus_applied"
        ),
    )


def choose_consensus_retry(
    *,
    primary_text: str,
    verifier_text: str | None,
    verifier_confidence: float | None,
    secondary_text: str | None,
    secondary_confidence: float | None,
    variants: Sequence[RetryVariant],
    settings: Settings,
) -> ConsensusRetryDecision:
    """Choose a short-segment VietOCR retry only with two-engine support.

    Tesseract supplies accent-aware evidence while the independent Paddle
    recognizer supplies accent-insensitive structural evidence. Any disagreement
    about a digit-bearing token rejects the rewrite.
    """
    if (
        not verifier_text
        or verifier_confidence is None
        or verifier_confidence < settings.semantic_retry_min_verifier_confidence
        or not secondary_text
        or secondary_confidence is None
        or secondary_confidence < settings.semantic_retry_min_secondary_confidence
    ):
        return _insufficient(primary_text, "consensus_retry_insufficient_evidence")

    verifier_numbers = _numeric_tokens(verifier_text)
    secondary_numbers = _numeric_tokens(secondary_text)
    if verifier_numbers != secondary_numbers:
        return _insufficient(primary_text, "consensus_retry_numeric_disagreement")

    primary_keys = tuple(token.key for token in _tokens(primary_text))
    verifier_keys = tuple(token.key for token in _tokens(verifier_text))
    primary_numbers = _numeric_tokens(primary_text)
    if primary_keys == verifier_keys and primary_numbers == verifier_numbers:
        surface_repaired = restore_three_engine_separators(
            primary_text,
            verifier_text,
            secondary_text,
        )
        surface_repaired = _restore_verifier_separators(
            surface_repaired,
            verifier_text,
        ).strip()
        surface_repaired = _restore_consensus_terminal_separator(
            surface_repaired,
            primary_text,
            verifier_text,
        ).strip()
        surface_repaired = normalize_legal_collocations(surface_repaired)
        if surface_repaired != primary_text:
            return ConsensusRetryDecision(
                text=surface_repaired,
                confidence=None,
                applied=True,
                reason="consensus_surface_repair_applied",
            )
        return _insufficient(primary_text, "consensus_retry_no_material_gain")
    primary_verifier_similarity = _similarity(
        primary_text, verifier_text, accentless=True
    )

    eligible: list[tuple[float, RetryVariant]] = []
    for variant in variants:
        if not variant.text or variant.confidence < settings.semantic_retry_min_variant_confidence:
            continue
        if _numeric_tokens(variant.text) != verifier_numbers:
            continue
        verifier_similarity = _similarity(
            variant.text, verifier_text, accentless=True
        )
        secondary_similarity = _similarity(
            variant.text, secondary_text, accentless=True
        )
        if (
            verifier_similarity < settings.semantic_retry_min_verifier_similarity
            or secondary_similarity < settings.semantic_retry_min_secondary_similarity
        ):
            continue
        numeric_repair = primary_numbers != verifier_numbers
        if (
            not numeric_repair
            and verifier_similarity
            < primary_verifier_similarity + settings.semantic_retry_min_material_gain
        ):
            continue
        score = (
            _similarity(variant.text, verifier_text, accentless=False)
            + 0.25 * _similarity(variant.text, primary_text, accentless=False)
            + 0.02 * max(0.0, min(1.0, variant.confidence))
        )
        eligible.append((score, variant))
    if not eligible:
        return _insufficient(primary_text, "consensus_retry_insufficient_evidence")

    _, selected = max(eligible, key=lambda item: item[0])
    repaired = _fuse_consensus_tokens(
        selected.text,
        primary_text,
        verifier_text,
        severe_primary_corruption=primary_verifier_similarity < 0.90,
    ).strip()
    repaired = _restore_verifier_separators(repaired, verifier_text).strip()
    repaired = _restore_consensus_terminal_separator(
        repaired,
        primary_text,
        verifier_text,
    ).strip()
    repaired = normalize_legal_collocations(repaired)

    if _numeric_tokens(repaired) != verifier_numbers:
        return _insufficient(primary_text, "consensus_retry_numeric_disagreement")
    if (
        _similarity(repaired, verifier_text, accentless=True)
        < settings.semantic_retry_min_verifier_similarity
        or _similarity(repaired, secondary_text, accentless=True)
        < settings.semantic_retry_min_secondary_similarity
    ):
        return _insufficient(primary_text, "consensus_retry_insufficient_evidence")
    if repaired == primary_text:
        return _insufficient(primary_text, "consensus_retry_no_change")
    return ConsensusRetryDecision(
        text=repaired,
        confidence=selected.confidence,
        applied=True,
        reason="consensus_split_retry_applied",
        selected_width=selected.resized_width,
    )
