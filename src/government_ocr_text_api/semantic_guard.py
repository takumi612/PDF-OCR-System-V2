from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from .config import Settings


SemanticRisk = Literal["none", "medium", "high"]
_LEXICAL_TOKEN = re.compile(r"[\w]+(?:[/.-][\w]+)*", re.UNICODE)


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int
    key: str


@dataclass(frozen=True)
class _Comparison:
    primary_tokens: tuple[_Token, ...]
    secondary_tokens: tuple[_Token, ...]
    position_match_ratio: float
    tail_match_ratio: float
    material_similarity: float


@dataclass(frozen=True)
class SemanticDecision:
    text: str
    raw_text: str | None
    risk: SemanticRisk
    reasons: tuple[str, ...]
    secondary_confidence: float | None


def _comparison_key(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def _lexical_tokens(value: str) -> tuple[_Token, ...]:
    return tuple(
        _Token(
            text=match.group(0),
            start=match.start(),
            end=match.end(),
            key=_comparison_key(match.group(0)),
        )
        for match in _LEXICAL_TOKEN.finditer(value)
    )


def _token_similarity(left: _Token, right: _Token) -> float:
    if left.key == right.key:
        return 1.0
    if not left.key or not right.key:
        return 0.0
    return SequenceMatcher(None, left.key, right.key).ratio()


def _material_text(value: str) -> str:
    return " ".join(token.key for token in _lexical_tokens(value))


def _compare_token_sequences(primary: str, secondary: str) -> _Comparison:
    primary_tokens = _lexical_tokens(primary)
    secondary_tokens = _lexical_tokens(secondary)
    shared = min(len(primary_tokens), len(secondary_tokens))
    similarities = tuple(
        _token_similarity(primary_tokens[index], secondary_tokens[index])
        for index in range(shared)
    )
    position_match_ratio = (
        sum(score >= 0.45 for score in similarities) / shared if shared else 0.0
    )
    tail_scores = similarities[max(0, shared - 4) :]
    tail_match_ratio = (
        sum(score >= 0.45 for score in tail_scores) / len(tail_scores)
        if tail_scores
        else 0.0
    )
    material_similarity = SequenceMatcher(
        None,
        _material_text(primary),
        _material_text(secondary),
    ).ratio()
    return _Comparison(
        primary_tokens=primary_tokens,
        secondary_tokens=secondary_tokens,
        position_match_ratio=position_match_ratio,
        tail_match_ratio=tail_match_ratio,
        material_similarity=material_similarity,
    )


def _primary_is_risky(
    confidence: float,
    error_code: str | None,
    settings: Settings,
) -> bool:
    return bool(
        error_code
        or confidence < settings.semantic_primary_low_confidence
    )


def _decision_from_comparison(
    *,
    primary_text: str,
    primary_confidence: float,
    primary_error_code: str | None,
    secondary_confidence: float,
    comparison: _Comparison,
    settings: Settings,
) -> SemanticDecision:
    primary_tokens = comparison.primary_tokens
    secondary_tokens = comparison.secondary_tokens
    primary_risky = _primary_is_risky(
        primary_confidence,
        primary_error_code,
        settings,
    )
    extra_count = len(primary_tokens) - len(secondary_tokens)
    suffix_tokens = (
        primary_tokens[len(secondary_tokens) :]
        if extra_count > 0
        else ()
    )
    numeric_disagreement = any(
        (any(character.isdigit() for character in primary_token.text)
         or any(character.isdigit() for character in secondary_token.text))
        and primary_token.key != secondary_token.key
        for primary_token, secondary_token in zip(primary_tokens, secondary_tokens)
    )
    if numeric_disagreement:
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high",
            reasons=("secondary_numeric_disagreement",),
            secondary_confidence=secondary_confidence,
        )

    numeric_suffix = any(
        any(character.isdigit() for character in token.text)
        for token in suffix_tokens
    )
    if numeric_suffix:
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high",
            reasons=("numeric_suffix_protected",),
            secondary_confidence=secondary_confidence,
        )

    anchored_suffix = bool(
        len(secondary_tokens) >= settings.semantic_prefix_min_tokens
        and extra_count >= settings.semantic_suffix_min_extra_tokens
        and comparison.position_match_ratio >= settings.semantic_position_match_ratio
        and comparison.tail_match_ratio >= settings.semantic_tail_match_ratio
    )

    if anchored_suffix:
        if settings.semantic_auto_trim_enabled and primary_risky:
            boundary = primary_tokens[len(secondary_tokens) - 1].end
            return SemanticDecision(
                text=primary_text[:boundary].rstrip(),
                raw_text=primary_text,
                risk="medium",
                reasons=("unsupported_suffix_removed",),
                secondary_confidence=secondary_confidence,
            )
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high",
            reasons=("secondary_suffix_disagreement",),
            secondary_confidence=secondary_confidence,
        )

    if (
        extra_count > 0
        and comparison.position_match_ratio
        >= settings.semantic_position_match_ratio
        and comparison.tail_match_ratio >= settings.semantic_tail_match_ratio
    ):
        # The evidence is not strong enough to delete, but disagreement about
        # extra primary tokens must never silently pass into the AI-safe channel.
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high",
            reasons=("secondary_suffix_disagreement",),
            secondary_confidence=secondary_confidence,
        )

    if len(primary_tokens) < len(secondary_tokens):
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high",
            reasons=("secondary_indicates_primary_omission",),
            secondary_confidence=secondary_confidence,
        )
    if comparison.material_similarity < settings.semantic_material_similarity:
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high",
            reasons=("secondary_material_disagreement",),
            secondary_confidence=secondary_confidence,
        )
    if primary_risky:
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high",
            reasons=("primary_recognition_risk",),
            secondary_confidence=secondary_confidence,
        )
    return SemanticDecision(
        text=primary_text,
        raw_text=None,
        risk="none",
        reasons=(),
        secondary_confidence=secondary_confidence,
    )


def evaluate_semantic_line(
    *,
    primary_text: str,
    primary_confidence: float,
    primary_error_code: str | None,
    secondary_text: str | None,
    secondary_confidence: float | None,
    settings: Settings,
) -> SemanticDecision:
    primary_tokens = _lexical_tokens(primary_text)
    if not primary_text or not primary_tokens:
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high",
            reasons=("empty_primary_text",),
            secondary_confidence=secondary_confidence,
        )

    primary_risky = _primary_is_risky(
        primary_confidence,
        primary_error_code,
        settings,
    )
    if not settings.semantic_verification_enabled:
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high" if primary_risky else "none",
            reasons=("primary_recognition_risk",) if primary_risky else (),
            secondary_confidence=None,
        )
    if not secondary_text or secondary_confidence is None:
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high" if primary_risky else "none",
            reasons=("secondary_unavailable",) if primary_risky else (),
            secondary_confidence=None,
        )
    if secondary_confidence < settings.semantic_secondary_min_confidence:
        return SemanticDecision(
            text=primary_text,
            raw_text=None,
            risk="high" if primary_risky else "none",
            reasons=("secondary_low_confidence",) if primary_risky else (),
            secondary_confidence=secondary_confidence,
        )

    return _decision_from_comparison(
        primary_text=primary_text,
        primary_confidence=primary_confidence,
        primary_error_code=primary_error_code,
        secondary_confidence=secondary_confidence,
        comparison=_compare_token_sequences(primary_text, secondary_text),
        settings=settings,
    )
