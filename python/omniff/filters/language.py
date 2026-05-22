from __future__ import annotations

import re


CYRILLIC_KAZAKH_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
CYRILLIC_RANGE = re.compile(r"[Ѐ-ӿ]")
LATIN_RANGE = re.compile(r"[a-zA-Z]")


def detect_language(text: str) -> str:
    if not text.strip():
        return "unknown"

    has_kazakh = any(c in CYRILLIC_KAZAKH_CHARS for c in text)
    cyrillic_count = len(CYRILLIC_RANGE.findall(text))
    latin_count = len(LATIN_RANGE.findall(text))
    total = cyrillic_count + latin_count

    if total == 0:
        return "unknown"

    if has_kazakh:
        return "kk"
    if cyrillic_count > latin_count:
        return "ru"
    return "en"
