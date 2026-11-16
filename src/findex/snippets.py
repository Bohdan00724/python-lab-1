import re
from collections.abc import Sequence
from pathlib import Path


def build_snippet(
    doc_path: str | Path,
    terms: Sequence[str],
    phrases: Sequence[tuple[str, ...]] = (),
    window: int = 80,
) -> str:
    """
    Знаходить найкраще вікно ±window символів навколо збігів у документі
    та підсвічує знайдені терміни/фрази у форматі **слово**.
    """
    if not doc_path:
        return ""
    path = Path(doc_path)
    if not path.exists():
        return ""

    raw_text = path.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", " ", raw_text).strip()
    if not compact:
        return ""

    patterns: list[str] = []
    for phrase in phrases:
        if phrase:
            patterns.append(r"\b" + r"\s+".join(re.escape(w) for w in phrase) + r"\b")
    for term in dict.fromkeys(terms):
        if term:
            patterns.append(r"\b" + re.escape(term) + r"\b")

    if not patterns:
        return compact[: window * 2] + ("..." if len(compact) > window * 2 else "")

    combined_re = re.compile("|".join(patterns), flags=re.IGNORECASE)
    matches = list(combined_re.finditer(compact))

    if not matches:
        return compact[: window * 2] + ("..." if len(compact) > window * 2 else "")

    best_match = matches[0]
    best_score = (-1, -1)
    for m in matches:
        center = (m.start() + m.end()) // 2
        w_start = max(0, center - window)
        w_end = min(len(compact), center + window)
        sub = compact[w_start:w_end]
        sub_matches = list(combined_re.finditer(sub))
        unique_hits = len({sm.group(0).lower() for sm in sub_matches})
        score = (unique_hits, len(sub_matches))
        if score > best_score:
            best_score = score
            best_match = m

    center = (best_match.start() + best_match.end()) // 2
    start = max(0, center - window)
    end = min(len(compact), center + window)

    if start > 0:
        space_pos = compact.rfind(" ", 0, start)
        if space_pos != -1 and start - space_pos < 15:
            start = space_pos + 1
    if end < len(compact):
        space_pos = compact.find(" ", end)
        if space_pos != -1 and space_pos - end < 15:
            end = space_pos

    fragment = compact[start:end]
    highlighted = combined_re.sub(lambda m: f"**{m.group(0)}**", fragment)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(compact) else ""
    return f"{prefix}{highlighted}{suffix}"
