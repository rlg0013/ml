import re
import unicodedata


def normalize_text(text):
    """
    Canonical Phase 7 normalization.

    Preserves Unicode letters/numbers across scripts.
    Converts punctuation/symbols to spaces and collapses whitespace.
    """
    if text is None or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text).casefold()

    cleaned = []
    for ch in text:
        category = unicodedata.category(ch)

        if category.startswith(("P", "S")):
            cleaned.append(" ")
        elif ch.isspace():
            cleaned.append(" ")
        else:
            cleaned.append(ch)

    text = "".join(cleaned)
    return re.sub(r"\s+", " ", text).strip()


def first_token(text):
    tokens = text.split()
    return tokens[0] if tokens else ""
