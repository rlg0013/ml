import re
import unicodedata


def normalize_text(text):
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
