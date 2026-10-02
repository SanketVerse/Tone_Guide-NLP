"""Small helper utilities (formatting, validation, truncation)."""

MAX_CHARS = 2000  # guard against very long inputs
MIN_CHARS_WARNING = 3


def validate_input(text: str) -> tuple:
    """Validate user input. Returns (ok: bool, message: str)."""
    if text is None or not str(text).strip():
        return False, "Please enter some text first."
    cleaned = str(text).strip()
    if len(cleaned) < MIN_CHARS_WARNING:
        return False, "Input is very short — results may be unreliable."
    if len(cleaned) > MAX_CHARS:
        return False, (
            f"Input is too long ({len(cleaned)} chars). "
            f"Please keep it under {MAX_CHARS} characters."
        )
    return True, ""


def truncate(text: str, limit: int = MAX_CHARS) -> str:
    """Safely truncate long text."""
    text = str(text)
    return text if len(text) <= limit else text[:limit]


def format_percent(confidence_0_1: float) -> str:
    """Format 0..1 confidence as '91%' string."""
    return f"{round(float(confidence_0_1) * 100, 1)}%"


def word_count(text: str) -> int:
    """Simple whitespace word count for display."""
    return len(str(text).split())
