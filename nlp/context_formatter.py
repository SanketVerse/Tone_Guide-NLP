"""Rule-based output-context formatting for ToneGuide.

This module deliberately performs small, visible structural edits after the
tone transformation.  It does not use a language model or external service,
which makes every context decision suitable for explaining in a project viva.
"""

import re


OUTPUT_CONTEXTS = [
    "General Text",
    "Email",
    "WhatsApp / Chat",
    "LinkedIn Post",
    "Academic Paragraph",
    "Feedback to Teammate",
]

_FORMAL_TONES = {"Professional", "Formal"}
_FRIENDLY_TONES = {"Friendly", "Casual"}
_GREETING_RE = re.compile(r"^(dear\b[^\n,]*,|hello\b[^\n,]*,|hi\b[^\n,]*,|greetings\b[^\n,]*,)", re.I)
_CLOSING_RE = re.compile(
    r"(?:\n|^)(kind regards|best regards|regards|sincerely|thanks|thank you|best)[,!]?\s*$",
    re.I,
)


def _sentences(text: str) -> list[str]:
    """Use a simple, explainable sentence splitter that preserves punctuation."""
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item.strip()]


def _ensure_terminal_punctuation(text: str) -> str:
    return text if not text or text[-1] in ".!?" else text + "."


def _soften_feedback_language(text: str) -> tuple[str, bool]:
    """Replace a small, documented set of harsh feedback phrases."""
    replacements = [
        (r"\byou\s+failed\s+to\b", "this would benefit from"),
        (r"\bfix\b", "review and improve"),
        (r"\bimmediately\b", "when possible"),
        (r"\bmust\b", "could"),
        (r"\b(many|too many)\s+errors\b", "areas that need revision"),
        (r"\b(bad|poor|terrible)\b", "in need of improvement"),
        (r"\bwrong\b", "in need of revision"),
    ]
    softened = text
    for pattern, replacement in replacements:
        softened = re.sub(pattern, replacement, softened, flags=re.I)
    return softened, softened != text


def _without_greeting_or_closing(text: str) -> tuple[str, bool, bool]:
    """Return body text plus flags for removed message framing."""
    cleaned = text.strip()
    greeting_removed = False
    closing_removed = False

    greeting = _GREETING_RE.match(cleaned)
    if greeting:
        cleaned = cleaned[greeting.end():].lstrip(" \n")
        greeting_removed = True

    closing = _CLOSING_RE.search(cleaned)
    if closing:
        cleaned = cleaned[:closing.start()].rstrip()
        closing_removed = True
    return cleaned, greeting_removed, closing_removed


def _email_paragraphs(body: str) -> str:
    """Make a long email body scannable without changing its wording."""
    sentences = _sentences(body)
    if len(sentences) <= 2:
        return " ".join(sentences) or body
    return "\n\n".join(" ".join(sentences[i:i + 2]) for i in range(0, len(sentences), 2))


def apply_context_formatting(text: str, context: str, tone: str) -> dict:
    """Apply deterministic structure for one supported output context.

    Returns the formatted text, the resolved context, and human-readable notes
    describing only the formatting rules that were applied.
    """
    selected_context = context if context in OUTPUT_CONTEXTS else "General Text"
    text = text.strip()
    notes: list[str] = []

    if selected_context == "General Text" or not text:
        if context not in OUTPUT_CONTEXTS:
            notes.append("Unrecognised output context; kept General Text formatting.")
        return {"text": text, "context": selected_context, "formatting_applied": notes}

    if selected_context == "Email":
        body, has_removed_greeting, has_removed_closing = _without_greeting_or_closing(text)
        greeting = "Hello,"
        # A supplied greeting remains, but is placed on its own line.
        if has_removed_greeting:
            original_greeting = _GREETING_RE.match(text)
            greeting = original_greeting.group(0) if original_greeting else greeting
            notes.append("Placed the existing greeting on its own line.")
        else:
            notes.append("Added a neutral email greeting without inventing a recipient name.")
        if has_removed_closing:
            notes.append("Kept the existing email closing.")
        body = _email_paragraphs(body)
        if "\n\n" in body:
            notes.append("Grouped the message into readable email paragraphs.")
        closing = "Kind regards," if tone in _FORMAL_TONES else "Thanks,"
        if not has_removed_closing:
            notes.append(f"Added the {closing.rstrip(',').lower()} email closing for the selected tone.")
        else:
            closing_match = _CLOSING_RE.search(text)
            closing = closing_match.group(1).strip().title() + "," if closing_match else closing
        formatted = f"{greeting}\n\n{body}\n\n{closing}"

    elif selected_context == "WhatsApp / Chat":
        body, greeting_removed, closing_removed = _without_greeting_or_closing(text)
        # Preserve a natural supplied greeting, but avoid creating a formal one.
        greeting_match = _GREETING_RE.match(text)
        prefix = (greeting_match.group(0) + " ") if greeting_match else ""
        formatted = prefix + " ".join(_sentences(body))
        if closing_removed:
            notes.append("Removed the formal sign-off for a chat-style message.")
        if greeting_removed:
            notes.append("Kept the existing light greeting.")
        if len(_sentences(body)) > 1:
            notes.append("Kept short sentences together for quick scanning.")

    elif selected_context == "LinkedIn Post":
        body, _, closing_removed = _without_greeting_or_closing(text)
        first_sentence = _sentences(body)[0] if _sentences(body) else body
        if re.match(r"^(a small takeaway:|here is a thought:|\w[^\n]{0,70}[!?])", first_sentence, re.I):
            hook = first_sentence
            notes.append("Used the opening sentence as the concise hook.")
        else:
            hook = "A small takeaway:"
            notes.append("Added a concise professional opening hook.")
        paragraphs = _email_paragraphs(body)
        ending = "I welcome thoughtful perspectives."
        formatted = f"{hook}\n\n{paragraphs}\n\n{ending}"
        notes.append("Added short, readable post paragraphs and an approachable ending.")
        if closing_removed:
            notes.append("Removed the message-style sign-off.")

    elif selected_context == "Academic Paragraph":
        body, greeting_removed, closing_removed = _without_greeting_or_closing(text)
        # Academic context has an additional explicit contraction pass.
        contractions = {
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "isn't": "is not", "aren't": "are not", "I'm": "I am",
            "it's": "it is", "that's": "that is", "we're": "we are",
        }
        for short, expanded in contractions.items():
            body = re.sub(rf"\b{re.escape(short)}\b", expanded, body, flags=re.I)
        formatted = _ensure_terminal_punctuation(" ".join(_sentences(body)))
        if greeting_removed or closing_removed:
            notes.append("Removed greeting and/or closing for an academic paragraph.")
        notes.append("Combined the text into one coherent paragraph and expanded contractions.")
        notes.append("Retained formal, objective wording from the tone transformation.")

    else:  # Feedback to Teammate
        body, _, _ = _without_greeting_or_closing(text)
        body = _ensure_terminal_punctuation(" ".join(_sentences(body)))
        body, softened = _soften_feedback_language(body)
        formatted = (
            "Thank you for your work on this.\n\n"
            f"One area to strengthen is: {body}\n\n"
            "A helpful next step would be to review this point and share an updated version.\n\n"
            "I appreciate your effort and collaboration."
        )
        notes.extend([
            "Added constructive opening and positive closing.",
            "Presented the feedback respectfully and added a clear next step.",
        ])
        if softened:
            notes.append("Softened direct or harsh wording using documented replacement rules.")

    return {"text": formatted.strip(), "context": selected_context, "formatting_applied": notes}
