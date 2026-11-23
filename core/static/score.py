"""Ask a plain test question, pull the final answer out of the reply, and check it against the known answer."""

import re
import string
from decimal import Decimal

from core.schema import StaticItem

LETTERS = "ABCDEFGHIJ"
ANSWER_FORMAT = {
    "letter": "Answer: <letter>",
    "number": "Answer: <number>",
    "text": "Answer: <answer>",
}

# a line starting with "Answer:" or "Final answer:", allowing markdown such as **Answer:** or "## Answer:" before and
# after the colon
ANSWER_LINE = re.compile(r"^[ \t*_#>`]*(?:final[ \t]+)?answer[ \t]*:[ \t*_`]*(.*)$", re.IGNORECASE | re.MULTILINE)
# LaTeX that wraps an answer in braces, such as \boxed{C} or \text{C}, with no braces inside
LATEX_WRAPPER = re.compile(r"\\(?:boxed|text|mathrm)\s*\{([^{}]*)\}")
# the content of a \boxed{...}, which may hold one level of braces, as in \boxed{\text{C}}
BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
# LaTeX math delimiters: "$", "\(", "\)", "\[" and "\]"
LATEX_DELIMITER = re.compile(r"\$|\\[()\[\]]")
QUOTES = "\"'`"
# a capital letter A-J on its own: in parentheses, or followed by ")", ".", ",", ":", ";", "*" or the end of the line
LETTER = re.compile(r"(?<![A-Za-z0-9])(?:\(([A-J])\)|([A-J])(?:[).,:;*]|\s*$))", re.MULTILINE)
# a number with optional sign, thousands commas and decimals, not part of a word or a longer decimal
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")
# a number as written in an answer: like NUMBER, and also a decimal without a leading zero such as ".5"
ANSWER_NUMBER = re.compile(r"(?<![\w.])-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)")
INTEGER = re.compile(r"-?\d+")
# a reply's letter A-J at its start, after an optional "option" or "choice": in brackets, or not followed by another
# letter or digit ("C", "C.", "C - foo", "C is right"); "A" or "I" followed by a lowercase word is English ("I think C",
# "A robin"), not an option
LEADING_LETTER = re.compile(
    r"^[\s*]*(?:(?i:option|choice)\s+)?"
    r"(?:\(([A-Ja-j])\)|\[([A-Ja-j])\]|([A-Ja-j])(?![A-Za-z0-9])(?!(?<=[AaIi])\s+[a-z]))"
)
BRACKETED_LETTER = re.compile(r"^\(([a-z])\)$")
# BBH's multiple-choice answers are an option letter in brackets, such as "(C)"
OPTION_ANSWER = re.compile(r"^\(([A-Z])\)$")
# a reply's option letter at its start: "(C)" or "(c)", "(C) the robin", "C)", "C." or "C" alone
LEADING_OPTION = re.compile(r"^[\s*\[]*(?:\(([A-Za-z])\)|([A-Z])(?:[).:]|$))")
# BBH answers that are one word out of a pair
ONE_WORD_ANSWERS = {"yes", "no", "true", "false", "valid", "invalid"}


def build_prompt(item: StaticItem) -> list[dict]:
    """One user message: the question, the lettered choices if any, how to end the reply, and the answer format hint.

    The hint is `item.meta["format"]`, a sentence saying what kind of answer the question expects; it is left out when
    the item has none.
    """
    parts = [item.question.strip()]
    if item.choices:
        parts.append("\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, item.choices)))
    parts.append(
        "Think step by step if needed, then end your reply with a line of the form "
        f"`{ANSWER_FORMAT[item.answer_type]}`."
    )
    if item.meta.get("format"):
        parts.append(item.meta["format"])
    return [{"role": "user", "content": "\n\n".join(parts)}]


def _unwrap_latex(text: str) -> str:
    """The text without LaTeX around the answer: \\boxed{x}, \\text{x} and \\mathrm{x} become x, and "$", "\\(", "\\)",
    "\\[" and "\\]" are removed."""
    previous = None
    while previous != text:
        previous = text
        text = LATEX_WRAPPER.sub(r"\1", text)
    return LATEX_DELIMITER.sub("", text)


def _tidy_answer(text: str) -> str:
    """The text without LaTeX, without markdown emphasis or backticks around it, and without matching quotes around it."""
    text = _unwrap_latex(text).strip(" \t*_`")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in QUOTES:
        text = text[1:-1].strip(" \t*_`")
    return text


def extract_answer(text: str, answer_type: str) -> str | None:
    """The text after the last "Answer:" or "Final answer:" line of the reply, without LaTeX and quotes around it.

    Without one, the content of the last \\boxed{...} of the reply. Without that, the last thing that looks like an
    answer: the last standalone letter A-J, the last number, or the last non-empty line, by answer_type. None if there
    is nothing.
    """
    given = [_tidy_answer(m) for m in ANSWER_LINE.findall(text)]
    given = [g for g in given if g]
    if given:
        return given[-1]
    boxed = [_tidy_answer(b) for b in BOXED.findall(text)]
    boxed = [b for b in boxed if b]
    if boxed:
        return boxed[-1]
    if answer_type == "letter":
        found = LETTER.findall(text)
        return (found[-1][0] or found[-1][1]) if found else None
    if answer_type == "number":
        found = NUMBER.findall(text)
        return found[-1] if found else None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _first_number(text: str) -> Decimal | None:
    """The first number of the text, or of the part after its last "=", reading "−" as a minus sign."""
    text = text.replace("\N{MINUS SIGN}", "-").rsplit("=", 1)[-1]
    match = ANSWER_NUMBER.search(text)
    return Decimal(match.group().replace(",", "")) if match else None


def _plain_text(text: str) -> str:
    """Lower case, single spaces, no trailing period, and an option letter without its brackets."""
    text = " ".join(text.split()).rstrip(".").strip().lower()
    bracketed = BRACKETED_LETTER.match(text)
    return bracketed.group(1) if bracketed else text


def is_correct(item: StaticItem, extracted: str | None) -> bool:
    """Whether the extracted answer matches the item's answer; no answer is wrong.

    LaTeX around the answer is ignored. Letters are compared by the letter alone ("C", "(C)", "C.", "C) foo" and
    "option C" all give C; the first letter of a word such as "I think" is not a letter). Numbers, and text answers that
    are whole numbers, are compared by value, ignoring commas, "$" and trailing zeros, using the first number found
    (after the last "=" if there is one). Other text is compared ignoring case, extra spaces, a trailing period and
    brackets around an option letter. Text answers have these special cases:
    - an option letter such as "(C)": only the reply's leading option letter is compared, so "(C) the robin" is right,
      and a lower case letter in brackets such as "(c)" counts;
    - a single word out of yes/no, true/false or valid/invalid: only the reply's first word is compared;
    - the bracket sequence of the dyck_languages subset: spaces and commas are ignored;
    - the sorted words of the word_sorting subset: the words are compared as lists split on spaces and commas.
    """
    if extracted is None:
        return False
    extracted = _unwrap_latex(extracted)
    gold = item.answer.strip()
    if item.answer_type == "letter":
        match = LEADING_LETTER.match(extracted)
        return bool(match) and next(g for g in match.groups() if g).upper() == gold.upper()
    if item.answer_type == "number" or INTEGER.fullmatch(gold):
        got = _first_number(extracted)
        return got is not None and got == _first_number(gold)
    if item.subset == "dyck_languages":
        return re.sub(r"[\s,]", "", extracted) == re.sub(r"[\s,]", "", gold)
    if item.subset == "word_sorting":
        return re.split(r"[\s,]+", _plain_text(extracted).strip(" ,")) == re.split(r"[\s,]+", _plain_text(gold))
    option = OPTION_ANSWER.match(gold)
    if option:
        match = LEADING_OPTION.match(extracted)
        return bool(match) and (match.group(1) or match.group(2)).upper() == option.group(1)
    if gold.lower() in ONE_WORD_ANSWERS:
        words = extracted.split()
        return bool(words) and words[0].strip(string.punctuation).lower() == gold.lower()
    return _plain_text(extracted) == _plain_text(gold)
