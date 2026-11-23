"""Ask a plain test question, pull the final answer out of the reply, and check it against the known answer."""

import re
from decimal import Decimal

from core.schema import StaticItem

LETTERS = "ABCDEFGHIJ"
ANSWER_FORMAT = {
    "letter": "Answer: <letter>",
    "number": "Answer: <number>",
    "text": "Answer: <answer>",
}

# a line starting with "Answer:", allowing markdown such as **Answer:** or "## Answer:" before and after the colon
ANSWER_LINE = re.compile(r"^[ \t*_#>`]*answer[ \t]*:[ \t*_`]*(.*)$", re.IGNORECASE | re.MULTILINE)
# a capital letter A-J on its own: in parentheses, or followed by ")", ".", ",", ":", ";", "*" or the end of the line
LETTER = re.compile(r"(?<![A-Za-z0-9])(?:\(([A-J])\)|([A-J])(?:[).,:;*]|\s*$))", re.MULTILINE)
# a number with optional sign, thousands commas and decimals, not part of a word or a longer decimal
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")
LEADING_LETTER = re.compile(r"^[\s(*\[]*([A-J])(?![A-Za-z0-9])", re.IGNORECASE)
BRACKETED_LETTER = re.compile(r"^\(([a-z])\)$")
# BBH's multiple-choice answers are an option letter in brackets, such as "(C)"
OPTION_ANSWER = re.compile(r"^\(([A-Z])\)$")
# a reply's option letter at its start: "(C)", "(C) the robin", "C)", "C." or "C" alone
LEADING_OPTION = re.compile(r"^[\s*\[]*(?:\(([A-Z])\)|([A-Z])(?:[).:]|$))")


def build_prompt(item: StaticItem) -> list[dict]:
    """One user message: the question, the lettered choices if any, and how to end the reply."""
    parts = [item.question.strip()]
    if item.choices:
        parts.append("\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, item.choices)))
    parts.append(
        "Think step by step if needed, then end your reply with a line of the form "
        f"`{ANSWER_FORMAT[item.answer_type]}`."
    )
    return [{"role": "user", "content": "\n\n".join(parts)}]


def extract_answer(text: str, answer_type: str) -> str | None:
    """The text after the last "Answer:" line of the reply.

    Without one, the last thing that looks like an answer: the last standalone letter A-J, the last number, or the
    last non-empty line, by answer_type. None if there is nothing.
    """
    given = [m.strip(" \t*_`") for m in ANSWER_LINE.findall(text)]
    given = [g for g in given if g]
    if given:
        return given[-1]
    if answer_type == "letter":
        found = LETTER.findall(text)
        return (found[-1][0] or found[-1][1]) if found else None
    if answer_type == "number":
        found = NUMBER.findall(text)
        return found[-1] if found else None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _first_number(text: str) -> Decimal | None:
    match = NUMBER.search(text)
    return Decimal(match.group().replace(",", "")) if match else None


def _plain_text(text: str) -> str:
    """Lower case, single spaces, no trailing period, and an option letter without its brackets."""
    text = " ".join(text.split()).rstrip(".").strip().lower()
    bracketed = BRACKETED_LETTER.match(text)
    return bracketed.group(1) if bracketed else text


def is_correct(item: StaticItem, extracted: str | None) -> bool:
    """Whether the extracted answer matches the item's answer; no answer is wrong.

    Letters are compared by the letter alone ("C", "(C)", "C." and "C) foo" all give C). Numbers are compared by value,
    ignoring commas, "$" and trailing zeros, using the first number found. Text is compared ignoring case, extra
    spaces, a trailing period and brackets around an option letter; when the answer is an option letter such as "(C)",
    only the reply's leading option letter is compared, so "(C) the robin" is right.
    """
    if extracted is None:
        return False
    if item.answer_type == "letter":
        match = LEADING_LETTER.match(extracted)
        return bool(match) and match.group(1).upper() == item.answer.strip().upper()
    if item.answer_type == "number":
        got = _first_number(extracted)
        return got is not None and got == _first_number(item.answer)
    option = OPTION_ANSWER.match(item.answer.strip())
    if option:
        match = LEADING_OPTION.match(extracted)
        return bool(match) and (match.group(1) or match.group(2)) == option.group(1)
    return _plain_text(extracted) == _plain_text(item.answer)
