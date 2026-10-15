"""Shared chat-template handling and answer scoring for Qwen3 (thinking mode
disabled everywhere, so generated length reflects the policy, not hidden
reasoning tokens)."""

from common import parse_answer, parse_int, parse_json_obj


def render_prompt(tokenizer, user_text):
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


ATTENDEE_WORDS = {
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "twelve": 12,
}


def score_prediction(task, gold, text):
    """Return (correct: bool|None, invalid: bool, pred_payload).

    correct is None only when the output is unparseable (counted invalid).
    """
    payload = parse_answer(text)
    if payload is None:
        return None, True, None
    if task == "arith":
        val = parse_int(payload)
        return (val == int(gold)), val is None, val
    obj = parse_json_obj(payload)
    if obj is None:
        return None, True, None
    try:
        got = {
            "date": str(obj.get("date", "")).strip().lower(),
            "amount": int(str(obj.get("amount", "")).replace(",", "").replace("$", "") or -10**9),
            "attendees": _to_int(obj.get("attendees")),
            "project": str(obj.get("project", "")).strip().lower(),
        }
    except (ValueError, TypeError):
        return None, True, None
    if got["attendees"] is None:
        return None, True, None
    correct = (
        got["date"] == str(gold["date"]).lower()
        and got["amount"] == int(gold["amount"])
        and got["attendees"] == int(gold["attendees"])
        and got["project"] == str(gold["project"]).lower()
    )
    return correct, False, got


def _to_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    s = str(v).strip().lower().replace(",", "")
    if s in ATTENDEE_WORDS:
        return ATTENDEE_WORDS[s]
    try:
        return int(s)
    except ValueError:
        return None
