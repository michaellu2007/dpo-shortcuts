"""Task generators and answer templates for the DPO shortcut audit.

Design constraints:
- Correctness must be machine-checkable (arithmetic EM, extraction field EM).
- "Wrong" responses must be internally coherent: a wrong answer comes with a
  wrong-but-self-consistent reasoning chain, never a contradiction, so the
  preference signal is about the final answer, not about detectable nonsense.
- Short and long templates are drawn from fixed pools so per-pair token totals
  are comparable across the two conditions.
"""

import json
import random
import re

# ---------------------------------------------------------------- arithmetic

ARITH_INSTR = (
    "Compute the value of the expression. "
    "End your reply with a line of the form 'Answer: <integer>'."
)

OP_SYMBOL = {"+": "+", "-": "-", "*": "*"}


def _arith_chain(rng):
    """Build a left-associated integer chain with non-negative intermediates."""
    n_steps = rng.choice([2, 3])
    while True:
        vals = [rng.randint(2, 60) for _ in range(n_steps + 1)]
        ops = [rng.choice("+-*") for _ in range(n_steps)]
        inter = [vals[0]]
        ok = True
        for op, v in zip(ops, vals[1:]):
            prev = inter[-1]
            cur = prev + v if op == "+" else prev - v if op == "-" else prev * v
            if cur < 0:
                ok = False
                break
            inter.append(cur)
        if ok:
            return vals, ops, inter


def _chain_expr(vals, ops):
    expr = str(vals[0])
    for op, v in zip(ops, vals[1:]):
        expr = f"({expr} {OP_SYMBOL[op]} {v})"
    return expr


def _chain_narration(vals, ops, inter):
    parts = []
    for i, (op, v) in enumerate(zip(ops, vals[1:])):
        lead = ["First", "Next", "Then", "Finally"][i] if i < 3 else "Then"
        parts.append(f"{lead}, {inter[i]} {OP_SYMBOL[op]} {v} = {inter[i + 1]}.")
    return " ".join(parts)


def gen_arith(rng):
    """Return (prompt, gold_int, make_long_correct, make_long_wrong).

    The wrong chain flips one operator and re-evaluates, so the reasoning in
    the long-wrong response is internally consistent up to a plausible slip.
    """
    vals, ops, inter = _arith_chain(rng)
    gold = inter[-1]

    wrong = None
    idx = list(range(len(ops)))
    rng.shuffle(idx)
    for i in idx:
        for alt in "+-*":
            if alt == ops[i]:
                continue
            ops2 = ops.copy()
            ops2[i] = alt
            vals2, ops3, inter2 = vals, ops2, [vals[0]]
            ok = True
            for op, v in zip(ops3, vals2[1:]):
                prev = inter2[-1]
                cur = prev + v if op == "+" else prev - v if op == "-" else prev * v
                if cur < 0:
                    ok = False
                    break
                inter2.append(cur)
            if ok and inter2[-1] != gold:
                wrong = (vals2, ops3, inter2)
                break
        if wrong:
            break
    assert wrong is not None

    expr = _chain_expr(vals, ops)
    prompt = f"{ARITH_INSTR}\n\nExpression: {expr}"

    def long_correct():
        body = _chain_narration(vals, ops, inter)
        return f"{body} The value of the expression is {gold}. Answer: {gold}"

    def long_wrong():
        wv, wo, wi = wrong
        body = _chain_narration(wv, wo, wi)
        return f"{body} The value of the expression is {wi[-1]}. Answer: {wi[-1]}"

    return prompt, gold, long_correct, long_wrong


# ------------------------------------------------------- structured extraction

EXTRACT_INSTR = (
    "Read the memo and extract four fields into a JSON object with exactly "
    "these keys: date, amount, attendees, project. "
    "End your reply with a line of the form 'Answer: <json object>'."
)

DATES = ["March 14", "April 2", "May 21", "June 9", "September 30", "November 12"]
PROJECTS = ["Atlas", "Borealis", "Cinder", "Driftwood", "Ember", "Foxglove"]
AMOUNTS = [1200, 2400, 3800, 4200, 5600, 7500, 8100, 9600]
ATTENDEES = [4, 5, 6, 7, 8, 9, 10, 12]

MEMO_TEMPLATE = (
    "Team memo — {project}\n"
    "The {project} budget review is scheduled for {date} "
    "(it was originally planned for {distractor_date}). "
    "The approved amount for the next phase is ${amount}, "
    "up from ${distractor_amount} last quarter. "
    "Expect {attendees} attendees; {distractor_attendees} joined the last call."
)


def _fmt_attendees_word(n):
    words = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 12: "twelve"}
    return words[n]


def _norm_gold(gold):
    return {
        "date": gold["date"].lower(),
        "amount": int(gold["amount"]),
        "attendees": int(gold["attendees"]),
        "project": gold["project"].lower(),
    }


def gen_extract(rng):
    """Return (prompt, gold_dict, make_long_correct, make_long_wrong).

    The memo plants one distractor per field; every wrong value is one of
    those planted distractors (or a digit-level near miss), never arbitrary.
    """
    project = rng.choice(PROJECTS)
    date = rng.choice(DATES)
    amount = rng.choice(AMOUNTS)
    attendees = rng.choice(ATTENDEES)
    distractor_date = rng.choice([d for d in DATES if d != date])
    distractor_amount = rng.choice([a for a in AMOUNTS if a != amount])
    distractor_attendees = rng.choice([a for a in ATTENDEES if a != attendees])

    memo = MEMO_TEMPLATE.format(
        project=project, date=date, distractor_date=distractor_date,
        amount=amount, distractor_amount=distractor_amount,
        attendees=attendees, distractor_attendees=distractor_attendees,
    )
    prompt = f"{EXTRACT_INSTR}\n\nMemo:\n{memo}"

    gold = {"date": date, "amount": amount, "attendees": attendees, "project": project}

    def json_line(g):
        return json.dumps(g, separators=(", ", ": "))

    def long_correct():
        attendees_txt = (
            f"{attendees}" if rng.random() < 0.5 else _fmt_attendees_word(attendees)
        )
        body = (
            f"The memo states that the {project} budget review is scheduled for {date}, "
            f"so the date is {date}. The approved amount for the next phase is ${amount}, "
            f"which gives amount = {amount}. The memo expects {attendees_txt} attendees, "
            f"so attendees = {attendees}. The memo is titled {project}, "
            f"so the project is {project}. "
            f"Answer: {json_line(gold)}"
        )
        return body

    field = rng.choice(["date", "amount", "attendees", "project"])
    wrong = dict(gold)
    if field == "date":
        wrong["date"] = distractor_date
    elif field == "amount":
        wrong["amount"] = distractor_amount
    elif field == "attendees":
        wrong["attendees"] = distractor_attendees
    else:
        wrong["project"] = rng.choice([p for p in PROJECTS if p != project])

    def long_wrong():
        # Only the perturbed field gets a wrong justification; the other
        # fields are stated plainly and correctly, so the response is
        # internally coherent and the only error is the final value.
        s_date = (
            f"The memo mentions {distractor_date} in connection with the schedule, "
            f"so the date is {wrong['date']}."
            if field == "date"
            else f"The {project} budget review is scheduled for {date}, so the date is {date}."
        )
        s_amount = (
            f"Last quarter's figure was ${distractor_amount}, so amount = {wrong['amount']}."
            if field == "amount"
            else f"The approved amount for the next phase is ${amount}, so amount = {amount}."
        )
        s_att = (
            f"{distractor_attendees} people joined the last call, so attendees = {wrong['attendees']}."
            if field == "attendees"
            else f"The memo expects {attendees} attendees, so attendees = {attendees}."
        )
        s_proj = (
            f"The memo discusses project {wrong['project']}, so the project is {wrong['project']}."
            if field == "project"
            else f"The memo is titled {project}, so the project is {project}."
        )
        body = f"{s_date} {s_amount} {s_att} {s_proj} "
        return f"{body}Answer: {json_line(wrong)}"

    return prompt, gold, long_correct, long_wrong


# -------------------------------------------------------------- response pools

def short_correct(gold_repr):
    return f"Answer: {gold_repr}"


def short_wrong(gold_repr):
    return f"Answer: {gold_repr}"


# -------------------------------------------------------------------- parsing

ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.DOTALL)


def parse_answer(text):
    """Return the payload after the last 'Answer:' or None if absent/empty."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    payload = matches[-1].strip().strip("'\"")
    return payload or None


def parse_int(text):
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


def parse_json_obj(text):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
