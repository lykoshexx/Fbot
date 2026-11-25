#!/usr/bin/env python3
"""fbot7_tools.py -- gives the fbot7 bot a calculator, clock, calendar, notebook and web search.

Put this file in the SAME folder as fbot7.py and start the bot with this file instead of fbot7.py:

    python3 fbot7_tools.py

The model does only the part a small model is good at: it picks the tool and copies words from what you typed.
Everything that must be exact is done here in Python and comes back as a finished sentence:

    you:   how many days until 5 december 2026
    model: <CALL> until 5 december 2026 </CALL>             (never shown as text)
    here:  There are 17 days until 5 December 2026.          (printed as the reply and kept in the chat history)

Tools the model can ask for (the words after the tool name are copied from what you typed):
    calc <expr> [ ; <sentence with {R} for the result and {A} for the previous answer> ]
                              calc 37 * 12 ; 37 times 12 is {R}.        "ans" in <expr> = the previous answer
    time                      It is 14:35.
    date [weekday|month|year] Today is Wednesday, 18 November 2026.
    check <what>              check morning | afternoon | evening | monday .. sunday | weekend | weekday
    day <date>                day 15 august 1947      -> 15 August 1947 was a Friday.
    isday <weekday> <date>    isday friday 8 august 2008
    until <date|holiday>      until 25 december 2026 | until 9 june | until christmas
    since <date|holiday>      since 1 september 2026
    note add|list|find|delete|clear ...     a notebook kept in notebook.json beside this file
    search <words>            first sentence(s) of a Wikipedia article, DuckDuckGo as a backup (needs internet)
It needs a model trained on dataset v31 or later.
Copy slips are repaired from your own words (dates, search words, repeated words); a calculation whose numbers are not the ones you typed is refused.
"""
import ast
import datetime
import json
import operator
import os
import re
import sys
import unicodedata
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NOTES_PATH = os.path.join(HERE, "notebook.json")
SEARCH_MAX_CHARS = 160          # longest snippet shown; the model never has to copy it, so this is only about screen space
SEARCH_TIMEOUT = 8
NOTE_LIST_MAX_CHARS = 120       # the notebook list kept short too, so a reply in the chat history stays small
NOTE_SHOW_CHARS = 40
HISTORY_REPLY_CHARS = 90        # the chat history keeps only this much of a long reply (the model has a small context)

STATE = {"ans": None}           # the previous calculator answer, for "that" / "it" follow-ups


def _now():
    return datetime.datetime.now()


def _today():
    return _now().date()


# ---------------------------------------------------------------- text that goes back into the prompt or the screen
_CHAR_FIX = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
             "\u2026": "...", "\u00a0": " "}


def clean_for_model(text):
    """One line of plain ASCII with no < or > : the tokenizer only knows printable ASCII, and a web page
    must not be able to type a fake  </REPLY>  or  <USER>  tag into the chat history."""
    for a, b in _CHAR_FIX.items():
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("<", " ").replace(">", " ")
    return " ".join(text.split())


def history_text(sentence):
    """What the chat history keeps of a reply: the whole sentence, or a cut at a word boundary if it is long."""
    sentence = " ".join(sentence.split())
    if len(sentence) <= HISTORY_REPLY_CHARS:
        return sentence
    return sentence[:HISTORY_REPLY_CHARS].rsplit(" ", 1)[0].rstrip(",;: ")


# ---------------------------------------------------------------- calculator
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
SORRY_CALC = "I could not work that out. Check the numbers."


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 100:
            raise ValueError("power too big")
        return _BIN[type(node.op)](a, b)
    raise ValueError("not a plain calculation")


def _fmt_number(v):
    """(text, was_cut): was_cut is True when a long decimal had to be cut at 6 places."""
    if isinstance(v, int):
        s = str(v)
        return (s if len(s) <= 24 else "too big"), False
    if v != v or v in (float("inf"), float("-inf")):
        return "error", False
    r = round(v, 6)
    cut = abs(r - v) > 1e-9
    if abs(r) >= 1e15:
        return "%.6g" % r, cut
    if r == int(r):
        return str(int(r)), cut
    return ("%.6f" % r).rstrip("0").rstrip("."), cut


def _calc_value(expr):
    """Evaluate a plain calculation; returns (text, was_cut) or raises ZeroDivisionError / ValueError."""
    return _fmt_number(_eval(ast.parse(expr, mode="eval")))


def tool_calc(arg):
    expr, _, template = arg.partition(" ; ")
    expr, template = expr.strip(), template.strip()
    prev = STATE["ans"]
    shown_expr = expr
    if re.search(r"\bans\b", expr):
        if prev is None:
            return "I do not have an earlier answer to use."
        shown_expr = re.sub(r"\bans\b", prev, expr)
        expr = re.sub(r"\bans\b", "(" + prev + ")", expr)
    if not expr or len(expr) > 80:
        return SORRY_CALC
    try:
        text, cut = _calc_value(expr)
    except ZeroDivisionError:
        return "That is not possible. You cannot divide by zero."
    except (ValueError, SyntaxError, OverflowError, TypeError, RecursionError, MemoryError):
        return SORRY_CALC
    if text == "too big":
        return "That number is too big for me."
    if text == "error":
        return SORRY_CALC
    STATE["ans"] = text
    shown = ("about " + text) if cut else text
    template = template.replace("about {R}", "{R}")
    expr_numbers = set(re.findall(r"\d+(?:\.\d+)?", shown_expr))
    if "{R}" in template and all(n in expr_numbers for n in re.findall(r"\d+(?:\.\d+)?", template)):
        return template.replace("{R}", shown).replace("{A}", prev if prev is not None else "the last answer")
    return f"{shown_expr} is {shown}."


# ---------------------------------------------------------------- dates
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = {}
for _i, _n in enumerate(MONTH_NAMES, 1):
    _MONTHS[_n.lower()] = _i
    _MONTHS[_n.lower()[:3]] = _i
_MONTHS["sept"] = 9
_WEEKDAYS = {n.lower(): i for i, n in enumerate(WEEKDAY_NAMES)}

# holiday words -> (month, day, how it is written in an answer)
HOLIDAYS = {
    "christmas": (12, 25, "Christmas"),
    "christmas eve": (12, 24, "Christmas Eve"),
    "new year": (1, 1, "the new year"),
    "new years": (1, 1, "the new year"),
    "new years eve": (12, 31, "New Year's Eve"),
    "new year eve": (12, 31, "New Year's Eve"),
    "end of year": (12, 31, "the end of the year"),
    "valentines": (2, 14, "Valentine's Day"),
    "valentine": (2, 14, "Valentine's Day"),
}


def _holiday(text):
    t = re.sub(r"[^a-z ]", "", text.lower().replace("'", ""))
    words = [w for w in t.split() if w not in ("the", "day", "on", "my")]
    return HOLIDAYS.get(" ".join(words))


def parse_date(text):
    """(day, month, year or None) from words such as '15 august 1947', 'august 15', '9th oct 2026', '2026-08-15'."""
    t = text.lower().replace(",", " ")
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    else:
        t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)
        mo, nums = None, []
        for w in re.findall(r"[a-z]+|\d+", t):
            if w.isdigit():
                nums.append(int(w))
            elif w in _MONTHS and mo is None:
                mo = _MONTHS[w]
        if mo is None:
            return None
        days = [n for n in nums if 1 <= n <= 31]
        years = [n for n in nums if n >= 1000]
        if not days:
            return None
        d, y = days[0], (years[0] if years else None)
    try:
        datetime.date(y or 2000, mo, d)
    except ValueError:
        return None
    return d, mo, y


def _fmt_date(d, mo, y):
    return f"{d} {MONTH_NAMES[mo - 1]}" + (f" {y}" if y else "")


def _plural_days(n):
    return f"{n} day" + ("" if n == 1 else "s")


def _occurrence(mo, d, y, after, today):
    """Date of a month-day: in year y if given, else the next (after=True) or latest past (after=False) one."""
    if y:
        return datetime.date(y, mo, d)
    years = range(today.year, today.year + 9) if after else range(today.year, today.year - 9, -1)
    for yy in years:
        try:
            t = datetime.date(yy, mo, d)
        except ValueError:
            continue
        if (after and t >= today) or (not after and t <= today):
            return t
    return None


def _target(arg, after):
    """(date, label) for a date or holiday; or (None, error sentence)."""
    today = _today()
    hol = _holiday(arg)
    if hol:
        mo, d, label = hol
        return _occurrence(mo, d, None, after, today), label
    p = parse_date(arg)
    if not p:
        return None, "I could not understand that date."
    d, mo, y = p
    try:
        t = _occurrence(mo, d, y, after, today)
    except ValueError:
        return None, "That date does not exist."
    return t, _fmt_date(d, mo, y)


def _verb(dt, today):
    return "was" if dt < today else "is" if dt == today else "will be"


def tool_day(arg):
    p = parse_date(arg)
    if not p:
        return "I could not understand that date."
    d, mo, y = p
    y = y or _today().year
    try:
        dt = datetime.date(y, mo, d)
    except ValueError:
        return "That date does not exist."
    return f"{_fmt_date(d, mo, y)} {_verb(dt, _today())} a {WEEKDAY_NAMES[dt.weekday()]}."


def tool_isday(arg):
    parts = arg.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() not in _WEEKDAYS:
        return "I could not understand that."
    p = parse_date(parts[1])
    if not p:
        return "I could not understand that date."
    d, mo, y = p
    y = y or _today().year
    try:
        dt = datetime.date(y, mo, d)
    except ValueError:
        return "That date does not exist."
    yes = dt.weekday() == _WEEKDAYS[parts[0].lower()]
    return f"{'Yes' if yes else 'No'}. {_fmt_date(d, mo, y)} {_verb(dt, _today())} a {WEEKDAY_NAMES[dt.weekday()]}."


def tool_until(arg):
    t, label = _target(arg, after=True)
    if t is None:
        return label
    n = (t - _today()).days
    if n == 0:
        return "That date is today." if label[0].isdigit() else f"{label[0].upper()}{label[1:]} is today."
    if n > 0:
        return f"There {'is' if n == 1 else 'are'} {_plural_days(n)} until {label}."
    return f"That date was {_plural_days(-n)} ago."


def tool_since(arg):
    t, label = _target(arg, after=False)
    if t is None:
        return label
    n = (_today() - t).days
    if n == 0:
        return "That date is today."
    if n < 0:
        return "That date has not come yet."
    return f"It has been {_plural_days(n)} since {label}."


def tool_time(_arg=""):
    return f"It is {_now().strftime('%H:%M')}."


def tool_date(arg=""):
    n = _now()
    a = arg.strip().lower()
    if a in ("weekday", "day"):
        return f"Today is {n.strftime('%A')}."
    if a == "month":
        return f"It is {n.strftime('%B')}."
    if a == "year":
        return f"It is {n.year}."
    return f"Today is {n.strftime('%A')}, {n.day} {n.strftime('%B %Y')}."


def tool_check(arg):
    n = _now()
    a = arg.strip().lower()
    hh = n.hour
    if a in ("morning", "afternoon", "evening"):
        yes = {"morning": hh < 12, "afternoon": 12 <= hh < 18, "evening": hh >= 18}[a]
        return f"{'Yes' if yes else 'No'}. It is {n.strftime('%H:%M')}."
    wd = n.strftime("%A")
    if a in _WEEKDAYS:
        yes = n.weekday() == _WEEKDAYS[a]
    elif a == "weekend":
        yes = n.weekday() >= 5
    elif a == "weekday":
        yes = n.weekday() < 5
    else:
        return "I could not tell what to check."
    return f"{'Yes' if yes else 'No'}. Today is {wd}."


# ---------------------------------------------------------------- notebook
def _load_notes():
    try:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("notes"), list):
            return data
    except (OSError, ValueError):
        pass
    return {"next": 1, "notes": []}


def _save_notes(data):
    tmp = NOTES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=1)
    os.replace(tmp, NOTES_PATH)


def _note_line(n):
    t = n["text"]
    return f"{n['id']}. {t[:NOTE_SHOW_CHARS].rstrip()}{'...' if len(t) > NOTE_SHOW_CHARS else ''}"


def _fit(lines, newest_first):
    """Join as many lines as fit in NOTE_LIST_MAX_CHARS: the newest notes for `list`, the first matches for `find`."""
    seq = list(reversed(lines)) if newest_first else list(lines)
    out, total = [], 0
    for ln in seq:
        add = len(ln) + (3 if out else 0)
        if out and total + add > NOTE_LIST_MAX_CHARS:
            break
        out.append(ln)
        total += add
    return " | ".join(reversed(out) if newest_first else out)


def _find_forms(q):
    """The query as typed plus the same word with a common ending cut, so that 'reminders' also finds 'remind me to ...'
    and 'reminder to feed the cat' (the copy repair can stretch a short query into the longer word the person typed)."""
    q = q.lower().strip()
    forms = [q]
    for end in ("ers", "er", "s", "ed", "ing"):
        if q.endswith(end) and len(q) - len(end) >= 4:
            forms.append(q[:-len(end)])
    return forms


def tool_note(arg):
    sub, _, rest = arg.strip().partition(" ")
    rest = rest.strip()
    sub = sub.lower()
    data = _load_notes()
    notes = data["notes"]
    if sub == "add":
        text = clean_for_model(rest)
        if not text:
            return "What should I write in your notebook?"
        nid = data["next"]
        notes.append({"id": nid, "text": text})
        data["next"] = nid + 1
        _save_notes(data)
        return f"Done. It is saved as note {nid}."
    if sub == "list":
        return f"Your notebook: {_fit([_note_line(n) for n in notes], True)}" if notes else "Your notebook is empty."
    if sub == "find":
        q = clean_for_model(rest)
        if not q:
            return "What should I look for in your notebook?"
        hits = [n for n in notes if any(w in n["text"].lower() for w in _find_forms(q))]
        return f"I found: {_fit([_note_line(n) for n in hits], False)}" if hits else f"I found nothing about {q} in your notebook."
    if sub == "delete":
        if not rest.isdigit():
            return "Which note number should I delete?"
        keep = [n for n in notes if n["id"] != int(rest)]
        if len(keep) == len(notes):
            return f"I could not find note {int(rest)}."
        data["notes"] = keep
        _save_notes(data)
        return f"Done. I deleted note {int(rest)}."
    if sub == "clear":
        data["notes"] = []
        _save_notes(data)
        return "Done. Your notebook is empty."
    return "I could not tell what to do with your notebook."


# ---------------------------------------------------------------- web search (Wikipedia first, DuckDuckGo second)
NOTHING_FOUND = "I could not find anything for that. Try different words."


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fbot7-tools/2.0 (personal offline-first chatbot)"})
    with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def shorten_snippet(text, limit=SEARCH_MAX_CHARS):
    """Whole sentences up to `limit` characters; if the first sentence is already too long, cut at a word."""
    text = clean_for_model(re.sub(r"\([^)]*\)", " ", text))        # drop (pronunciations), (dates in brackets)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    if len(text) <= limit:
        return text
    out = ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if len((out + " " + s).strip()) > limit:
            break
        out = (out + " " + s).strip()
    if out:
        return out
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;: ") + "..."


def clean_query(q):
    """The model copies the words after 'search' as typed; drop the little words around the real question."""
    q = clean_for_model(q).strip()
    q = re.sub(r"^(?:for|about)\s+", "", q, flags=re.I)
    q = re.sub(r"\s+(?:please|for me)$", "", q, flags=re.I)
    return q.strip(" ?.!")


def _wikipedia(query):
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro=1&explaintext=1"
           "&exsentences=3&generator=search&gsrlimit=1&gsrsearch=" + urllib.parse.quote(query))
    pages = (_get_json(url).get("query") or {}).get("pages") or {}
    for page in pages.values():
        if page.get("extract"):
            return page["extract"]
    return ""


def _duckduckgo(query):
    url = "https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q=" + urllib.parse.quote(query)
    return _get_json(url).get("AbstractText") or ""


def tool_search(arg):
    query = clean_query(arg)
    if not query:
        return "What should I search for?"
    for backend in (_wikipedia, _duckduckgo):
        try:
            text = backend(query)
        except Exception:                       # no internet, timeout, bad JSON ... try the next one
            continue
        text = shorten_snippet(text) if text else ""
        if text:
            return "Here is what I found: " + text
    return NOTHING_FOUND


# ---------------------------------------------------------------- dispatch
TOOLS = {"calc": tool_calc, "time": tool_time, "date": tool_date, "check": tool_check, "day": tool_day,
         "isday": tool_isday, "until": tool_until, "since": tool_since, "note": tool_note, "search": tool_search}


def parse_call(decoded):
    """Text of the first <CALL> .. </CALL> in the model's output (a missing </CALL> is tolerated), or None."""
    m = re.search(r"<CALL>\s*(.*?)\s*(?:</CALL>|</|<|$)", decoded, re.S)
    return " ".join(m.group(1).split()) if m and m.group(1).strip() else None


def run_tool(call):
    """The finished sentence for a tool request."""
    name, _, arg = call.partition(" ")
    fn = TOOLS.get(name.lower())
    if fn is None:
        return "I do not know how to do that."
    try:
        return clean_for_model(fn(arg)) or NOTHING_FOUND
    except Exception as e:                      # a tool must never crash the chat
        return "Sorry, that did not work (" + clean_for_model(type(e).__name__) + ")."


# ---------------------------------------------------------------- repairing copy slips with the person's own words
_MON_RX = r"(?:jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)"
_DATE_WORDS = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?(?: of)? {_MON_RX}\b(?: \d{{4}}\b)?"
    rf"|\b{_MON_RX} \d{{1,2}}(?:st|nd|rd|th)?\b(?:,? \d{{4}}\b)?"
    r"|\b\d{4}-\d{1,2}-\d{1,2}\b", re.I)
_WEEKDAY_RX = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_TRIGGER = re.compile(r"^(?:(?:please|can you|could you|ok)\s+)?(?:look up|google|find out|search the internet for|search the web for|"
                      r"search online for|search for|do a search for|search)\s+(.*)$", re.I)
_WHAT_GOOGLE = re.compile(r"^what does google say about\s+(.*)$", re.I)
_ALLOWED_CONSTANTS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "24", "52", "60", "100", "365", "1000", "1024", "0.5"}   # 0.5: square root is N ** 0.5
NOT_SURE_NUMBERS = "I am not sure I read your numbers right. Please say it again."


def date_words(text):
    m = _DATE_WORDS.search(text)
    return m.group(0).strip() if m else None


def extract_search_query(user):
    """The words after the search trigger ('google', 'look up', 'search the web for' ...), as typed; None if there is no trigger."""
    u = user.strip()
    m = _WHAT_GOOGLE.match(u) or _TRIGGER.match(u)
    if not m:
        return None
    q = re.sub(r"\s+(?:please|for me)$", "", m.group(1).strip(), flags=re.I).strip(" ?.!")
    return q or None


def collapse_repeats(text):
    """'the odyssey the odyssey' -> 'the odyssey', 'river river river' -> 'river' (a copy that did not know where to stop)."""
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, len(words) // 2 + 1):
            for i in range(len(words) - 2 * n + 1):
                if [w.lower() for w in words[i:i + n]] == [w.lower() for w in words[i + n:i + 2 * n]]:
                    del words[i + n:i + 2 * n]
                    changed = True
                    break
            if changed:
                break
    return " ".join(words)


def calc_numbers_ok(arg, user):
    """Every number in a calculation must be one the person typed (or a small constant like 100 for percent, 60 for minutes)."""
    typed = set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d{3})", "", user)))
    return all(n in typed or n in _ALLOWED_CONSTANTS for n in re.findall(r"\d+(?:\.\d+)?", arg.partition(" ; ")[0]))


def _typed_numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d{3})", "", text)))


def snap_numbers(expr, source):
    """A number the person did not type, but that is the beginning of exactly one number they did ('67' for '672',
    '458' for '4589'), is a copy that stopped one digit early: use the typed number."""
    typed = _typed_numbers(source)

    def fix(m):
        n = m.group(0)
        if n in typed or n in _ALLOWED_CONSTANTS:
            return n
        longer = [t for t in typed if t.startswith(n) and len(t) > len(n)]
        return longer[0] if len(longer) == 1 else n
    return re.sub(r"\d+(?:\.\d+)?", fix, expr)


def snap_words(text, source):
    """Same for words: 'othell' -> 'othello' when exactly one word the person typed starts with it."""
    src = re.findall(r"[A-Za-z0-9']+", source)
    low = {w.lower() for w in src}
    out = []
    for w in text.split():
        core = w.strip(".,?!")
        if len(core) < 3 or core.lower() in low:
            out.append(w)
            continue
        longer = sorted({x.lower() for x in src if x.lower().startswith(core.lower()) and len(x) > len(core)})
        if len(longer) == 1:
            out.append(w.replace(core, next(x for x in src if x.lower() == longer[0])))
        else:
            out.append(w)
    return " ".join(out)


_OP_WORDS = [(r"\bplus\b|\badd\b|\badded\b|\baddition\b", {"+"}), (r"\bminus\b|\bsubtract\w*|\btake away\b|\btake off\b", {"-"}),
             (r"\btimes\b|\bmultipl\w*|\bdouble\b|\btriple\b|\btwice\b|\d\s*x\s*\d|\bx\s*\d", {"*"}),
             (r"\bdivid\w*|\bhalf\b|\bquarter\b|\bthird\b|\bhalve\b", {"/", "%"}), (r"\bpercent\b|%", {"*", "/"}),
             (r"\bsquared\b|\bcubed\b|\bpower\b", {"**"}), (r"\bremainder\b|\bmod\b|\bleft over\b", {"%", "/"})]


def operators_ok(arg, user):
    """If the person used an operation word (plus, minus, times, divided ...), the calculation must use a matching operator:
    'now subtract 672' cannot become 'ans * 672'."""
    expected = set()
    low = user.lower()
    for rx, ops in _OP_WORDS:
        if re.search(rx, low):
            expected |= ops
    expr = arg.partition(" ; ")[0]
    for rx, n in ((r"\bdouble\b|\btwice\b|\bhalf\b|\bhalve\b", "2"), (r"\btriple\b|\bthird\b", "3"), (r"\bquarter\b", "4")):
        if re.search(rx, low) and n not in re.findall(r"\d+(?:\.\d+)?", expr):
            return False                      # "double that" cannot become "ans * 1"
    if not expected:
        return True
    used = set(re.findall(r"\*\*|[*/%+-]", expr))
    return bool(used & expected)


def repair_call(call, user, context=""):
    """Fix what a small model often slips on while COPYING: dates are re-read from the person's own words, the search
    words are the words after the trigger, repeated words are collapsed.  The model still chooses the tool."""
    name, _, arg = call.partition(" ")
    low = name.lower()
    if low not in TOOLS and re.search(r"\bago\b|\bsince\b", user, re.I):       # a garbled tool name on a "how long ago / since" question
        low = "since"
    if low in ("day", "until", "since", "isday"):
        dw = date_words(user)
        if dw:
            if low == "isday":
                parts = arg.split(None, 1)
                wd = parts[0].lower() if parts and parts[0].lower() in _WEEKDAYS and parts[0].lower() in user.lower() else None
                if wd is None:
                    m = _WEEKDAY_RX.search(user)
                    wd = m.group(1).lower() if m else (parts[0].lower() if parts else "")
                return f"isday {wd} {dw}"
            if not _holiday(arg):
                return f"{low} {dw}"
        return call
    if low == "search":
        q = extract_search_query(user)
        return f"search {q}" if q else f"search {snap_words(collapse_repeats(arg), user + ' ' + context)}"
    if low == "note":
        sub, _, rest = arg.partition(" ")
        if sub.lower() in ("add", "find") and rest.strip():
            return f"note {sub.lower()} {snap_words(collapse_repeats(rest), user)}"
    if low == "calc":
        expr, sep, tpl = arg.partition(" ; ")
        return f"calc {snap_numbers(expr, user + ' ' + context)}{sep}{tpl}"
    return call


def run_request(call, user_text="", context=""):
    """(the request actually run, the finished sentence): repairs copy slips first and refuses a calculation whose numbers
    are not ones the person typed (or that were in the last turn: 'and 15 percent' keeps the earlier base)."""
    if user_text:
        call = repair_call(call, user_text, context)
        name, _, arg = call.partition(" ")
        if name.lower() == "calc" and not (calc_numbers_ok(arg, user_text + " " + context) and operators_ok(arg, user_text)):
            return call, NOT_SURE_NUMBERS
    return call, run_tool(call)


def canonical_call(call):
    """What a request MEANS, so two differently written but equal requests compare equal (used to score probes):
    '2 feb 2027' = '2 february 2027', 'calc 250%9' = 'calc 250 % 9', the sentence part of a calc is ignored."""
    name, _, arg = call.strip().partition(" ")
    name, arg = name.lower(), arg.strip()
    if name == "calc":
        expr = re.sub(r"\s+", "", arg.partition(" ; ")[0])
        if "ans" not in expr:
            try:
                return ("calc", _calc_value(expr))
            except Exception:
                pass
        return ("calc", expr)
    if name in ("day", "until", "since", "isday"):
        pre = ""
        if name == "isday":
            pre, _, arg = arg.partition(" ")
            pre = pre.lower()
        hol = _holiday(arg)
        if hol:
            return (name, pre, hol)
        p = parse_date(arg)
        return (name, pre, p) if p else (name, pre, arg.lower())
    if name == "search":
        return ("search", re.sub(r"^(?:the|a|an)\s+", "", clean_query(arg).lower()))
    if name == "note":
        sub, _, rest = arg.partition(" ")
        return ("note", sub.lower(), " ".join(rest.lower().split()))
    return (name, " ".join(arg.lower().split()))


# ---------------------------------------------------------------- hooking into fbot7's chat loop
_fb = None
_orig_stream_generate = None


def _one_pass(gpt, tok, user_text, obs):
    """One generation, like fbot7.stream_generate, except that it only starts printing once it knows the model
    is writing a reply.  A <CALL> is never shown as text.  Returns (new_ids, is_call)."""
    fb = _fb
    prompt_text = fb.wrap_prompt(user_text, obs)
    prompt_ids = tok.encode_prompt(prompt_text)
    while fb.HISTORY and len(prompt_ids) > fb.CONFIG["max_seq_len"] - 60:      # forget the oldest turn first
        fb.HISTORY.pop(0)
        prompt_text = fb.wrap_prompt(user_text, obs)
        prompt_ids = tok.encode_prompt(prompt_text)
    max_new_tokens = max(1, min(fb.CHECK_MAX_NEW, fb.CONFIG["max_seq_len"] - len(prompt_ids)))
    flt = fb.ReplyFilter(bool(fb.SETTINGS.get("showcheck")))
    st = {"ids": [], "pieces": [], "mode": None}

    def show(piece):
        text = flt.feed(piece)
        if text:
            print(text, end="", flush=True)

    def start_reply():
        st["mode"] = "reply"
        print("assistant> ", end="", flush=True)
        for p in st["pieces"]:
            show(p)

    def on_token(tid):
        piece = tok.itos.get(tid, "")
        if piece in ("<EOS>", "<BOS>", "<PAD>"):
            return
        st["ids"].append(tid)
        st["pieces"].append(piece)
        if st["mode"] is None:
            head = tok.decode(st["ids"]).lstrip()
            if len(head) >= 5:                   # "<CALL" versus "<REPL" / "<CHEC"
                if head.startswith("<CALL"):
                    st["mode"] = "call"
                else:
                    start_reply()
        elif st["mode"] == "reply":
            show(piece)

    out_ids = gpt.generate(prompt_ids, max_new_tokens=max_new_tokens,
                           temperature=(None if fb.CHAT_GREEDY else 0.8), top_k=(1 if fb.CHAT_GREEDY else 20),
                           eos_id=tok.eos_id, on_token=on_token)
    new_ids = out_ids[len(prompt_ids):]
    if st["mode"] is None:                       # a very short output: never reached 5 characters
        start_reply()
    if st["mode"] == "reply":
        if not flt.shown_any:
            print(f"[no <REPLY> block -- raw output: {tok.decode(new_ids)}]", end="")
        print()
    return new_ids, st["mode"] == "call"


def tool_stream_generate(gpt, tok, prompt_text, max_new_tokens=80, obs=None):
    """Drop-in replacement for fbot7.stream_generate that understands <CALL> requests: the tool's sentence is
    printed as the reply and returned as a <REPLY> block, so the chat history keeps it."""
    fb = _fb
    if not fb.SETTINGS.get("checkmode"):         # the tool tags only exist in the tagged (check-mode) prompt format
        return _orig_stream_generate(gpt, tok, prompt_text, max_new_tokens, obs)
    new_ids, is_call = _one_pass(gpt, tok, prompt_text, obs)
    if not is_call:
        return new_ids
    call = parse_call(tok.decode(new_ids))
    if call is None:
        print("assistant> [tool request could not be completed]")
        return new_ids
    call, sentence = run_request(call, prompt_text, " ".join(fb.HISTORY[-1]) if fb.HISTORY else "")
    print(f"  [tool: {call}]")
    print("assistant> " + sentence)
    return tok.encode("<REPLY> " + history_text(sentence) + " </REPLY>", add_bos=False, add_eos=False)


def install():
    global _fb, _orig_stream_generate
    import fbot7 as fb
    _fb, _orig_stream_generate = fb, fb.stream_generate
    fb.stream_generate = tool_stream_generate
    return fb


if __name__ == "__main__":
    install().main()
