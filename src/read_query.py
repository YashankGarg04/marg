"""
Step 8a: read the person's details out of their message (the 'query reader').

    python src/read_query.py "I am a 24 year old woman from Bihar, family income 2 lakh"
    python src/eval_reader.py                       # score on the validation split

Returns only what the message clearly says. Anything unclear is left out (= unknown),
because a wrong detail is worse than a missing one. Later turns override earlier ones
("I am 82, no wait I am 80" -> 80). Works on English, Hinglish and Hindi directly; other
languages get whatever is language-neutral (digits, SC/ST/OBC, BPL, state names in Latin script).
"""
import difflib
import re
import sys
import unicodedata

STATES = ["andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
          "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
          "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
          "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
          "uttarakhand", "west bengal", "delhi", "jammu and kashmir", "ladakh", "puducherry",
          "chandigarh", "andaman and nicobar islands", "lakshadweep", "dadra and nagar haveli and daman and diu"]
STATE_ALIAS = {"orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry",
               "j&k": "jammu and kashmir", "jammu & kashmir": "jammu and kashmir", "nct of delhi": "delhi",
               "bengal": "west bengal", "up": None, "mp": None}

ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

OCC = [  # (label, regex)  -- English, Hinglish, Hindi
    ("farmer", r"farm|kisan|किसान|खेती|cultivat|agricultur|kheti"),
    ("student", r"student|studying|college|school|chhatr|छात्र|विद्यार्थी|vidyarthi|padh"),
    ("unemployed", r"unemploy|jobless|no job|berozgar|बेरोज़गार|बेरोजगार|out of work"),
    ("vendor", r"vendor|hawker|street seller|thela|रेहड़ी|ठेला"),
    ("entrepreneur", r"entrepreneur|own (?:enterprise|business)|start(?:ing)? (?:a |my )?(?:own )?(?:business|enterprise)|"
                     r"set up my own|self.?employ|business|enterprise|startup|start-up|udyam|व्यापार|उद्यम|dukan|shop owner"),
    ("worker", r"worker|labour|labor|daily wage|mazdoor|majdoor|मजदूर|श्रमिक|construction|artisan|weaver|karigar"),
]


def normalize(text):
    out = []
    for ch in text:
        if ch.isdigit() and not ch.isascii():
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except ValueError:
                pass
        out.append(ch)
    t = "".join(out)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)                       # 1,40,000 -> 140000
    return re.sub(r"\s+", " ", t).strip()


def words_to_number(low):
    def num(m):
        a = TENS.get(m.group(1))
        b = ONES.get((m.group(2) if m.re.groups > 1 else None) or "", 0)
        return str(a + b) if a else str(ONES[m.group(1)])
    low = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[ -](one|two|three|four|five|six|seven|eight|nine))?\b",
                 num, low)
    return re.sub(r"\b(" + "|".join(ONES) + r")\b(?= (?:years?|yrs?|saal))", num, low)


def skeleton(w):
    return re.sub(r"[aeiou\W]", "", w.lower())


_AFTER_STATE = re.compile(r"\s*(?:university|univ\b|college|institute|metro|police|government college)")


def find_state(text):
    low = text.lower()
    best, pos = None, -1
    for name in sorted(STATES, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(name)}\b", low):
            if _AFTER_STATE.match(low[m.end():]) or re.search(r"(?:for|about|abt|at)\s+$", low[max(0, m.start() - 6):m.start()]) \
                    and name in ("delhi",):
                continue
            if m.start() >= pos:
                best, pos = name, m.start()
    for a, name in STATE_ALIAS.items():
        if name:
            for m in re.finditer(rf"\b{re.escape(a)}\b", low):
                if m.start() >= pos:
                    best, pos = name, m.start()
    if best:
        return best
    words = re.findall(r"[a-z]+", low)
    grams = [" ".join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)]
    skel = {skeleton(st_): st_ for st_ in STATES if len(skeleton(st_)) >= 4}
    for g in grams:                                              # 'mdhy prdsh', 'krntk', 'uttrkhnd'
        if len(skeleton(g)) >= 4 and skeleton(g) in skel and g not in ("hi",):
            return skel[skeleton(g)]
    for g in grams:                                              # typos: 'maharastra', 'gujrat'
        if len(g) > 5:
            c = difflib.get_close_matches(g, STATES, n=1, cutoff=0.88)
            if c:
                return c[0]
    return None


def _last(matches):
    return matches[-1] if matches else None


def read_age(low):
    cands = []
    pats = [r"(?:oops|no wait|sorry|correction|i mean|actually|wait)\W{0,3}(?:i mean |my age is |age:? |i am |i'm )?(\d{1,3})\b",
            r"\b(?:turned|tuened|tunred|turnd|turning)\s*(\d{1,3})\b", r"\bi ?a?m\s*(\d{1,3})\b(?!\s*(?:lakh|lac|rs|rupee|%|k\b))",
            r"\b(?:a[eg]{2}|ag e|age)\s*(?:is|of|:|-)?\s*(\d{1,3})\b",
            r"(\d{1,3})\s*(?:years?|yrs?|yr|saal|sal|वर्ष|साल)(?:\s*(?:old|ka|ki|ke|की|का|के|है))?",
            r"\bage(?:d)?\s*(?:is|of|:|-)?\s*(\d{1,3})\b", r"\b(?:i am|i'm|im|main|मैं)\s*(\d{1,3})\b(?!\s*(?:lakh|lac|rs|rupee|%|k\b))",
            r"(?:turned|turn)\s*(\d{1,3})\b", r"(?:उम्र|आयु|umr|age)\D{0,12}(\d{1,3})", r"(\d{1,3})\s*(?:की|साल की)\s*उम्र"]
    for p in pats:
        for m in re.finditer(p, low):
            v = int(m.group(1))
            tail = low[m.end():m.end() + 12]
            if 1 <= v <= 110 and not re.match(r"\s*(?:lakh|lac|thousand|crore|%|percent)", tail):
                cands.append((m.start(), v))
    return sorted(cands)[-1][1] if cands else None


def read_income(low):
    cands = []
    num = r"(\d+(?:\.\d+)?)"
    unit = r"\s*(lakhs?|lacs?|lac|crores?|thousand|k\b|l\b|हज़ार|हजार|लाख|करोड़)?"
    for m in re.finditer(rf"(?:income|incm|earn\w*|salary|कमाई|आय|आमदनी|aay|aamdani|kamai)\D{{0,40}}?(?:rs\.?|₹|inr|rupees?|रुपये?)?\s*{num}{unit}", low):
        cands.append((m.start(), m.group(1), m.group(2)))
    for m in re.finditer(rf"(?:rs\.?|₹|inr|rupees?)\s*{num}{unit}|{num}{unit}\s*(?:rs\b|rupees?|rps|रुपये?|₹)|{num}\s*(lakhs?|lacs?|lac|crores?|लाख)", low):
        g = [x for x in m.groups() if x]
        cands.append((m.start(), g[0], g[1] if len(g) > 1 else None))
    if not cands:
        return None
    _, n, u = sorted(cands)[-1]
    v = float(n)
    u = (u or "").lower()
    if u == "l" or u.startswith(("lakh", "lac", "लाख")):
        v *= 100000
    elif u.startswith(("crore", "करोड़")):
        v *= 10000000
    elif u in ("thousand", "k", "हज़ार", "हजार"):
        v *= 1000
    if re.search(r"per (?:month|day)|monthly|a month|/month|mahin|महीन|प्रति माह", low[max(0, low.find(n) - 20):low.find(n) + 40]):
        v *= 12
    v = int(round(v))
    return v if 1000 <= v <= 100_000_000 else None


def _neg(low, pat):
    """True if pat appears and is NOT negated just before it."""
    for m in re.finditer(pat, low):
        before = low[max(0, m.start() - 25):m.start()]
        if not re.search(r"\b(?:not|no|nahi|nahin|never|without)\b|नहीं|नही", before):
            return True
    return False


def read_query(text):
    """text may be one message or a list of turns (later turns win). Returns {field: value}."""
    turns = text if isinstance(text, (list, tuple)) else [text]
    prof = {}
    for turn in turns:
        for k, v in _read_one(turn).items():
            prof[k] = v
    return prof


def _read_one(text):
    t = normalize(text)
    low = words_to_number(t.lower())
    _load_lex()
    lw = (_LEX or {}).get("_lakh", {}).get("words", [])
    if lw:                                                   # 'लाख', 'லட்சம்' ... -> 'lakh'
        low = re.sub(r"(\d(?:\.\d+)?)\s*(" + "|".join(map(re.escape, lw)) + r")(?![^\s\.,;:!?()]*[a-z0-9])", r"\1 lakh", low)
    p = {}
    age = read_age(low)
    if age is None:                                          # other languages: the one bare number that fits an age
        nums = [int(x) for x in re.findall(r"(?<![\d\.])(\d{1,3})(?![\d\.]|\s*(?:lakh|%|k\b))", low) if 5 <= int(x) <= 100]
        if len(set(nums)) == 1:
            age = nums[0]
    if age is not None:
        p["age"] = age
    st = find_state(t)
    if st:
        p["state"] = st
    inc = read_income(low)
    if inc:
        p["income"] = inc

    # gender
    fem = re.search(r"\b(?:woman|female|girl|lady|ladies|mahila|aurat|ladki|widow|woamn|womn|wman|femail|femal|famale|lday)\b|महिला|औरत|लड़की|स्त्री|विधवा", low)
    male = re.search(r"\b(?:man|male|boy|gentleman|purush|aadmi|ladka|mael|mle|maale|mal e|make)\b|पुरुष|आदमी|लड़का|मर्द", low)
    if fem and not male:
        p["gender"] = "female"
    elif male and not fem:
        p["gender"] = "male"

    # caste: only when the person says it about themselves (not 'scheme for Scheduled Tribe ...')
    full = {"scheduled caste": "sc", "scheduled tribe": "st", "other backward class": "obc", "other backward classes": "obc",
            "sc": "sc", "st": "st", "obc": "obc", "general": "general", "gen": "general", "open": "general",
            "अनुसूचित जाति": "sc", "अनुसूचित जनजाति": "st", "सामान्य": "general", "पिछड़ा वर्ग": "obc", "ओबीसी": "obc"}
    names = "|".join(sorted(map(re.escape, full), key=len, reverse=True))
    caste = None
    for m in re.finditer(rf"(?:categor\w*|caste|श्रेणी|वर्ग|jati|belong\w*|come\w* under|\bi am|\bi'm|\bim|\bunder|hold)[^.;]{{0,25}}?\b(?:the |a )?({names})\b", low):
        caste = full[m.group(1)]
    if caste is None:                                       # a bare 'SC' / 'ST' / 'OBC' not inside a scheme topic
        pat = r"(?<![a-z])(sc|st|obc|general)(?![a-z])" if re.search(r"[^\x00-\x7f]", low) else r"(?<![a-z])(sc|st|obc)(?![a-z])"
        for m in re.finditer(pat, low):
            if not re.search(r"(?:scheme|schm|for|about|abt|help with)\s*$", low[max(0, m.start() - 12):m.start()]):
                caste = m.group(1)
    if caste:
        p["caste"] = caste

    if _neg(low, r"disab|divyang|differently.?abled|handicap|specially.?abled|विकलांग|दिव्यांग|\bpwd\b|visually impaired|\bblind\b|\bdeaf\b"):
        p["disability"] = True
    if _neg(low, r"below poverty|\bbpl\b|गरीबी रेखा|गरीबी|garibi rekha|ration card.{0,12}(?:bpl|antyodaya)|antyodaya"):
        p["bpl"] = True
    rural = re.search(r"\bvillage|\brural|\bgaon\b|\bgram\b|गाँव|गांव|ग्रामीण|देहात", low)
    urban = re.search(r"\bcity|\burban|\btown\b|\bshehar|\bsheher|शहर|नगर|महानगर", low)
    if rural and not urban:
        p["residence"] = "rural"
    elif urban and not rural:
        p["residence"] = "urban"

    best = None
    for label, pat in OCC:
        for m in re.finditer(pat, low):
            before = low[max(0, m.start() - 16):m.start()]
            after = low[m.end():m.end() + 14]
            if re.search(r"\b(?:for|about|abt|with|get|on|of|regarding|tell me)\s+(?:a |the |my |any )?(?:small |new |own )?$", before):
                continue
            if re.match(r"\w*\s*(?:loan|scheme|schm|schemes|yojana|related|program|prog|subsid|help)", after):
                continue
            if best is None or m.start() < best[0]:
                best = (m.start(), label)
    if best:
        p["occupation"] = best[1]
    for k, v in lexicon_read(t).items():                    # learned words for the other languages
        p.setdefault(k, v)
    return p


_LEX = None


def _load_lex(path="data/lexicon/lexicon.json"):
    global _LEX
    if _LEX is None:
        import json, os
        _LEX = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    return _LEX


def lexicon_read(text):
    """Fill fields the patterns could not, using words learned from the training split."""
    _load_lex()
    if not _LEX:
        return {}
    seq = re.findall(r"[^\s\.,;:!?()\"'।|/]+", text.lower())
    NEG = {"not", "no", "nahi", "nahin", "never", "without", "नहीं", "नही", "illa", "illai"}
    PREV = {"for", "about", "abt", "with", "get", "on", "of", "regarding", "any"}
    NEXT = {"loan", "scheme", "schemes", "schm", "yojana", "related", "program", "subsidy", "help"}
    toks = {w for i, w in enumerate(seq)                                         # 'not disabled', 'scheme for farming': no hit
            if not (i and (seq[i - 1] in NEG or seq[i - 1] in PREV)) and not (i + 1 < len(seq) and seq[i + 1] in NEXT)}
    out = {}
    for field, vals in _LEX.items():
        if field.startswith("_"):
            continue
        hit = {v for v, words in vals.items() if toks & set(words)}
        if len(hit) == 1:
            v = hit.pop()
            out[field] = True if v == "true" else v
    return out


if __name__ == "__main__":
    print(read_query(" ".join(sys.argv[1:]) or "I am a 24 year old woman from Bihar, family income 2 lakh"))