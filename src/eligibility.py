"""
MARG eligibility engine: three-valued rule checking + the profile manager.

A scheme's rule (from data/rag_schemes.json) looks like
    {"max_age": 28, "occupation": "student", "state": "gujarat"}
A person's profile has 9 fields; any of them may be unknown (None).

Each condition is  met / failed / unknown.   A scheme is then
    "no"       if ANY condition failed
    "yes"      if ALL conditions are met
    "unknown"  otherwise (nothing failed, but at least one detail is missing)
    "unverified" if we have no rule for the scheme at all
A missing detail is NEVER treated as "no".
"""
import json
from pathlib import Path

FIELDS = ["state", "age", "gender", "caste", "income", "occupation",
          "disability", "bpl", "residence"]
RULE_FIELD = {"state": "state", "min_age": "age", "max_age": "age", "max_income": "income",
              "castes": "caste", "gender": "gender", "disability": "disability",
              "bpl": "bpl", "residence": "residence", "occupation": "occupation"}
MET, FAILED, UNKNOWN = "met", "failed", "unknown"
YES, NO, MAYBE, UNVERIFIED = "yes", "no", "unknown", "unverified"


# ------------------------------------------------------------------ loading
def load_schemes(path="data/rag_schemes.json"):
    """slug -> scheme record (name, category, state, level, rule, url ...)."""
    return {s["id"]: s for s in json.load(open(path, encoding="utf-8"))}


# ------------------------------------------------------------------ rules
def check_condition(key, value, profile):
    field = RULE_FIELD.get(key)
    if field is None:
        return UNKNOWN                      # a rule type we do not know: never block on it
    have = profile.get(field)
    if have is None:
        return UNKNOWN
    if key == "state":
        return MET if have == value else FAILED
    if key == "min_age":
        return MET if have >= value else FAILED
    if key == "max_age":
        return MET if have <= value else FAILED
    if key == "max_income":
        return MET if have <= value else FAILED
    if key == "castes":
        return MET if have in value else FAILED
    if key == "gender":
        return MET if have == value else FAILED
    if key in ("disability", "bpl"):
        if value is True:
            return MET if have else FAILED
        return MET                           # the rule does not require it
    if key == "residence":
        return MET if have == value else FAILED
    if key == "occupation":
        allowed = value if isinstance(value, list) else [value]
        return MET if have in allowed else FAILED
    return UNKNOWN


def check_rule(rule, profile):
    return {k: check_condition(k, v, profile) for k, v in rule.items()}


def scheme_status(rule, profile):
    if rule is None:
        return UNVERIFIED
    results = check_rule(rule, profile).values()
    if FAILED in results:
        return NO
    if UNKNOWN in results:
        return MAYBE
    return YES


def missing_for(rule, profile):
    """Profile fields that would settle this scheme's unknown conditions."""
    return sorted({RULE_FIELD[k] for k, s in check_rule(rule, profile).items()
                   if s == UNKNOWN and k in RULE_FIELD})


def best_questions(rules, profile):
    """Which missing detail would settle the most still-undecided schemes? [(field, count)]"""
    counts = {}
    for rule in rules.values():
        if scheme_status(rule, profile) == MAYBE:
            for f in missing_for(rule, profile):
                counts[f] = counts.get(f, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


# ------------------------------------------------------------------ explanations
def describe(key, value):
    if key == "state":
        return f"lives in {str(value).title()}"
    if key == "min_age":
        return f"age {value} or more"
    if key == "max_age":
        return f"age {value} or less"
    if key == "max_income":
        return f"family income up to Rs {value:,}"
    if key == "castes":
        return "category " + "/".join(c.upper() for c in value)
    if key == "gender":
        return f"gender {value}"
    if key == "disability":
        return "person with disability"
    if key == "bpl":
        return "BPL household"
    if key == "residence":
        return f"lives in a {value} area"
    if key == "occupation":
        vals = value if isinstance(value, list) else [value]
        return "occupation " + " or ".join(vals)
    return f"{key}: {value}"


def explain(rule, profile):
    """[(status, text)] one line per condition, ready to show to the user."""
    out = []
    for k, status in check_rule(rule, profile).items():
        out.append((status, describe(k, rule[k])))
    return out


# ------------------------------------------------------------------ profile manager
class ProfileManager:
    """Keeps the user's details across a chat.

    * the form fills the defaults
    * an explicit correction in the chat overwrites a value (old one goes to history)
    * a big or contradictory change is held until the user confirms it
    * "what if ..." questions run on a temporary copy that is never saved
    * details about someone else ("my sister is 40") never touch the user's profile
    * missing means unknown, never false
    """

    def __init__(self):
        self.values = {f: None for f in FIELDS}
        self.meta = {}
        self.history = []
        self.pending = {}

    # -- writing
    def set_defaults(self, form, turn=0):
        for f, v in form.items():
            if f in FIELDS and v not in (None, ""):
                self._set(f, v, "form", turn)

    def _set(self, field, value, source, turn):
        old = self.values[field]
        if old is not None and old != value:
            self.history.append({"field": field, "old": old, "new": value,
                                 "source": source, "turn": turn})
        self.values[field] = value
        self.meta[field] = {"source": source, "turn": turn}

    @staticmethod
    def _is_large_change(field, old, new):
        if field in ("state", "gender", "caste"):
            return True
        if field == "age":
            return abs(new - old) > 10
        if field == "income":
            return max(new, old) > 3 * max(1, min(new, old))
        return False

    def update(self, found, turn, source="query"):
        """Apply details found in a chat message. Returns what was applied / needs confirmation."""
        applied, needs = {}, {}
        for f, v in found.items():
            if f not in FIELDS or v is None:
                continue
            old = self.values[f]
            if old is None or old == v:
                self._set(f, v, source, turn)
                applied[f] = v
            elif self._is_large_change(f, old, v):
                self.pending[f] = {"old": old, "new": v, "turn": turn, "source": source}
                needs[f] = v
            else:
                self._set(f, v, source, turn)
                applied[f] = v
        return {"applied": applied, "needs_confirmation": needs}

    def confirm(self, field, accept=True):
        p = self.pending.pop(field, None)
        if p and accept:
            self._set(field, p["new"], p["source"], p["turn"])

    # -- reading
    def current(self):
        return dict(self.values)

    def missing(self):
        return [f for f in FIELDS if self.values[f] is None]

    def hypothetical(self, changes):
        """A temporary profile for a 'what if ...' question. The saved profile is untouched."""
        temp = dict(self.values)
        temp.update({f: v for f, v in changes.items() if f in FIELDS})
        return temp

    @staticmethod
    def about_someone_else(found):
        """A separate profile for a relative or friend: only what was said, the rest unknown."""
        p = {f: None for f in FIELDS}
        p.update({f: v for f, v in found.items() if f in FIELDS})
        return p