"""Small checks for src/eligibility.py.   Run:  python src/test_eligibility.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eligibility import (FAILED, MAYBE, MET, NO, UNKNOWN, UNVERIFIED, YES,  # noqa: E402
                         ProfileManager, best_questions, check_rule, explain,
                         scheme_status)

RULE = {"max_age": 28, "occupation": "student", "state": "gujarat"}


def test_yes_no_unknown():
    ok = {"age": 24, "occupation": "student", "state": "gujarat"}
    assert scheme_status(RULE, ok) == YES
    assert scheme_status(RULE, dict(ok, age=40)) == NO                    # one failure is enough
    assert scheme_status(RULE, dict(ok, age=None)) == MAYBE               # missing is not "no"
    assert scheme_status(RULE, {}) == MAYBE
    assert scheme_status(RULE, dict(ok, age=40, state=None)) == NO        # failed beats unknown
    assert scheme_status(None, ok) == UNVERIFIED


def test_each_condition_type():
    p = {"age": 30, "income": 200000, "caste": "sc", "gender": "female", "disability": True,
         "bpl": False, "residence": "rural", "occupation": "farmer", "state": "bihar"}
    assert scheme_status({"min_age": 30}, p) == YES
    assert scheme_status({"min_age": 31}, p) == NO
    assert scheme_status({"max_income": 200000}, p) == YES
    assert scheme_status({"max_income": 199999}, p) == NO
    assert scheme_status({"castes": ["sc", "st"]}, p) == YES
    assert scheme_status({"castes": ["obc"]}, p) == NO
    assert scheme_status({"gender": "female"}, p) == YES
    assert scheme_status({"gender": "male"}, p) == NO
    assert scheme_status({"disability": True}, p) == YES
    assert scheme_status({"bpl": True}, p) == NO
    assert scheme_status({"residence": "urban"}, p) == NO
    assert scheme_status({"occupation": ["farmer", "worker"]}, p) == YES
    assert scheme_status({"occupation": ["student"]}, p) == NO


def test_hiding_never_creates_a_wrong_no():
    full = {"age": 24, "occupation": "student", "state": "gujarat"}
    assert scheme_status(RULE, full) == YES
    for hidden in full:
        part = dict(full, **{hidden: None})
        assert scheme_status(RULE, part) in (YES, MAYBE)


def test_best_question():
    rules = {"a": {"max_age": 30, "state": "goa"}, "b": {"max_age": 40}, "c": {"castes": ["sc"]}}
    q = best_questions(rules, {"state": "goa"})
    assert q[0][0] == "age" and q[0][1] == 2, q


def test_explain():
    lines = explain(RULE, {"age": 40, "state": "gujarat"})
    assert (FAILED, "age 28 or less") in lines
    assert (MET, "lives in Gujarat") in lines
    assert (UNKNOWN, "occupation student") in lines


def test_profile_manager():
    pm = ProfileManager()
    pm.set_defaults({"state": "goa", "age": 30, "income": 100000})
    assert pm.current()["state"] == "goa" and "caste" in pm.missing()
    r = pm.update({"age": 32}, turn=1)                                     # small correction
    assert pm.current()["age"] == 32 and r["applied"] == {"age": 32}
    assert pm.history[-1]["old"] == 30
    r = pm.update({"state": "kerala"}, turn=2)                             # big change: ask first
    assert pm.current()["state"] == "goa" and r["needs_confirmation"] == {"state": "kerala"}
    pm.confirm("state")
    assert pm.current()["state"] == "kerala"
    r = pm.update({"income": 900000}, turn=3)                              # 9x jump: ask first
    assert "income" in r["needs_confirmation"] and pm.current()["income"] == 100000
    pm.confirm("income", accept=False)
    assert pm.current()["income"] == 100000
    temp = pm.hypothetical({"income": 300000})                             # what-if: not saved
    assert temp["income"] == 300000 and pm.current()["income"] == 100000
    sister = ProfileManager.about_someone_else({"age": 40})                # someone else
    assert sister["age"] == 40 and sister["state"] is None and pm.current()["age"] == 32


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} checks passed")