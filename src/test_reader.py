"""Small checks for src/read_query.py.   python src/test_reader.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_query import read_query


def test_basic():
    p = read_query("I am a 24 year old woman from Bihar, family income 2 lakh, I am SC and disabled, I live in a village")
    assert p == {"age": 24, "state": "bihar", "income": 200000, "gender": "female", "caste": "sc",
                 "disability": True, "residence": "rural"}, p


def test_correction_and_words():
    assert read_query("I am 82, no wait I am 80")["age"] == 80
    assert read_query("i'm thirty yrs old")["age"] == 30
    assert read_query(["I am 40", "sorry, I am 42"])["age"] == 42           # later turn wins


def test_negation_and_topic():
    assert "disability" not in read_query("I am not disabled, I am 30")
    assert "occupation" not in read_query("is there any scheme for farming")   # a topic, not who they are
    assert read_query("I am a farmer from Gujarat")["occupation"] == "farmer"
    assert "state" not in read_query("schemes for Delhi University")


def test_unknown_stays_unknown():
    assert read_query("Which schemes can I get?") == {}


def test_hindi():
    p = read_query("मेरी उम्र 37 साल है परिवार की सालाना आय 2.8 लाख रुपये है मैं Tamil Nadu से हूँ")
    assert p["age"] == 37 and p["income"] == 280000 and p["state"] == "tamil nadu", p


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"\n{len(tests)} checks passed")