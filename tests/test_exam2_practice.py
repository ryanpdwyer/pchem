"""Exam 2 practice page: question bank checks and a headless run through the page."""
import random

import pytest
from streamlit.testing.v1 import AppTest

from pchemapps.exam2_practice_bank import all_questions
from pchemapps.practice_log import responses

PAGE = "pages/thermodynamics/exam2_practice.py"


def test_bank_well_formed():
    qs = all_questions()
    ids = [q["id"] for _, _, q in qs]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for _, _, q in qs:
        assert 2 <= len(q["choices"]) <= 8
        assert 0 <= q["answer"] < len(q["choices"])
        assert len(set(q["choices"])) == len(q["choices"])
        assert q["prompt"] and q["explain"]
        assert "—" not in q["prompt"] + q["explain"] + "".join(q["choices"])


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("PCHEM_PRACTICE_DB", str(tmp_path / "practice.sqlite"))
    monkeypatch.setenv("PCHEM_INSTRUCTOR_PASSWORD", "pw")


def _letter_for(q, correct=True):
    order = list(range(len(q["choices"])))
    random.Random(q["id"]).shuffle(order)
    pos = order.index(q["answer"])
    if not correct:
        pos = (pos + 1) % len(order)
    return "ABCDEFGH"[pos]


def _submit(at, letter):
    at.radio[0].set_value(letter).run()
    next(b for b in at.button if b.label == "Submit").click().run()


def test_student_flow_logs_attempts(db):
    qs = [q for _, _, q in all_questions()]
    at = AppTest.from_file(PAGE).run()
    at.text_input[0].input("Test Student").run()
    at.button[0].click().run()

    # Q1: wrong, then right
    _submit(at, _letter_for(qs[0], correct=False))
    assert "not correct" in at.warning[0].value
    _submit(at, _letter_for(qs[0]))
    assert at.success[0].value.startswith("Correct")
    next(b for b in at.button if b.label == "Next").click().run()

    # Q2: wrong twice, answer revealed
    _submit(at, _letter_for(qs[1], correct=False))
    _submit(at, _letter_for(qs[1], correct=False))
    assert at.error[0].value.startswith("The answer is")

    df = responses("che341-exam2-26fa")
    assert list(df.question_id) == [qs[0]["id"], qs[0]["id"], qs[1]["id"], qs[1]["id"]]
    assert list(df.attempt) == [1, 2, 1, 2]
    assert list(df.correct) == [0, 1, 0, 0]
    assert (df.seconds >= 0).all()

    # Resume: a new session starts at question 3
    at2 = AppTest.from_file(PAGE).run()
    at2.text_input[0].input("test  student").run()
    at2.button[0].click().run()
    assert at2.session_state.idx == 2


def test_instructor_view(db):
    at = AppTest.from_file(PAGE)
    at.query_params["view"] = "instructor"
    at.run()
    at.text_input[0].input("pw").run()
    assert not at.exception


def test_sheet_refs_exist():
    from pchemapps import exam2_sheet
    from pchemapps.exam2_practice_bank import SHEET_REFS
    ids = {q["id"] for _, _, q in all_questions()}
    for qid, keys in SHEET_REFS.items():
        assert qid in ids
        for k in keys:
            assert exam2_sheet.path(k).exists(), k


def test_skip_and_jump(db):
    at = AppTest.from_file(PAGE).run()
    at.text_input[0].input("Jumper").run()
    at.button[0].click().run()
    next(b for b in at.button if b.label == "Skip").click().run()
    assert at.session_state.idx == 1
    at.selectbox[0].set_value(10).run()
    assert at.session_state.idx == 10
    assert at.session_state.status == "asking"
    assert not at.exception


def test_history_restored(db):
    qs = [q for _, _, q in all_questions()]
    at = AppTest.from_file(PAGE).run()
    at.text_input[0].input("Hist").run()
    at.button[0].click().run()
    _submit(at, _letter_for(qs[0], correct=False))
    _submit(at, _letter_for(qs[0]))

    # A refreshed page with the name in the URL signs back in and resumes at question 2
    at2 = AppTest.from_file(PAGE)
    at2.query_params["name"] = "Hist"
    at2.run()
    assert at2.session_state.idx == 1
    # Going back to question 1 shows the earlier answers instead of re-asking
    at2.selectbox[0].set_value(0).run()
    assert at2.session_state.status == "review"
    assert any("Your earlier answers" in m.value for m in at2.markdown)
    # The summary lists question 1 as missed on the first try
    at2.selectbox[0].set_value(len(qs)).run()
    assert any("Question 1" in m.value for m in at2.markdown)
    assert not at2.exception
