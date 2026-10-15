"""CHE 341 Exam 2 scaffolded multiple-choice practice, with a per-attempt log.

Students: /exam2-practice
Instructor view: /exam2-practice?view=instructor (password in $PCHEM_INSTRUCTOR_PASSWORD)
"""
import os
import random
import time
import uuid

import pandas as pd
import streamlit as st

from pchemapps import exam2_figs, exam2_sheet
from pchemapps.exam2_practice_bank import FOLLOW_UP, SHEET_REFS, all_questions
from pchemapps.practice_log import finished_questions, log_response, responses, student_key

PRACTICE_SET = "che341-exam2-26fa"
MAX_TRIES = 2  # after this many misses, show the answer and move on
LETTERS = "ABCDEFGH"

QUESTIONS = all_questions()
N = len(QUESTIONS)

try:
    st.page_link("pages/home.py", label="← Home")
except Exception:  # page_link needs st.navigation (absent in headless tests)
    pass


def choice_order(q):
    """Shuffled choice order, fixed per question id so every student sees the same layout."""
    order = list(range(len(q["choices"])))
    random.Random(q["id"]).shuffle(order)
    return order


def my_attempts():
    """This student's logged attempts, oldest first."""
    df = responses(PRACTICE_SET)
    return df[df.student_key == student_key(st.session_state.student)]


def reset_question():
    """Fresh question, or a review of earlier answers if the student already finished it."""
    st.session_state.tries = 0
    st.session_state.status = "asking"
    if st.session_state.idx < N:
        qid = QUESTIONS[st.session_state.idx][2]["id"]
        if qid in finished_questions(PRACTICE_SET, st.session_state.student, MAX_TRIES):
            st.session_state.status = "review"
    st.session_state.flash = None
    st.session_state.shown_at = time.time()


def show_sheet_refs(q):
    for key in SHEET_REFS.get(q["id"], []):
        if exam2_sheet.path(key).exists():
            st.image(str(exam2_sheet.path(key)), caption=exam2_sheet.caption(key))


def full_sheet():
    page = exam2_sheet.ASSET_DIR / "page.png"
    if page.exists():
        with st.expander("Exam 2 equation sheet"):
            st.image(str(page))


def instructor_view():
    st.title("Exam 2 practice: responses")
    password = os.environ.get("PCHEM_INSTRUCTOR_PASSWORD")
    if not password:
        st.error("Set PCHEM_INSTRUCTOR_PASSWORD on the server to enable this view.")
        return
    if st.text_input("Password", type="password") != password:
        return

    df = responses(PRACTICE_SET)
    if df.empty:
        st.info("No responses yet.")
        return
    first = df[df.attempt == 1]

    st.subheader("By student")
    by_student = df.groupby("student_key").agg(
        student=("student", "last"),
        questions=("question_id", "nunique"),
        attempts=("id", "count"),
        minutes=("seconds", lambda s: round(s.sum() / 60, 1)),
        last_activity=("ts", "max"),
    )
    by_student["first_try_correct"] = first.groupby("student_key").correct.sum()
    by_student = by_student.fillna({"first_try_correct": 0}).astype({"first_try_correct": int})
    st.dataframe(by_student.reset_index(drop=True), hide_index=True)

    st.subheader("By question (first attempts)")
    rows = []
    for _, title, q in QUESTIONS:
        f = first[first.question_id == q["id"]]
        wrong = f[f.correct == 0].choice_text
        rows.append({
            "question": q["id"],
            "ladder": title,
            "students": len(f),
            "pct_correct": round(100 * f.correct.mean()) if len(f) else None,
            "median_s": f.seconds.median() if len(f) else None,
            "top_wrong_choice": wrong.mode().iloc[0] if len(wrong) else "",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader("One student")
    who = st.selectbox("Student", sorted(df.student_key.unique()))
    st.dataframe(df[df.student_key == who][["ts", "question_id", "attempt", "choice_text", "correct", "seconds"]],
                 hide_index=True)

    st.subheader("Full log")
    st.dataframe(df, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{PRACTICE_SET}.csv", mime="text/csv")


def start(name):
    """Sign in and resume at the first unfinished question. The name stays in the URL so a refresh keeps it."""
    st.session_state.student = name.strip()
    st.session_state.session_id = uuid.uuid4().hex[:8]
    st.query_params["name"] = name.strip()
    done = finished_questions(PRACTICE_SET, name, MAX_TRIES)
    st.session_state.idx = next((i for i, (_, _, q) in enumerate(QUESTIONS) if q["id"] not in done), N)
    reset_question()


def sign_in():
    st.title("Exam 2 practice")
    st.caption("Use the same name each time to keep your progress.")
    with st.form("sign_in"):
        name = st.text_input("Your name")
        if st.form_submit_button("Start") and student_key(name):
            start(name)
            st.rerun()


def missed_list():
    """Questions missed on the first try, with where to practice more."""
    first = my_attempts()
    first = first[first.attempt == 1]
    missed = set(first[first.correct == 0].question_id)
    rows = [(i, title, q) for i, (_, title, q) in enumerate(QUESTIONS) if q["id"] in missed]
    if not rows:
        return
    st.subheader("Missed on the first try")
    for i, title, q in rows:
        st.markdown(f"**Question {i + 1}** ({title.split('. ', 1)[-1]}): {FOLLOW_UP.get(q['id'], '')}")


def finished_screen():
    first = my_attempts()
    first = first[first.attempt == 1]
    done = finished_questions(PRACTICE_SET, st.session_state.student, MAX_TRIES)
    if len(done) == N:
        st.success("You finished all the questions.")
    else:
        st.info(f"{len(done)} of {N} questions finished. Use the menu above to go back to the others.")
    st.write(f"First-try correct: {int(first.correct.sum())} of {len(first)} attempted")
    missed_list()
    if st.button("Practice again from the start"):
        st.session_state.idx = 0
        reset_question()
        st.rerun()


def go_to(i):
    st.session_state.idx = i
    reset_question()


def jump_menu():
    """Dropdown to move to any question; checkmarks show finished ones."""
    done = finished_questions(PRACTICE_SET, st.session_state.student, MAX_TRIES)
    labels = [f"{i + 1}. {q['id']}" + (" ✓" if q["id"] in done else "") for i, (_, _, q) in enumerate(QUESTIONS)]
    labels.append("Summary")
    st.session_state.jump = st.session_state.idx
    st.selectbox("Go to question", range(N + 1), format_func=labels.__getitem__, key="jump",
                 on_change=lambda: go_to(st.session_state.jump))


def question_screen():
    _, title, q = QUESTIONS[st.session_state.idx]
    order = choice_order(q)
    correct_pos = order.index(q["answer"])

    st.progress(st.session_state.idx / N, text=f"Question {st.session_state.idx + 1} of {N}")
    st.caption(title)
    st.markdown(q["prompt"])
    for pos, k in enumerate(order):
        st.markdown(f"**{LETTERS[pos]}.** {q['choices'][k]}")

    if st.session_state.status == "asking":
        if st.session_state.flash:
            st.warning(st.session_state.flash)
        full_sheet()
        pick = st.radio("Your answer", list(LETTERS[:len(order)]), index=None, horizontal=True,
                        key=f"{q['id']}-{st.session_state.tries}")
        b1, b2 = st.columns([1, 6])
        if b2.button("Skip"):
            go_to(st.session_state.idx + 1)
            st.rerun()
        if b1.button("Submit", type="primary"):
            if pick is None:
                st.warning("Choose an answer first.")
                return
            pos = LETTERS.index(pick)
            right = pos == correct_pos
            log_response(PRACTICE_SET, st.session_state.student, st.session_state.session_id, q["id"],
                         order[pos], q["choices"][order[pos]], right, time.time() - st.session_state.shown_at)
            st.session_state.tries += 1
            if right:
                st.session_state.status = "correct"
            elif st.session_state.tries >= MAX_TRIES:
                st.session_state.status = "revealed"
            else:
                st.session_state.flash = f"{pick} is not correct. Try again."
                st.session_state.shown_at = time.time()
            st.rerun()
        return

    if st.session_state.status == "review":
        mine = my_attempts()
        mine = mine[mine.question_id == q["id"]]
        past = ", ".join(f"{LETTERS[order.index(int(r.choice_index))]} ({'correct' if r.correct else 'not correct'})"
                         for r in mine.itertuples())
        st.write(f"Your earlier answers: {past}. The answer is {LETTERS[correct_pos]}.")
        if st.button("Try again"):
            st.session_state.status = "asking"
            st.session_state.shown_at = time.time()
            st.rerun()
    elif st.session_state.status == "correct":
        st.success(f"Correct: {LETTERS[correct_pos]}")
    else:
        st.error(f"The answer is {LETTERS[correct_pos]}.")
    st.info(q["explain"])
    if q.get("figure"):
        st.image(str(exam2_figs.path(q["figure"])), width=520)
    show_sheet_refs(q)
    if st.button("Next", type="primary"):
        go_to(st.session_state.idx + 1)
        st.rerun()


if st.query_params.get("view") == "instructor":
    instructor_view()
elif "student" not in st.session_state and not st.query_params.get("name"):
    sign_in()
else:
    if "student" not in st.session_state:
        start(st.query_params["name"])
    cols = st.columns([4, 1])
    cols[0].write(f"Signed in as **{st.session_state.student}**")
    if cols[1].button("Not you?"):
        del st.session_state["student"]
        del st.query_params["name"]
        st.rerun()
    jump_menu()
    if st.session_state.idx >= N:
        finished_screen()
    else:
        question_screen()
