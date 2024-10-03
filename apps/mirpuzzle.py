import streamlit as st
import uuid
import csv
from datetime import datetime
from util import sci_form

def log_response(user_id, puzzle_number, P, D, calculated_rate, expected_rate, user_expected_rate, is_correct):
    with open('puzzle_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), user_id, puzzle_number, P, D, calculated_rate, expected_rate, user_expected_rate, is_correct])

def run():
    # Initialize session state variables
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if 'ever_submitted' not in st.session_state:
        st.session_state.ever_submitted = False
    if 'final_submitted' not in st.session_state:
        st.session_state.final_submitted = False
    if 'is_rate_correct' not in st.session_state:
        st.session_state.is_rate_correct = None
    if 'is_bob_correct' not in st.session_state:
        st.session_state.is_bob_correct = None
    
    st.title("Method of Initial Rates Puzzles")

    # Custom CSS for input highlighting
    st.markdown("""
    <style>
    .correct-input {
        border-color: green !important;
        background-color: #e6ffe6 !important;
    }
    .incorrect-input {
        border-color: red !important;
        background-color: #ffe6e6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## Puzzle 1")
    st.markdown("""Bob performs a method of initial rates experiment for the reaction between a protein P and a drug D (P + D -> PD). From his analysis, he thought the reaction was first order in each reactant with an overall rate constant of 1.2×10⁻³ M⁻¹s⁻¹, but unfortunately, he lost his data. You have enough reactants left to perform one additional trial. 
                
Perform the trial and determine if Bob's hypothesis was correct.""")
    
    P = st.number_input("Initial concentration of P (M)", value=0.1, step=0.01)
    D = st.number_input("Initial concentration of D (M)", value=0.1, step=0.01)
    button = st.button("Run trial", disabled=st.session_state.ever_submitted)

    # Calculate the actual rate of the reaction (first order in D only)
    rate = 1.2e-3 * D

    # Calculate the expected rate if Bob was correct (first order in both P and D)
    exp_rate = 1.2e-3 * P * D

    if button:
        st.session_state.ever_submitted = True
    
    if st.session_state.ever_submitted:
        st.write(f"The rate of the reaction is {sci_form(rate, 2)} M/s", unsafe_allow_html=True)
        st.write("\n\n")
        st.markdown("If Bob was correct, what would the reaction rate have been?")
        # st.markdown("Remember, Bob thought the reaction was first order in each reactant.")
        # st.latex(r"\text{Rate} = k[P][D] = 1.2 \times 10^{-3} \text{ M}^{-1}\text{s}^{-1} \times [P] \times [D]")

        if st.session_state.is_rate_correct is not None:
            rate_input_class = "correct-input" if st.session_state.is_rate_correct else "incorrect-input"
            st.markdown(f'<style>div.row-widget.stNumberInput > div > div > input {{ {rate_input_class} }}</style>', unsafe_allow_html=True)

        user_expected_rate = st.number_input("Expected Rate (M/s)", value=1.2e-3, step=1e-6, format="%.2e", key="rate")


        bob_correct = st.radio("Was Bob correct?", ["Yes", "No"], index=0)
        
        final_submit = st.button("Submit", key="final_submit")
        
        if final_submit:
            st.session_state.final_submitted = True
            
            is_rate_correct = abs(user_expected_rate - exp_rate) < 1e-7
            st.session_state.is_rate_correct = is_rate_correct
            is_bob_correct = bob_correct == "No"
            st.session_state.is_bob_correct = is_bob_correct
            
            if is_rate_correct:
                st.success(f"Correct! The expected rate if Bob was right is indeed {exp_rate:.2e} M/s")
            else:
                st.error(f"The expected rate if Bob was right should be {exp_rate:.2e} M/s")
                st.info(f"To calculate this, use: Rate = (1.2×10⁻³ M⁻¹s⁻¹)[P][D] = (1.2×10⁻³ M⁻¹s⁻¹) × ({P} M) × ({D} M)")
            
            if is_bob_correct:
                st.success("Correct! Bob was not right.")
                st.write(f"The actual rate ({sci_form(rate, 2)} M/s) ≠ the expected rate if Bob was right ({sci_form(exp_rate, 2)} M/s)", unsafe_allow_html=True)
                st.write("This shows that the reaction is not first order in P, contrary to Bob's hypothesis.")
            else:
                st.error("Check whether your expected rate matches the measured rate.")
            

            
            # Log the response
            log_response(st.session_state.user_id, 1, P, D, rate, exp_rate, user_expected_rate, is_rate_correct and is_bob_correct)

if __name__ == "__main__":
    run()