import streamlit as st

def run():
    # Include a session state variable to track if the user has ever submitted
    if 'ever_submitted' not in st.session_state:
        st.session_state.ever_submitted = False
    
    st.title("Method of Initial Rates Puzzles")

    st.markdown("## Puzzle 1")
    st.markdown("""Bob performs a method of initial rates experiment for the reaction between a protein P and a drug D (P + D -> PD). From his analysis, he thought the reaction was first order in each reactant with an overall rate constant of 1.2×10⁻³ M⁻¹s⁻¹, but unfortunately, he lost his data. You have enough reactants left to perform one additional trial. 
                
Perform the trial and determine if Bob's hypothesis was correct.""")
    
    P = st.number_input("Initial concentration of P (M)", value=0.1, step=0.01)
    D = st.number_input("Initial concentration of D (M)", value=0.1, step=0.01)
    button = st.button("Run trial", disabled=st.session_state.ever_submitted)

    # Calculate the rate of the reaction
    rate = 1.2e-3 * D
    exp_rate = 1.2e-3 *P * D

    if button:
        
        st.session_state.ever_submitted = True
    
    if st.session_state.ever_submitted:
        st.write(f"The rate of the reaction is {rate:.2e} M/s")
        st.write("\n\n")
        st.markdown("If Bob was correct, what would the reaction rate have been?")
        expected_rate = st.number_input("Expected Rate (M/s)", value=1.2e-3, step=1e-6, format="%.2e", key="rate")
        st.radio("Was Bob correct?", ["Yes", "No"], index=0)
        final_submit = st.button("Submit", key="final_submit")
        if final_submit:
            st.write(f"The correct expected rate was {exp_rate:.2e} M/s")
            st.write(f"Bob was not correct, since the expected rate ≠ the calculated rate {rate:.2e} M/s")



if __name__ == "__main__":
    run()