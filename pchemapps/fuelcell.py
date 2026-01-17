import streamlit as st
import pandas as pd
import numpy as np
import time

def run():

    st.title("Electrochemisty Example")
    R = 1.12 # ohms
    deltaG = -250e3 # J/mol
    n_elec = 2
    F = 96485 # C/mol
    T = 298 # K
    E0 = -deltaG/(n_elec*F)
    dt = 1.0

    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'echemData' not in st.session_state:
        st.session_state.echemData = dict(
            t=[0], E=[E0], I=[0], Q=[0])

    I = st.slider("Current", 0.0, 1.0, 0.001)
    Ve = E0 - R*I
    Pe = I*Ve
    dq_irrev = R*I**2
    st.write(f"E = {E0:.3f} V")
    st.write(f"Ve = {Ve:.3f} V")
    st.write(f"Pe = {Pe:.3f} W")
    st.write(f"dq_irrev/dt = {dq_irrev:.4g} W")

    # if started:
    #     st.session_state.running = True
    
    # if st.session_state.running:
    #     st.write("Running")
    #     st.write(st.session_state.eChemData[-1])
    #     prev = st.session_state.eChemData[-1]
    #     Ve=E0-I*R
    #     P=Ve*I
    #     Q = prev['Q']+I*dt
    #     st.session_state.eChemData.append(
    #         dict(t=prev['t']+dt, I=I, Q=Q,
    #                 deltaG=deltaG,
    #                 moles_electrons=Q/F,
    #                 q_irrev=prev['q_irrev']+prev['I']**2*R*dt,
    #                 V_e=E0-I*R,
    #                 P=P,
    #                 we=prev['we']+P*dt
    #             )
    #     )
    #     time.sleep(0.5)
    #     st.experimental_rerun()


if __name__ == "__main__":
    run()


