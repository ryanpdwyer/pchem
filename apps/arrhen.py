import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px


def run():

    st.markdown("# How does data become a line?")

    st.markdown("""
    Alice measured the rate of reaction for the decomposition of hydrogen peroxide
    at several temperatures. What should she plot to get a line?
    """)

    T_celsius = np.array([0, 20, 40, 60, 80])
    T_kelvin = T_celsius + 273.15

    A = 2.432e5 # Exponential prefactor
    Ea = 68.9e3 # Activation energy
    R = 8.3145 # Ideal gas constant

    np.random.seed(239049034)
    k = A * np.exp(-Ea/(R*T_kelvin)) * (1+np.random.randn(5)*0.15)

    ln_k = np.log(k)

    T_inverse = 1/T_kelvin

    log_T = np.log(T_kelvin)

    df = pd.DataFrame.from_dict({"T (°C)": T_celsius, "k (s⁻¹)": k, 
                    "ln(k/s⁻¹)": ln_k, "1/k (s)": 1/k,
                     "log(T/K)": log_T,
                    "1/T (K⁻¹)": T_inverse,
                    "T (K)": T_kelvin})

    

    st.dataframe(df[["T (°C)", "k (s⁻¹)"]].style
                    .format("{:.2e}", subset="k (s⁻¹)")
                    .format("{:.1f}", subset="T (°C)"))

    y_options = [x for x in df.columns if 'k' in x]
    x_options = [x for x in df.columns if 'T' in x]
    y_axis = st.selectbox("Y axis", y_options)

    x_axis = st.selectbox("X axis", x_options)

    fig = px.scatter(df, x=x_axis, y=y_axis, trendline='ols')
    
    results = px.get_trendline_results(fig)


    results = results.px_fit_results[0]

    b, m = results.params 

    R_squared = results.rsquared

    st.write(f"""Trendline
    
    y = {m:.2f}x + {b:.2f}
    R² = {R_squared:.4f}
    """)

    st.plotly_chart(fig)

    st.markdown("""
    ### Questions
    1. What is the activation energy for the reaction in kJ/mol?
    """)

    correct_Ea = 8196.29*8.3145 / 1000

    Ea_response = st.number_input("Activation energy (kJ/mol)", value=0.0)

    if Ea_response != 0.0 and abs(Ea_response-correct_Ea) > 0.2:
        st.write("Incorrect. Try again.")
    elif abs(Ea_response-correct_Ea) < 0.2:
        st.write("Correct!")

    st.markdown("2. What is the pre-exponential factor?")

    A_response = st.number_input("Pre-exponential factor:", value=0.0, format="%2e")
    
    correct_A = np.exp(12.07)

    if A_response != 0.0 and abs(A_response-correct_A) > 1e4:
        st.write("Incorrect. Try again.")
    elif abs(A_response-correct_A) < 0.2e4:
        st.write("Correct!")

if __name__ == '__main__':
    run()