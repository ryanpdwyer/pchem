
import numpy as np
import streamlit as st
import combineCSV
import combineCSVElectrochem
import thermoFirstLaw
import ace
import openai_ex

st.title("Physical Chemistry Helpers")

apps = {"Combine CSV": combineCSV, 
        "Combine CSV Electrochem": combineCSVElectrochem,
        "1st Law of Thermodynamics": thermoFirstLaw,
        "AI Demo": openai_ex
# 'ACE Editor': ace
}

app = st.selectbox("Choose an application:", list(apps.keys()))

apps[app].run()