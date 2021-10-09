
import numpy as np
import streamlit as st
import combineCSV
import combineCSVElectrochem
import thermoFirstLaw
import ace
import openai_ex
import solartronData

st.title("Physical Chemistry Helpers")

apps = {"Combine UV-Vis Data": combineCSV, 
        "Combine CSV Electrochem": combineCSVElectrochem,
        "Plot Solartron Data": solartronData,
        "1st Law of Thermodynamics": thermoFirstLaw,
        "Sympy Shell": ace,
        "AI Demo": openai_ex
# 'ACE Editor': ace
}

app = st.selectbox("Choose an application:", list(apps.keys()))

apps[app].run()