
import numpy as np
import streamlit as st
import combineCSV
import combineCSVElectrochem
import thermoFirstLaw
import thermoCalorimeter
import ace
import openai_ex
import solartronData
import thermoGas
import entropySplit
import plotExcel

st.title("Physical Chemistry Tools")

apps = {"Combine UV-Vis Data": combineCSV, 
        "Combine CSV Electrochem": combineCSVElectrochem,
        'Plot Excel Data': plotExcel,
        "Plot Solartron Data": solartronData,
        "1st Law of Thermodynamics": thermoFirstLaw,
        "2nd Law Calorimeter": thermoCalorimeter,
        "Mystery Gas": thermoGas,
        "Where is Equilibrium?": entropySplit,
        "Sympy Shell": ace,
        "AI Demo": openai_ex
# 'ACE Editor': ace
}

app = st.selectbox("Choose an application:", list(apps.keys()))

apps[app].run()