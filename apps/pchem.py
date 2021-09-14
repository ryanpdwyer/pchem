
import numpy as np
import streamlit as st
import combineCSV
import combineCSVElectrochem
import ace

st.title("Physical Chemistry Helpers")

apps = {"Combine CSV": combineCSV, 
        "Combine CSV Electrochem": combineCSVElectrochem
# 'ACE Editor': ace
}

app = st.selectbox("Choose an application:", list(apps.keys()))

apps[app].run()