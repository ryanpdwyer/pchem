
import numpy as np
import streamlit as st
import combineCSV

st.title("Physical Chemistry Helpers")

apps = {"Combine CSV": combineCSV}

app = st.selectbox("Choose an application:", list(apps.keys()))

apps[app].run()