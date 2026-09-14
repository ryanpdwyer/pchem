"""BioTek ABTS plate reader -> Excel"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.biotek_abts import run
run()
