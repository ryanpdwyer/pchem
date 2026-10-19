"""NMR peak picking (Anasazi EFT-90 NUTS files)"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.nmr import run
run()
