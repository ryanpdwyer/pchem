"""Plot Excel Data"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.plotExcel import run
run()
