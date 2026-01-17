"""Zeff - Effective Nuclear Charge"""
from pchemapps.periodic_table import periodic_table
from pchemapps.iso_electronic import iso_electronic
import streamlit as st

iso_electronic_ions = st.checkbox("View isoelectronic ions")

if iso_electronic_ions:
    iso_electronic()
else:
    periodic_table()
