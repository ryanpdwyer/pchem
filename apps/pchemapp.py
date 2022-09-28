
import logging
import time
import numpy as np
import streamlit as st
import combineCSV
import combineRaman
import combineCSVElectrochem
import thermoFirstLaw
import thermoCalorimeter
import ace
import kinetics
import solartronData
import thermoGas
import impedance
import entropySplit
import plotExcel
import arrhen

import socket
import copy

from streamlit.script_run_context import get_script_run_ctx
from streamlit.server.server import Server


def _get_session():
    """Get the session object from Streamlit

    Returns:
        object: the session object for the current thread

    """
    # Hack to get the session object from Streamlit.

    ctx = get_script_run_ctx()

    session = None
    session_infos = Server.get_current()._session_info_by_id.items()

    session_id = None
    for id, session_info in session_infos:
        s = session_info.session
        if (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr):
            session_id = id

    if session_id is None:
        raise RuntimeError(
            "Oh no! Couldn't get your Streamlit Session object"
            'Are you doing something fancy with threads?')
    return session_id



@st.cache
def configLog():
        logging.basicConfig(filename='debug-log.log', encoding='utf-8',
                level=logging.INFO, force=True,
                    format='%(asctime)s.%(msecs)03d %(levelname)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)


configLog()

startTime = time.time_ns()/1e6
logging.info(f"Start Script - id: "+_get_session())


st.title("Physical Chemistry Tools")

apps = {"Combine UV-Vis Data": combineCSV, 
        "Combine Raman Data": combineRaman,
        "CHE 120 Kinetics": arrhen,
        "Combine CSV Electrochem": combineCSVElectrochem,
        'Plot Excel Data': plotExcel,
        "Plot Solartron Data": solartronData,
        "1st Law of Thermodynamics": thermoFirstLaw,
        "Kinetics - Ocean Optics": kinetics,
        "2nd Law Calorimeter": thermoCalorimeter,
        "Mystery Gas": thermoGas,
        "Where is Equilibrium?": entropySplit,
        "Sympy Shell": ace,
        "Impedance": impedance
}

app = st.selectbox("Choose an application:", list(apps.keys()))

apps[app].run()

runTime = (time.time_ns()/1e6-startTime)

logging.info(f"Run time: {runTime} ms\tApp: {app}")
