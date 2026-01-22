"""Where is Equilibrium? - Entropy and Statistical Mechanics"""
import streamlit as st
from copy import copy
import numpy as np
from scipy.special import factorial

st.page_link("pages/home.py", label="← Home")


def W(ns):
    N = sum(ns)
    return factorial(N)/np.prod(factorial(ns))


def display_level(n):
    return f"<p>{n}</p><hr>"


def display_levels(levels):
    text = [display_level(n) for n in reversed(levels)]
    html = "\n".join(text)
    return f"""<div class="entropy">
{html}
</div>"""


def display_all(left, right, ways, entropy, totals):
    cols = st.columns(2)

    cols[0].markdown(display_levels(left), unsafe_allow_html=True)

    W_L, W_R = W(left), W(right)

    S_L, S_R = np.log(W_L), np.log(W_R)

    cols[1].markdown(display_levels(right), unsafe_allow_html=True)

    if ways:
        cols[0].markdown(f"$W_\\text{{left}}=$ {int(W_L)}")
        cols[1].markdown(f"$W_\\text{{right}}=$ {int(W_R)}")

    if entropy:
        cols[0].markdown(f"$S_\\text{{left}}= {S_L:.2f}k_B$")
        cols[1].markdown(f"$S_\\text{{right}}= {S_R:.2f}k_B$")

    if totals:
        st.markdown(f"$W_\\text{{total}}=$ {int(W_L*W_R)}")
        st.markdown(f"$S_\\text{{total}}= {S_L+S_R:.2f}k_B$")


# Page content starts here
st.markdown("## Where is equilibrium?")

showW = st.sidebar.checkbox("Show ways?")
showS = st.sidebar.checkbox("Show entropy?")
showTotals = st.sidebar.checkbox("Show totals?")

left0 = [10, 6]
right0 = [26, 6]

left = copy(left0)
right = copy(right0)

st.markdown("""
<style>
.entropy {
    max-width: 200px;
    text-align: center;
}

.entropy hr {
    border: 1px solid black;
}

</style>
""", unsafe_allow_html=True)

st.markdown("### State")

container = st.container()

units = st.slider("Transfer energy", value=0, min_value=-6, max_value=6)

left[0] = left0[0] + units
left[1] = left0[1] - units
right[0] = right0[0] - units
right[1] = right[1] + units

with container:
    display_all(left, right, showW, showS, showTotals)
