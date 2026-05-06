"""Boltzmann Dollar Game: how random exchange produces an exponential wealth distribution."""
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pchemapps.dollar_game import boltzmann_curve, halve_wealth, play_rounds

st.page_link("pages/home.py", label="← Home")

st.markdown(r"""# The Boltzmann Dollar Game

Imagine $N$ players in a room. Each player starts with some dollars.
Every round, two players are paired at random and play **rock-paper-scissors**.
The loser hands \$1 to the winner — unless the loser has \$0, in which case
nothing happens. Nobody can go negative, and the total amount of money in the
room never changes.

Even though each exchange is completely random, the distribution of wealth
doesn't stay flat — it settles into a characteristic shape with many people
near \$0 and a few players who happen to be rich. That shape is the
**Boltzmann distribution**, and it shows up all over chemistry and physics
whenever particles share a fixed amount of something (like energy).

Press **Run** below and watch it happen.
""")


# ---- Sidebar controls ----
N = st.sidebar.slider("Number of players", min_value=20, max_value=500, value=50, step=10)
starting_dollars = st.sidebar.slider("Starting dollars per player", min_value=0, max_value=10, value=1, step=1)
show_theory = st.sidebar.checkbox("Overlay Boltzmann curve", value=True)

with st.sidebar.expander("Advanced", expanded=False):
    # Default: N/2 transactions per frame — on average each player participates
    # in one collision (as giver or receiver) per frame.
    swaps_per_frame = st.slider("Rounds per frame", min_value=1, max_value=5000, value=max(N // 2, 1), step=1)
    frame_delay = st.slider("Frame delay (s)", min_value=0.3, max_value=1.5, value=0.75, step=0.05)

# ---- Session state (reset when N changes) ----
if "dollars" not in st.session_state or st.session_state.dollars["N"] != N:
    st.session_state.dollars = dict(
        wealth=np.full(N, starting_dollars, dtype=int),
        rounds=0,
        running=False,
        N=N,
    )

state = st.session_state.dollars

# ---- Buttons ----
c1, c2 = st.sidebar.columns(2)
run_button = c1.button("Pause" if state["running"] else "Run")
reset_button = c2.button("Reset")

c3, c4 = st.sidebar.columns(2)
add_button = c3.button("Increase temp")
halve_button = c4.button("Decrease temp")

if run_button:
    state["running"] = not state["running"]
    st.rerun()

if reset_button:
    state["wealth"] = np.full(N, starting_dollars, dtype=int)
    state["rounds"] = 0
    state["running"] = False
    st.rerun()

if add_button:
    state["wealth"] += 1
    st.rerun()

if halve_button:
    state["wealth"] = halve_wealth(state["wealth"])
    st.rerun()

# ---- Current state ----
wealth = state["wealth"]
total = int(wealth.sum())
mean = total / N

status = "running" if state["running"] else "paused"
st.markdown(
    f"**Status:** {status} &nbsp;&nbsp; **Rounds played:** {state['rounds']:,} &nbsp;&nbsp; "
    f"**Total money:** \\${total} &nbsp;&nbsp; "
    f"**Average wealth:** \\${mean:.2f}"
)

# ---- Stem plot of wealth distribution ----
max_d = int(wealth.max())
levels = np.arange(0, max_d + 1)
counts = np.bincount(wealth, minlength=max_d + 1)
fraction = counts / N

fig, ax = plt.subplots(figsize=(7, 4))
markerline, _, _ = ax.stem(levels, fraction, basefmt=" ", label="players")
markerline.set_markersize(7)
ax.set_xlabel("Dollars")
ax.set_ylabel("Fraction of players")
ax.set_xlim(-0.5, max(max_d + 0.5, 5))
ax.set_ylim(0, max(fraction.max() * 1.15, 0.1))

if show_theory and mean > 0:
    ks, p_theory = boltzmann_curve(mean, max_d)
    ax.plot(ks, p_theory, "o--", color="tab:red", alpha=0.75,
            label="Boltzmann prediction")
    ax.legend(loc="upper right")

ax.set_title(f"{N} players, \\${total} in circulation")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

with st.expander("Raw wealth array"):
    st.write(wealth)

# ---- Advance simulation (must be last) ----
if state["running"]:
    play_rounds(state["wealth"], swaps_per_frame)
    state["rounds"] += swaps_per_frame
    time.sleep(frame_delay)
    st.rerun()
