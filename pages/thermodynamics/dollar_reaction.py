"""Dollar Game → Reaction: molecules as players, wealth bins as L vs R."""
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pchemapps.dollar_game import (
    boltzmann_curve,
    concentrate_wealth,
    cool_from_top,
    halve_wealth,
    play_rounds,
)


def _rotate(xs: np.ndarray, ys: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(angle), np.sin(angle)
    return c * xs - s * ys, s * xs + c * ys


def draw_entropy_icons(wealth: np.ndarray, threshold: int):
    """Draw one exemplar of each form. Wealth (temperature) controls *kinetics*,
    not shape diversity — R jiggles and rotates a bit faster at higher ⟨$⟩,
    L keeps the same flexibility but tumbles faster and swaps conformations more
    often."""
    rng = np.random.default_rng()
    t = time.time()
    fig, (ax_R, ax_L) = plt.subplots(1, 2, figsize=(8, 2.8))

    R_idx = np.flatnonzero(wealth < threshold)
    L_idx = np.flatnonzero(wealth >= threshold)

    # --- Ring: 0 quanta = perfect hexagon; $1 adds small jitter ---
    ax_R.set_aspect("equal"); ax_R.axis("off")
    ax_R.set_xlim(-0.7, 0.7); ax_R.set_ylim(-0.7, 0.7)
    if R_idx.size:
        mean_R = float(wealth[R_idx].mean())
        n_atoms = 6
        thetas = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
        jitter_amp = 0.022 * mean_R           # 0 if all ring molecules have $0
        jitter = jitter_amp * rng.normal(size=(n_atoms, 2))
        xs = 0.4 * np.cos(thetas) + jitter[:, 0]
        ys = 0.4 * np.sin(thetas) + jitter[:, 1]
        xs, ys = _rotate(xs, ys, 0.1 * (1 + mean_R) * t)
        ax_R.plot(np.append(xs, xs[0]), np.append(ys, ys[0]),
                  "-", color="tab:blue", lw=2)
        ax_R.plot(xs, ys, "o", color="tab:blue", markersize=10)
        ax_R.set_title(f"Ring (R) — ⟨\\${mean_R:.1f}⟩", fontsize=12)
    else:
        ax_R.set_title("Ring (R) — none present", fontsize=11)

    # --- Linear: sp3-like ±70.5° bends (109.5° interior angle) with thermal wobble;
    # tumbling + conformation-swap rate scale with ⟨$⟩ ---
    ax_L.set_aspect("equal"); ax_L.axis("off")
    ax_L.set_xlim(-0.9, 0.9); ax_L.set_ylim(-0.9, 0.9)
    if L_idx.size:
        mean_L = float(wealth[L_idx].mean())
        n_beads = 6
        bond = 0.18
        bend_nominal = np.deg2rad(70.5)       # 180° − 109.5°
        wobble = 0.10 + 0.05 * mean_L         # rad; small at low T, larger at high T
        # Hold each conformation for roughly this long; shorter at higher T
        persistence = max(0.15, 2.5 / (1 + mean_L))
        seed_bucket = int(t / persistence)
        rng_L = np.random.default_rng(seed_bucket * 7919 + 101)
        signs = rng_L.choice([-1.0, 1.0], size=n_beads - 1)
        angles = signs * bend_nominal + rng_L.normal(0, wobble, n_beads - 1)
        heading = 0.0
        xs = [0.0]; ys = [0.0]
        for a in angles:
            heading += a
            xs.append(xs[-1] + bond * np.cos(heading))
            ys.append(ys[-1] + bond * np.sin(heading))
        xs = np.array(xs) - np.mean(xs)
        ys = np.array(ys) - np.mean(ys)
        rot_rate = 0.15 + 0.15 * mean_L       # slow at $2, fast at $10
        xs, ys = _rotate(xs, ys, rot_rate * t)
        ax_L.plot(xs, ys, "-", color="tab:orange", lw=2)
        ax_L.plot(xs, ys, "o", color="tab:orange", markersize=10)
        ax_L.set_title(f"Linear (L) — ⟨\\${mean_L:.1f}⟩", fontsize=12)
    else:
        ax_L.set_title("Linear (L) — none present", fontsize=11)

    fig.tight_layout()
    return fig


st.page_link("pages/home.py", label="← Home")

st.markdown(r"""# Dollar Game: Reaction

$$L \rightleftharpoons R$$

Molecules swap \$1 via rock-paper-scissors. Those with ≥ \$2 are **linear (L)**;
those with < \$2 are **ring (R)**. $Q = [R]/[L]$, and $Q = K$ at equilibrium.
""")

mode = st.radio(
    "Focus",
    ["Stress test", "Entropy", "Advanced"],
    index=0,
    horizontal=True,
    help=(
        "Stress test: start at equilibrium, perturb, watch relaxation.  "
        "Entropy: visualize why L and R differ.  "
        "Advanced: show everything."
    ),
)

# ---- Sidebar controls ----
N = st.sidebar.slider("Number of molecules", min_value=20, max_value=500, value=50, step=10)
starting_dollars = st.sidebar.slider("Starting dollars per molecule", min_value=0, max_value=10, value=2, step=1)
show_K = st.sidebar.checkbox("Show predicted $K$", value=False)

with st.sidebar.expander("Advanced", expanded=False):
    threshold = st.slider("Linear threshold (dollars ≥)", min_value=1, max_value=6, value=2, step=1)
    swaps_per_frame = st.slider("Rounds per frame", min_value=1, max_value=5000, value=max(N // 2, 1), step=1)
    frame_delay = st.slider("Frame delay (s)", min_value=0.3, max_value=1.5, value=0.75, step=0.05)
    show_stem = st.checkbox("Show energy distribution", value=True)
    history_len = st.slider("Q history length", min_value=50, max_value=2000, value=400, step=50)

# ---- Session state ----
def _q_from(wealth: np.ndarray) -> float:
    n_L = int(np.sum(wealth >= threshold))
    n_R = len(wealth) - n_L
    return (n_R / n_L) if n_L > 0 else float("inf")


def _k_from(wealth: np.ndarray) -> float:
    m = float(wealth.mean())
    if m <= 0:
        return float("inf")
    x = m / (1 + m)
    P_L = x**threshold
    return (1 - P_L) / P_L if P_L > 0 else float("inf")


def _record(state: dict) -> None:
    state["Q_history"].append(_q_from(state["wealth"]))
    state["K_history"].append(_k_from(state["wealth"]))
    state["round_history"].append(state["rounds"])


if ("reaction" not in st.session_state
        or st.session_state.reaction["N"] != N
        or "K_history" not in st.session_state.reaction):
    st.session_state.reaction = dict(
        wealth=np.full(N, starting_dollars, dtype=int),
        rounds=0,
        running=False,
        N=N,
        Q_history=[],
        K_history=[],
        round_history=[],
        last_sim_time=0.0,
    )
    _record(st.session_state.reaction)

state = st.session_state.reaction

# ---- Buttons ----
c1, c2 = st.sidebar.columns(2)
run_button = c1.button("Pause" if state["running"] else "Run")
reset_button = c2.button("Reset")

c3, c4 = st.sidebar.columns(2)
add_button = c3.button("Increase temp")
halve_button = c4.button("Decrease temp")

preeq_button = False
concentrate_button = False
cool_top_button = False
if mode == "Stress test":
    st.sidebar.markdown("**Perturbations**")
    preeq_button = st.sidebar.button("Pre-equilibrate (5000 rounds)")
    concentrate_button = st.sidebar.button(
        "Concentrate \\$ on one",
        help="All dollars on a single molecule. Q spikes high, then relaxes to K.",
    )
    cool_top_button = st.sidebar.button(
        "Cool from top (halve total)",
        help="Remove half the total \\$ from the richest players first. Q barely moves at first, then drifts to the new (higher) K.",
    )

if run_button:
    state["running"] = not state["running"]
    st.rerun()

if reset_button:
    state["wealth"] = np.full(N, starting_dollars, dtype=int)
    state["rounds"] = 0
    state["running"] = False
    state["Q_history"] = []
    state["K_history"] = []
    state["round_history"] = []
    _record(state)
    st.rerun()

if add_button:
    state["wealth"] += 1
    _record(state)
    st.rerun()

if halve_button:
    state["wealth"] = halve_wealth(state["wealth"])
    _record(state)
    st.rerun()

if preeq_button:
    with st.spinner("Pre-equilibrating..."):
        play_rounds(state["wealth"], 5000)
        state["rounds"] += 5000
        _record(state)
    st.rerun()

if concentrate_button:
    state["wealth"] = concentrate_wealth(state["wealth"])
    _record(state)
    st.rerun()

if cool_top_button:
    total = int(state["wealth"].sum())
    state["wealth"] = cool_from_top(state["wealth"], total // 2)
    _record(state)
    st.rerun()

# ---- Current state ----
wealth = state["wealth"]
total = int(wealth.sum())
mean = total / N
N_L = int(np.sum(wealth >= threshold))
N_R = N - N_L
Q = (N_R / N_L) if N_L > 0 else float("inf")

# Predicted K from Boltzmann geometric distribution
if mean > 0:
    x = mean / (1 + mean)
    P_L = x**threshold
    K_pred = (1 - P_L) / P_L if P_L > 0 else float("inf")
else:
    K_pred = float("inf")

# ---- Headline metrics ----
status = "running" if state["running"] else "paused"
st.markdown(
    f"**Status:** {status} &nbsp;&nbsp; **Rounds played:** {state['rounds']:,} &nbsp;&nbsp; "
    f"**Total money:** \\${total} &nbsp;&nbsp; **Average:** \\${mean:.2f}/molecule"
)

if mode in ("Entropy", "Advanced"):
    st.pyplot(draw_entropy_icons(wealth, threshold), use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric(f"[L]  (dollars ≥ {threshold})", f"{N_L}", help=f"Fraction: {N_L/N:.2f}")
m2.metric(f"[R]  (dollars < {threshold})", f"{N_R}", help=f"Fraction: {N_R/N:.2f}")
Q_label = f"{Q:.3f}" if np.isfinite(Q) else "∞"
m3.metric("Q = [R]/[L]", Q_label,
          delta=f"K ≈ {K_pred:.3f}" if show_K and np.isfinite(K_pred) else None,
          delta_color="off")

# ---- Q vs K number line ----
if mode in ("Stress test", "Advanced") and show_K and np.isfinite(K_pred) and K_pred > 0:
    Q_plot = Q if np.isfinite(Q) else K_pred * 5  # cap infinities for drawing
    x_max = max(Q_plot, K_pred) * 1.4 + 0.2
    fig_num, ax_num = plt.subplots(figsize=(7, 1.2))
    ax_num.hlines(0, 0, x_max, color="0.5", lw=1)
    # K marker
    ax_num.axvline(K_pred, color="tab:red", ls="--", lw=2)
    ax_num.text(K_pred, 0.35, f"K = {K_pred:.2f}", color="tab:red",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    # Q marker
    ax_num.plot(Q_plot, 0, "o", color="tab:blue", markersize=14, zorder=5)
    q_text = f"Q = {Q:.2f}" if np.isfinite(Q) else "Q = ∞"
    ax_num.text(Q_plot, -0.55, q_text, color="tab:blue",
                ha="center", va="top", fontsize=11, fontweight="bold")
    # Arrow from Q toward K if off
    if np.isfinite(Q) and abs(Q - K_pred) > 0.02 * x_max:
        direction = "→ more R" if Q < K_pred else "← more L"
        ax_num.annotate("", xy=(K_pred, 0.05), xytext=(Q_plot, 0.05),
                        arrowprops=dict(arrowstyle="->", color="tab:green", lw=2))
        midx = 0.5 * (Q_plot + K_pred)
        ax_num.text(midx, 0.18, direction, color="tab:green",
                    ha="center", va="bottom", fontsize=10)
    ax_num.set_xlim(0, x_max)
    ax_num.set_ylim(-0.9, 0.9)
    ax_num.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_num.spines[spine].set_visible(False)
    ax_num.set_xlabel("Q, K")
    fig_num.tight_layout()
    st.pyplot(fig_num, use_container_width=True)

# ---- Stem plot (optional) ----
if mode in ("Entropy", "Advanced") and show_stem:
    max_d = int(wealth.max())
    levels = np.arange(0, max_d + 1)
    counts = np.bincount(wealth, minlength=max_d + 1)
    fraction = counts / N
    is_L = levels >= threshold

    fig, ax = plt.subplots(figsize=(7, 3.5))
    # R side
    if np.any(~is_L):
        ml, _, _ = ax.stem(levels[~is_L], fraction[~is_L], basefmt=" ",
                           linefmt="tab:blue", markerfmt="o", label="Ring (R)")
        ml.set_color("tab:blue")
        ml.set_markersize(7)
    # L side
    if np.any(is_L):
        ml, _, _ = ax.stem(levels[is_L], fraction[is_L], basefmt=" ",
                           linefmt="tab:orange", markerfmt="s", label="Linear (L)")
        ml.set_color("tab:orange")
        ml.set_markersize(7)
    ax.axvline(threshold - 0.5, color="0.6", ls=":", lw=1)
    ax.set_xlabel("Dollars")
    ax.set_ylabel("Fraction of molecules")
    ax.set_xlim(-0.5, max(max_d + 0.5, 5))
    ymax_data = fraction.max() if fraction.size else 0.1
    ax.set_ylim(0, max(ymax_data * 1.15, 0.1))
    ax.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---- Q vs rounds time series ----
if mode in ("Stress test", "Advanced") and state["Q_history"]:
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(state["round_history"], state["Q_history"], color="tab:green", lw=1.5, label="Q")
    if show_K:
        K_arr = np.array(state["K_history"], dtype=float)
        K_arr[~np.isfinite(K_arr)] = np.nan
        ax2.plot(state["round_history"], K_arr, color="tab:red", ls="--", lw=1.5,
                 drawstyle="steps-post", label="K (predicted)")
    ax2.set_xlabel("Rounds played")
    ax2.set_ylabel("Q, K  ([R] / [L])")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

# ---- Advance simulation (must be last) ----
# Simulation steps fire at `frame_delay`; the page reruns much faster so the
# entropy icons keep tumbling smoothly even between simulation steps.
animate_icons = mode in ("Entropy", "Advanced")
now = time.time()
if state["running"] and (now - state.get("last_sim_time", 0.0)) >= frame_delay:
    play_rounds(state["wealth"], swaps_per_frame)
    state["rounds"] += swaps_per_frame
    _record(state)
    if len(state["Q_history"]) > history_len:
        state["Q_history"] = state["Q_history"][-history_len:]
        state["K_history"] = state["K_history"][-history_len:]
        state["round_history"] = state["round_history"][-history_len:]
    state["last_sim_time"] = now

if state["running"] or animate_icons:
    time.sleep(0.1)
    st.rerun()
