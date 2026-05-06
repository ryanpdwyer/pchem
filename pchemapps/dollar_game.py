"""Shared utilities for the Boltzmann Dollar Game and its variants."""
import numpy as np


def play_rounds(wealth: np.ndarray, n_rounds: int) -> None:
    """In-place random $1 exchange. Each round a giver and receiver are picked
    uniformly at random; the giver transfers $1 if they have any. Total wealth
    is conserved."""
    N = len(wealth)
    givers = np.random.randint(0, N, n_rounds)
    receivers = np.random.randint(0, N, n_rounds)
    for g, r in zip(givers, receivers):
        if g != r and wealth[g] > 0:
            wealth[g] -= 1
            wealth[r] += 1


def halve_wealth(wealth: np.ndarray) -> np.ndarray:
    """Cut each player's wealth in half, with random rounding on odd values so
    that the expected total removed is exactly half. Returns a new array."""
    halved = wealth // 2
    odd_mask = (wealth % 2 == 1)
    n_odd = int(odd_mask.sum())
    if n_odd:
        halved[odd_mask] = halved[odd_mask] + np.random.randint(0, 2, n_odd)
    return halved


def concentrate_wealth(wealth: np.ndarray) -> np.ndarray:
    """Put every dollar on a single player; total is preserved."""
    total = int(wealth.sum())
    new = np.zeros_like(wealth)
    if len(new):
        new[0] = total
    return new


def cool_from_top(wealth: np.ndarray, amount: int) -> np.ndarray:
    """Remove `amount` dollars, always taking from the current richest player.
    Leaves the lower tail of the distribution (and thus Q) nearly unchanged."""
    w = wealth.copy()
    remaining = int(amount)
    while remaining > 0 and w.sum() > 0:
        idx = int(np.argmax(w))
        take = min(int(w[idx]), remaining)
        w[idx] -= take
        remaining -= take
    return w


def boltzmann_curve(mean: float, max_level: int) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrium geometric distribution for mean wealth: p(k) = (1 - x) x^k
    with x = mean / (1 + mean). Returns (levels, probabilities)."""
    levels = np.arange(0, max_level + 1)
    if mean <= 0:
        p = np.zeros_like(levels, dtype=float)
        p[0] = 1.0
        return levels, p
    x = mean / (1 + mean)
    p = (1 - x) * x**levels
    return levels, p
