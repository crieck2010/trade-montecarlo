"""Block bootstrap: resample history instead of assuming a model.

Cuts the historical return matrix into overlapping blocks and strings
random blocks together into new paths.  Within-block autocorrelation and
cross-asset correlation survive; the model zoo's distributional
assumptions don't enter.  The honest way to ask "how lucky was this
backtest?" — see ``bootstrap_for_backtest`` in adapters.
"""

from __future__ import annotations

from .rng import RandomStream


def block_bootstrap(returns: list[list[float]], n_paths: int,
                    block_len: int = 20, seed: int = 7) -> list[list[list[float]]]:
    """Resample T×N returns into ``n_paths`` new T×N matrices.

    Circular blocks: a block starting near the end wraps around, so every
    observation is equally likely to lead a block.
    """
    t = len(returns)
    if t == 0 or block_len <= 0 or block_len > t:
        raise ValueError("need 0 < block_len <= len(returns)")
    stream = RandomStream(seed)
    out = []
    for _ in range(n_paths):
        path = []
        while len(path) < t:
            start = int(stream.uniform() * t)
            for i in range(block_len):
                path.append(returns[(start + i) % t])
                if len(path) == t:
                    break
        out.append(path)
    return out
