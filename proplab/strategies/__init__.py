"""Strategy registry. To add a strategy: write a module with DEFAULT_PARAMS and
generate_signals(df, params) -> DataFrame[idx, side, stop_dist, target_dist, exit_time],
then add it here. `idx` is the bar whose CLOSE triggered the signal."""
from . import futures_orb, sol_momentum

REGISTRY = {
    "orb": futures_orb,
    "sol_momentum": sol_momentum,
}


def get(name: str):
    if name not in REGISTRY:
        raise SystemExit(f"unknown strategy '{name}'. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name]
