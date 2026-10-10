"""Explicit policy for an authorized same-host baseline after migration."""

import os


def baseline_policy():
    policy = os.environ.get("QUICK_BASELINE_POLICY", "historical_replay")
    if policy not in ["historical_replay", "fresh_destination"]:
        raise ValueError("Unknown baseline policy")
    return policy


def requires_historical_replay():
    return baseline_policy() == "historical_replay"
