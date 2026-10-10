"""Rebind SwapAdamW's cached group references after optimizer restoration."""


def assert_swap_param_groups(optimizer):
    cached = getattr(optimizer, "_param_to_group_map", None)
    if cached is None:
        return {"applicable": False}
    current = {p: group for group in optimizer.param_groups for p in group["params"]}
    if current.keys() != cached.keys() or any(cached[p] is not group for p, group in current.items()):
        raise ValueError("SwapAdamW cached groups no longer track live optimizer groups")
    return {"applicable": True, "same_group_objects": True,
            "learning_rates": [group["lr"] for group in optimizer.param_groups]}


def rebind_swap_param_groups(optimizer):
    cached = getattr(optimizer, "_param_to_group_map", None)
    if cached is None:
        return {"applicable": False}
    if not isinstance(cached, dict):
        raise ValueError("Unexpected SwapAdamW group map")
    rebound = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter in rebound:
                raise ValueError("Duplicate optimizer parameter")
            rebound[parameter] = group
    if rebound.keys() != cached.keys():
        raise ValueError("Restored optimizer parameter identity changed")
    stale = sum(cached[p] is not group for p, group in rebound.items())
    optimizer._param_to_group_map = rebound
    assert_swap_param_groups(optimizer)
    return {"applicable": True, "parameters": len(rebound), "stale_groups_rebound": stale,
            "learning_rates": [group["lr"] for group in optimizer.param_groups]}
