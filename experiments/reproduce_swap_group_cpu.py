"""CPU-only reproduction of stale SwapAdamW groups after state restore."""

import copy
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


SOURCE = Path("/home/zhaopu/intern-decision/Intern-Decision/third_party/xtuner/xtuner/v1/optim/swap_adamw.py")
OUTPUT = Path("/home/zhaopu/intern-decision/outputs/goal-20261008-rebase-v1/swap_group_cpu_reproduction.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("This reproduction must run with CUDA hidden")
    utils = ModuleType("xtuner.v1.utils")
    utils.get_device = lambda: "cpu"
    utils.get_torch_device_module = lambda: SimpleNamespace(synchronize=lambda: None, current_device=lambda: 0)
    utils.get_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None)
    spec = importlib.util.spec_from_file_location("cpu_fixture_swap_adamw", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"xtuner.v1.utils": utils}), \
            patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
        spec.loader.exec_module(module)
        parameter = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
        optimizer = module.SwapAdamW([parameter], lr=0.0, fused=False, foreach=False)
        before_shared = optimizer._param_to_group_map[parameter] is optimizer.param_groups[0]
        state = copy.deepcopy(optimizer.state_dict())
        optimizer.load_state_dict(state)
        after_shared = optimizer._param_to_group_map[parameter] is optimizer.param_groups[0]
        optimizer.param_groups[0]["lr"] = 1e-3
        scheduled_lr = optimizer.param_groups[0]["lr"]
        stale_effective_lr = optimizer._param_to_group_map[parameter]["lr"]
        before = parameter.detach().clone()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        stale_delta = float((parameter.detach() - before).abs().max())
        stale_step = float(optimizer.state[parameter]["step"])
        stale_moment = float(optimizer._param_to_cpu_states_map[parameter]["exp_avg"].abs().max())
        from swap_restore_fix import rebind_swap_param_groups
        repair = rebind_swap_param_groups(optimizer)
        optimizer.step()
        rebound_delta = float((parameter.detach() - before).abs().max())
    result = {"cpu_only": True, "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "torch": torch.__version__, "before_load_groups_shared": before_shared,
              "after_load_groups_shared": after_shared, "scheduled_lr": scheduled_lr,
              "stale_effective_lr": stale_effective_lr, "stale_parameter_delta": stale_delta,
              "stale_step_advanced": stale_step, "stale_moment_abs_max": stale_moment,
              "rebound_parameter_delta": rebound_delta,
              "repair": repair,
              "scope": "Actual served SwapAdamW class; CPU adapter, plain optimizer restore, not a2BGPU/DCP fix verification"}
    assert before_shared and not after_shared and scheduled_lr > 0 and stale_effective_lr == 0
    assert stale_delta == 0 and stale_step == 1 and stale_moment > 0 and rebound_delta > 0
    result["result"] = "pass"
    output = OUTPUT.with_name(OUTPUT.stem + "_fixed.json") if args.fixed else OUTPUT
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
