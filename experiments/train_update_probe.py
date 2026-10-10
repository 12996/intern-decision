"""Representative update samples with an exact whole-language fallback."""

import hashlib

import torch


class LanguageUpdateProbe:
    def __init__(self, module, full_digest, baseline_digest, max_parameters=12):
        self.module = module
        self.full_digest = full_digest
        self.baseline_digest = baseline_digest
        trainable = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
        candidates = [(name, p) for name, p in trainable if "embed" not in name.lower()]
        candidates = candidates or trainable
        if not candidates:
            raise ValueError("No trainable language parameters")
        candidates.sort(key=lambda item: hashlib.sha256(item[0].encode()).hexdigest())
        chosen = candidates[:max_parameters]
        self.regions = {}
        for name, parameter in chosen:
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            count = local.numel()
            if count == 0:
                continue
            width = min(32, count)
            starts = sorted({0, max(0, count // 2 - width // 2), count - width})
            self.regions[name] = [(start, width) for start in starts]
        if not self.regions:
            raise ValueError("No local trainable language coordinates")
        self.before = self.capture()

    def capture(self):
        parameters = dict(self.module.named_parameters())
        values = {}
        for name, regions in self.regions.items():
            parameter = parameters[name]
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            flat = local.detach().reshape(-1)
            values[name] = torch.cat([flat[start:start + width] for start, width in regions]).float().cpu().clone()
        return values

    def check(self, step, verify_full=False):
        current = self.capture()
        changes = {}
        for name, before in self.before.items():
            if before.shape != current[name].shape or not torch.isfinite(current[name]).all():
                raise ValueError("Language update probe shape/nonfinite mismatch")
            changes[name] = float((current[name] - before).abs().max())
        maximum = max(changes.values())
        full = self.full_digest() if verify_full or (step >= 3 and maximum == 0) else None
        return {"step": step, "sample_max_abs_delta": maximum,
                "changed_sample_parameters": [name for name, value in changes.items() if value > 0],
                "sampled_parameter_count": len(changes), "full_hash_checked": full is not None,
                "before_sha256": self.baseline_digest, "after_sha256": full,
                "update_detected": maximum > 0 or (full is not None and full != self.baseline_digest)}
