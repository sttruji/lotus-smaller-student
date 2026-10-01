"""Shared YAML loading and command-line overrides, with no model imports."""
from pathlib import Path
import yaml


def load_hierarchical_yaml(config_path, _seen=None):
    path = Path(config_path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Cyclic config inheritance at {path}")
    seen.add(path)
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    bases = config.pop("base", [])
    if isinstance(bases, str):
        bases = [bases]
    merged = {}
    for base in bases:
        merged.update(load_hierarchical_yaml(path.parent / base, seen))
    merged.update(config)
    return merged


def apply_overrides(config, overrides):
    result = dict(config)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or key not in result:
            raise ValueError(f"Override must name an existing config key: {override}")
        result[key] = yaml.safe_load(value)
    return result


def accumulation_divisor(batch_index, batch_count, accumulation_steps):
    """Average over the actual window, including an incomplete final window."""
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    start = batch_index // accumulation_steps * accumulation_steps
    return min(accumulation_steps, batch_count - start)
