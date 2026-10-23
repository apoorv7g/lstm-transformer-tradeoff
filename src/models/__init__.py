"""Model registry: build any model in the study by preset name."""

from . import hybrid, lstm, transformer

_BUILDERS = {}
for _module, _builder in (
    (lstm, lstm.build_lstm),
    (transformer, transformer.build_transformer),
    (hybrid, hybrid.build_hybrid),
):
    for _name, _kwargs in _module.PRESETS.items():
        _BUILDERS[_name] = (_builder, _kwargs)

PRESET_NAMES = tuple(_BUILDERS)


def build_model(preset, num_frames=20, **overrides):
    """Build a model by preset name, e.g. build_model("transformer_base", num_frames=40)."""
    if preset not in _BUILDERS:
        raise KeyError(f"unknown preset {preset!r}; choose from {PRESET_NAMES}")
    builder, kwargs = _BUILDERS[preset]
    return builder(num_frames=num_frames, name=preset, **{**kwargs, **overrides})
