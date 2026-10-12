"""Model registry — lazy imports so only the selected model's module loads.

`build_model` maps a registry key to a model class. Model modules are imported
on demand so that importing this module never pulls in unused dependencies.
"""

from __future__ import annotations

from torch import nn

_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "cnn_baseline": ("models.cnn_baseline", "MultiSensorFusionCNN"),
    "cnn_lstm": ("models.cnn_lstm", "DeepConvLSTM"),
    "multitask": ("models.multitask", "MultiTaskNet"),
    "se_convnet": ("models.attention", "SEConvNet"),
    "gait_vit": ("models.vit", "GaitViT"),
}


def build_model(name: str, model_args: dict, n_channels: int, n_classes: int) -> nn.Module:
    """Build the model registered under `name`.

    Args:
        name: registry key (e.g. "cnn_lstm").
        model_args: keyword arguments for the model constructor. The keys
            "n_channels" and "n_classes" are consumed by this function and
            stripped before the constructor call; "multitask" additionally
            requires "n_subjects" here.
        n_channels: number of sensor channels.
        n_classes: number of activity classes.

    Returns:
        The instantiated `nn.Module`.

    Raises:
        ValueError: if `name` is not a registered model, or if a required
            constructor argument is missing.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; available models: {sorted(_MODEL_REGISTRY)}"
        )

    module_name, class_name = _MODEL_REGISTRY[name]
    import importlib

    cls = getattr(importlib.import_module(module_name), class_name)

    args = dict(model_args or {})
    args.pop("n_channels", None)
    args.pop("n_classes", None)

    if name == "multitask":
        if "n_subjects" not in args:
            raise ValueError(
                "multitask model requires 'n_subjects' in model_args "
                "(the number of subject classes)"
            )
        n_subjects = args.pop("n_subjects")
        return cls(n_channels=n_channels, n_activities=n_classes,
                   n_subjects=n_subjects, **args)

    return cls(n_channels=n_channels, n_classes=n_classes, **args)
