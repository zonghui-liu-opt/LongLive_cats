__all__ = [
    "DMD",
    "CausalDiffusion",
]


def __getattr__(name):
    """Load only the model family requested by the selected trainer."""

    if name == "DMD":
        from .dmd import DMD

        return DMD
    if name == "CausalDiffusion":
        from .diffusion import CausalDiffusion

        return CausalDiffusion
    raise AttributeError(name)
