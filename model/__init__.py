__all__ = [
    "DMD",
    "CausalDiffusion",
    "Stage2DMD",
]


def __getattr__(name):
    """Load only the model family requested by the selected trainer."""

    if name == "DMD":
        from .dmd import DMD

        return DMD
    if name == "CausalDiffusion":
        from .diffusion import CausalDiffusion

        return CausalDiffusion
    if name == "Stage2DMD":
        from .stage2_dmd import Stage2DMD

        return Stage2DMD
    raise AttributeError(name)
