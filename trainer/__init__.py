__all__ = [
    "ScoreDistillationTrainer",
    "DiffusionTrainer",
]


def __getattr__(name):
    """Keep public trainer names while avoiding unrelated eager imports."""

    if name == "ScoreDistillationTrainer":
        from .distillation import Trainer

        return Trainer
    if name == "DiffusionTrainer":
        from .diffusion import Trainer

        return Trainer
    raise AttributeError(name)
