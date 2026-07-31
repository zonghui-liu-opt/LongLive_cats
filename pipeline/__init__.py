__all__ = [
    "CausalDiffusionInferencePipeline",
    "SelfForcingTrainingPipeline",
]


def __getattr__(name):
    if name == "CausalDiffusionInferencePipeline":
        from .causal_diffusion_inference import CausalDiffusionInferencePipeline

        return CausalDiffusionInferencePipeline
    if name == "SelfForcingTrainingPipeline":
        from .self_forcing_training import SelfForcingTrainingPipeline

        return SelfForcingTrainingPipeline
    raise AttributeError(name)
