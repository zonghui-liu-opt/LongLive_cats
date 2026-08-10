__all__ = [
    "CausalDiffusionInferencePipeline",
    "SelfForcingTrainingPipeline",
    "Stage2RolloutPipeline",
]


def __getattr__(name):
    if name == "CausalDiffusionInferencePipeline":
        from .causal_diffusion_inference import CausalDiffusionInferencePipeline

        return CausalDiffusionInferencePipeline
    if name == "SelfForcingTrainingPipeline":
        from .self_forcing_training import SelfForcingTrainingPipeline

        return SelfForcingTrainingPipeline
    if name == "Stage2RolloutPipeline":
        from .stage2_rollout import Stage2RolloutPipeline

        return Stage2RolloutPipeline
    raise AttributeError(name)
