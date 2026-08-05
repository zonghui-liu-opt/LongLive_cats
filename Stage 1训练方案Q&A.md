# Stage 1训练方案Q&A

## 0. Task定义

- 从已有的双向猫咪 Wan2.2‑TI2V‑5B 权重出发，用 causal block attention 改造成“每次生成8个 latent frame”的块级自回归 I2V 模型，并只训练 LoRA；先用真实干净历史做 Teacher Forcing，再用 Error Recycling 模拟推理时的历史误差。

## 1. 问题

- 双向模型和Block-wise AR模型的forward方式具体有哪些不同? 
- 是否使用了首帧clean latent作为attention sink?
- 