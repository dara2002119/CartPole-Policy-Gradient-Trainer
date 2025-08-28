# CartPole-Policy-Gradient-Trainer
A compact policy-gradient trainer for Gymnasium’s CartPole-v1 built with PyTorch. It tracks a performance budget (reward threshold) and adapts exploration, entropy regularization, and learning-rate smoothly based on training progress. Observation statistics are tracked online and used to normalize inputs for stable learning.
