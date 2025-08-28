# CartPole Policy-Gradient Trainer

<img src="https://upload.wikimedia.org/wikipedia/commons/2/2c/Cart-pendulum.svg" align="right" alt="CartPole diagram" width="140" height="140">

A compact **policy-gradient** trainer for **Gymnasium’s CartPole-v1** built with **PyTorch**.  
It tracks a performance budget (reward threshold) and **adapts exploration, entropy regularization, and learning-rate** smoothly based on training progress. Observation statistics are tracked online and used to normalize inputs for stable learning.

- **REINFORCE with entropy bonus**, return normalization, and gradient clipping.  
- **Observation normalization** via running mean/std (Welford/EMA-style).  
- **Stateful, smoothed adapters** for temperature/entropy/LR with hysteresis near the goal.  
- **EMA and sliding window** performance metrics for stable decisions and readable logs.  
- Optional **human render** via `pygame` for visual inspection.

---

## Who Might Use This

- Students learning policy gradients and training hygiene.  
- Practitioners wanting a tiny, legible baseline for CartPole.  
- Folks comparing **adaptive exploration** vs. monotonic schedules.  
- Anyone who wants a **single-file** RL training loop that is easy to modify.  

---

## How It Works

1. **Policy network** (`PolicyNetwork`): 2-layer MLP → logits over actions.  
2. **Rollout** (`forward_pass`):
   - Normalizes each observation using running mean/std **before** the policy.
   - Samples actions from `Categorical(softmax(logits / temperature))`.
   - Records `log_prob(actions)`, rewards, and per-step entropies.
3. **Returns** (`calculate_stepwise_returns`):
   - Computes discounted returns \( G_t \) via a backward pass.
   - **Normalizes** returns (zero mean, unit std) to reduce variance.
4. **Loss & Update** (`calculate_loss`, `update_policy`):
   - REINFORCE: \(-\sum G_t \cdot \log \pi(a_t|s_t)\).
   - Optional entropy bonus (encourages exploration early).
   - Backprop, **gradient clipping** (`max_norm=0.5`), Adam update.
5. **Adaptive knobs** (`adapt_knobs`):
   - Uses a **stateful EMA** of performance to compute smooth targets for:
     - **temperature** (explore ↔ exploit),
     - **entropy coefficient**,
     - **learning-rate scale**.
   - Adds **hysteresis** (a small “stable band” near the goal) and **rate limiting** (max delta/episode) to avoid oscillations.
6. **Progress signals**:
   - **Mean(25)** (sliding window of last 25 episodes) for human-readable logs.
   - **EMA** for control/consistency.

---

## Installation

```bash
# Python 3.9+ recommended
pip install torch gymnasium pygame numpy
# If classic control envs are missing:
# pip install "gymnasium[classic-control]"
