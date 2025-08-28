import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import pygame  # for event pump (window responsiveness)

pygame.init()

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

# ---- Headless env for training (fast)
SEED = 1111
train_env = gym.make("CartPole-v1")

np.random.seed(SEED)
torch.manual_seed(SEED)

def calculate_loss(returns, log_prob_actions, entropy_terms=None, entropy_coef=0.0):
    base = -(returns * log_prob_actions).sum()
    if entropy_terms is not None and entropy_coef > 0.0:
        base = base - entropy_coef * entropy_terms.sum()
    return base

def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    #holds discounted returns for each step
    #R is accumulator for discounted sum of future rewards
    R = 0.0
    #More efficient to loop backwards
    for r in reversed(rewards):
        #Every return only needs the next return R_t+1 which is already computed
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    #Normalize to stabalize training
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

count  = 1e-4

obs_mean = np.zeros(train_env.observation_space.shape[0])
obs_var  = np.ones(train_env.observation_space.shape[0])
count = 1e-4  # to avoid div by zero

def update_running_stats(obs):
    global obs_mean, obs_var, count
    count += 1
    old_mean = obs_mean.copy()
    obs_mean += (obs - obs_mean) / count
    obs_var += (obs - old_mean) * (obs - obs_mean)

def normalize_obs(obs):
    obs_std = np.sqrt(obs_var / count)
    return (obs - obs_mean) / (obs_std + 1e-8)
#obs: Observation: Current state of the environment after taking an action
#Array of [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
#r: Reward: Numerical reward given by environment after taking the action
#+1 for every timestep that the pole remains upright and the episode hasnt ended
#terminated: Boolean flag to check if final condition has been reached
#truncated: Boolean flag to check if the episode ended due to lack of time
def forward_pass(env, policy, discount_factor, temperature=1.0):
    log_prob_actions, rewards, entropies = [], [], []
    done = False
    episode_return = 0.0
    policy.train()
    obs, info = env.reset()

    # initialize running stats with the first obs
    update_running_stats(obs)
    obs = normalize_obs(obs)

    while not done:
        # ALWAYS use normalized obs to act
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(obs_t)
        probs = F.softmax(logits / max(1e-6, temperature), dim=-1)
        dist = distributions.Categorical(probs)

        action = dist.sample()
        log_prob_actions.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        next_obs, r, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        rewards.append(r)
        episode_return += r

        # update stats with raw next_obs, then normalize FOR THE NEXT STEP
        update_running_stats(next_obs)
        obs = normalize_obs(next_obs)

    log_prob_actions = torch.cat(log_prob_actions)
    entropy_terms = torch.cat(entropies) if entropies else torch.tensor(0.0)
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)
    return episode_return, stepwise_returns, log_prob_actions, entropy_terms

#log_prob_actions: The log probabilities of the actins your policy actually took during forward_pass
#stepwise_returns: Discounted returns you just calculated for each timestep
def update_policy(stepwise_returns, log_prob_actions, optimizer, entropy_terms=None, entropy_coef=0.0):
    loss = calculate_loss(stepwise_returns.detach(), log_prob_actions, entropy_terms, entropy_coef)
    #Compute loss
    optimizer.zero_grad()
    #Clears out any old gradients from previous updates
    #Without this, PyTorch accumulates gradients

    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=0.5)
    #Computes gradients of the loss with respect to policys parameters
    #Policy gradient is calculated
    optimizer.step()
    #Applies computed gradient using the optimizer. Updates neural network parameters
    return loss.item()
#Returns the numeric value of the loss 




# ---- Render one episode with current policy (real-time)
def watch_episode(policy, seed=SEED):
    view_env = gym.make("CartPole-v1", render_mode="human")
    obs, info = view_env.reset(seed=SEED)
    policy.eval()  # disable dropout during viewing
    clock = pygame.time.Clock()
    done = False
    ep_ret = 0.0
    while not done:
        # pump pygame events so the window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                view_env.close()
                #pygame.quit()
                return

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = policy(obs_t)
            probs = F.softmax(logits, dim=-1)
            #print(f'| Probs: {probs} |')

            action = torch.argmax(probs, dim=-1)  # greedy for viewing
        obs, r, terminated, truncated, _ = view_env.step(action.item())
        done = terminated or truncated
        ep_ret += r

        # human mode renders on step; cap FPS for smoothness
        clock.tick(60)

    view_env.close()
    #pygame.event.pump()  # flush events after close
    #print(f"[Watch] Episode return: {ep_ret:.1f}")

def main():
    MAX_EPOCHS = 500
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    WATCH_EVERY = 10

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n
    BASE_LR = 1e-3

    policy = PolicyNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = optim.Adam(policy.parameters(), lr=BASE_LR)

    # ---------- stateful, smoothed adaptation ----------
    def adapt_knobs(score, state):
        """
        score: scalar performance signal (we'll pass EMA)
        state: dict with previous knob values to ensure smooth, consistent evolution
        Returns (temperature, entropy_coef, lr_scale)
        """
        # handle first-episode case
        if score is None or np.isnan(score):
            t_target, e_target, lr_target = 2.0, 0.03, 1.5
        else:
            gap = max(REWARD_THRESHOLD - float(score), 0.0)
            norm = min(gap / REWARD_THRESHOLD, 1.0)
            if gap == 0:
                # at/above threshold → stable mode
                t_target, e_target, lr_target = 0.6, 0.0, 0.6
            else:
                # below threshold → volatile mode (scaled by gap)
                t_target  = 1.0 + 1.5 * norm     # 1.0..2.5
                e_target  = 0.01 + 0.04 * norm   # 0.01..0.05
                lr_target = 1.0 + 2.0 * norm     # 1.0..3.0

            # hysteresis near threshold to avoid flip-flopping
            STABLE_BAND = 15.0
            if gap <= STABLE_BAND:
                t_target = min(t_target, 0.8)

        # smooth + rate limit toward targets
        ALPHA = 0.2
        MAX_DT, MAX_DE, MAX_DLR = 0.15, 0.01, 0.40

        def smooth(prev, target, max_delta):
            proposed = prev + ALPHA * (target - prev)   # EMA step
            delta = np.clip(proposed - prev, -max_delta, max_delta)  # rate limit
            return float(prev + delta)

        t_new  = smooth(state["temperature"],  t_target,  MAX_DT)
        e_new  = smooth(state["entropy_coef"], e_target,  MAX_DE)
        lr_new = smooth(state["lr_scale"],     lr_target, MAX_DLR)

        state["temperature"]  = t_new
        state["entropy_coef"] = e_new
        state["lr_scale"]     = lr_new
        return t_new, e_new, lr_new

    # knob state + performance EMA
    knob_state = {"temperature": 2.0, "entropy_coef": 0.03, "lr_scale": 1.5}
    EMA_BETA = 0.95
    perf_ema = None

    episode_returns = []
    for episode in range(1, MAX_EPOCHS + 1):
        # compute recent performance (window) for display
        mean_recent = np.mean(episode_returns[-N_TRIALS:]) if episode_returns else float('nan')

        # update EMA for stable control signal
        if episode_returns:
            last_ret = episode_returns[-1]
            perf_ema = last_ret if perf_ema is None else (EMA_BETA * perf_ema + (1 - EMA_BETA) * last_ret)

        # adapt knobs from EMA (more consistent than windowed mean)
        temperature, entropy_coef, lr_scale = adapt_knobs(perf_ema, knob_state)

        # set adaptive learning rate
        for pg in optimizer.param_groups:
            pg['lr'] = BASE_LR * lr_scale

        # training rollout with adaptive temperature
        ep_ret, step_returns, logps, ent_terms = forward_pass(
            train_env, policy, DISCOUNT_FACTOR, temperature=temperature
        )
        _ = update_policy(step_returns, logps, optimizer,
                          entropy_terms=ent_terms, entropy_coef=entropy_coef)

        episode_returns.append(ep_ret)

        # recompute display stats
        mean_recent = np.mean(episode_returns[-N_TRIALS:])
        if perf_ema is None:
            perf_ema = ep_ret  # seed EMA after first episode

        if episode % PRINT_INTERVAL == 0:
            print(
                f'| Episode: {episode:3} | Mean(25): {mean_recent:5.1f} | EMA: {perf_ema:5.1f} | '
                f'T={temperature:.2f} | Ent={entropy_coef:.3f} | LR={optimizer.param_groups[0]["lr"]:.2e}'
            )

        # # optional: render periodically
        #if episode % WATCH_EVERY == 0:
           #  watch_episode(policy, seed=SEED)

        # success condition (use EMA for consistency)
        if mean_recent >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            watch_episode(policy, seed=123)  # final victory lap
            break

    train_env.close()

if __name__ == "__main__":
    main()
