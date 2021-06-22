import argparse
from collections import namedtuple
from copy import copy
from typing import List, Tuple

import gym
import torch
import torch.optim as optim
from tqdm import tqdm

from agents import Agent, DiscreteAgent
from util import evaluate, plot_rewards, render_interaction

# For reproducibility
torch.manual_seed(24)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Trajectory = namedtuple("Trajectory", ["state", "action", "reward"])
Data = namedtuple("Data", ["values", "log_probs", "returns", "advantages"])


# pylint: disable=too-many-locals
def vpg(
    env: gym.Env,
    agent: Agent,
    epochs: int,
    num_episodes: int,
    max_steps: int,
    gamma: float,
    alpha: float,
    verbose: bool,
) -> List[float]:
    """Trains an agent using vanilla policy gradient (a.k.a REINFORCE) algorithm

    :param env: The environment to train the agent in
    :type env: gym.Env
    :param agent: The agent to train
    :type agent: Agent
    :param epochs: The number of epochs to train the agent for
    :type epochs: int
    :param num_episodes: The number of episodes to sample per epoch
    :type num_episodes: int
    :param max_steps: The max number of steps per episode
    :type max_steps: int
    :param gamma: The discount factor
    :type gamma: float
    :param alpha: The learning rate
    :type alpha: float
    :param verbose: Whether to run in verbose mode or not
    :type verbose: bool
    :return: The total reward per episode
    :rtype: List[float]
    """
    pi_optimizer = optim.Adam(agent.pi.parameters(), lr=alpha)
    v_optimizer = optim.Adam(agent.v.parameters(), lr=alpha)
    total_rewards = []

    for _ in tqdm(range(epochs), disable=not verbose):
        # Collect and process experience from the environment
        episodes, rewards = _sample_episodes(env, agent, num_episodes, max_steps)
        data = _process_episodes(episodes, agent, gamma)
        total_rewards.extend(rewards)

        # Update the policy function
        pi_optimizer.zero_grad()
        # We use the negative loss because policy optimization is done using gradient _ascent_
        # This is because in policy gradient methods, the "loss" is a performance measure that is _maximized_
        pi_loss = 0.0
        for log_prob, advantage in zip(data.log_probs, data.advantages):
            pi_loss += -advantage * log_prob
        pi_loss = pi_loss.mean()
        pi_loss.backward()
        pi_optimizer.step()

        # Update the value function
        v_optimizer.zero_grad()
        v_loss = 0.0
        for value, ret in zip(data.values, data.returns):
            v_loss += (value - ret) ** 2
        v_loss = v_loss.mean()
        v_loss.backward()
        v_optimizer.step()

    return total_rewards


def _sample_episodes(
    env: gym.Env, agent: Agent, num_episodes: int, max_steps: int
) -> Tuple[List[List[Trajectory]], List[float]]:
    episodes = []
    rewards = []

    for _ in range(num_episodes):
        trajectories = []
        s = torch.from_numpy(env.reset()).float()
        done = False
        steps = 0
        reward = 0.0

        while not done and steps < max_steps:
            a = agent.act(s)
            s_prime, r, done, _ = env.step(a)
            trajectories.append(Trajectory(s, a, r))
            s = torch.from_numpy(s_prime).float()
            steps += 1
            reward += r

        # If the episode got truncated then bootstrap to approximate the missing returns
        if not done:
            a = agent.act(s)
            v, _ = agent.step(s, a)
            trajectories.append(Trajectory(s, a, v))

        episodes.append(trajectories)
        rewards.append(reward)

    return episodes, rewards


def _process_episodes(episodes: List[List[Trajectory]], agent: Agent, gamma: float) -> Data:
    values = []
    log_probs = []
    returns = []
    advantages = []

    for episode in episodes:
        trajectories = copy(episode)
        # Reverse the list so we start backpropagating the return from the last episode
        trajectories.reverse()

        # Compute tuples of [V(s_t), log pi(A_t|S_t), G_t, Adv(S_t,A_t)]
        g = 0
        for t in trajectories:
            # Compute the return G_t:T
            g = t.reward + gamma * g

            v, log_prob = agent.step(t.state, t.action)

            # Baseline (i.e. advantage function) using the value function approximator
            advantage = g - v.detach()

            values.append(v)
            log_probs.append(log_prob)
            returns.append(g)
            advantages.append(advantage)

    return Data(values, log_probs, returns, advantages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Vanilla Policy Gradient against CartPole-v1 environment")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes to sample per epoch")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes to use for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    parser.add_argument("--save-gif", action="store_true", help="Save a GIF of an interaction after training")
    args = parser.parse_args()

    agent = DiscreteAgent(num_features=4, num_actions=2, device=device)
    env = gym.make("CartPole-v1")
    # For reproducibility
    env.seed(24)

    print(f"Training agent with the following args\n{args}")
    rewards = vpg(
        env,
        agent,
        epochs=args.epochs,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        alpha=args.alpha,
        verbose=args.verbose,
    )

    plot_rewards(rewards, ma_window=100, title="VPG on CartPole-v1", output_dir="vpg", filename="CartPole-v1")

    print("Evaluating agent")
    evaluate(env, agent, args.eval_episodes, args.verbose)

    if args.save_gif:
        print("Rendering interaction")
        render_interaction(env, agent, output_dir="vpg", filename="CartPole-v1")
