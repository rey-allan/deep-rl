import argparse
import random
from collections import namedtuple
from copy import copy
from typing import List, Tuple

import gym
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from agents import ActorCriticAgent
from util import evaluate, plot_rewards, render_interaction

# For reproducibility
torch.manual_seed(24)
random.seed(24)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Trajectory = namedtuple("Trajectory", ["state", "action", "reward"])
Data = namedtuple("Data", ["values", "log_probs", "returns", "advantages"])


# pylint: disable=too-few-public-methods
class ActorLearner:
    """An actor-learner that collects experience to be sent to the centralized learning process

    :param env: The environment to collect experience from
    :type env: gym.Env
    :param agent: The learning agent
    :type agent: ActorCriticAgent
    :param num_episodes: The max number of episodes to collect
    :type num_episodes: int
    :param max_steps: The max number of steps per episode
    :type max_steps: int
    """

    def __init__(self, env: gym.Env, agent: ActorCriticAgent, num_episodes: int, max_steps: int) -> None:
        self._env = env
        self._agent = agent
        self._num_episodes = num_episodes
        self._max_steps = max_steps

    def collect(self) -> Tuple[List[List[Trajectory]], List[float]]:
        """Collects experience from the environment

        :return: A tuple with a list of episodes and a list of total reward per episode
        :rtype: Tuple[List[List[Trajectory]], List[float]]
        """
        episodes = []
        rewards = []
        s = torch.from_numpy(self._env.reset()).float()
        done = False

        # Collect experience
        for _ in range(self._num_episodes):
            trajectories = []
            steps = 0
            reward = 0.0

            while not done and steps < self._max_steps:
                a = self._agent.act(s)
                s_prime, r, done, _ = self._env.step(a)
                trajectories.append(Trajectory(s, a, r))
                s = torch.from_numpy(s_prime).float()
                steps += 1
                reward += r

            # If the episode got truncated then bootstrap to approximate the missing returns
            if not done:
                a = self._agent.act(s)
                v, _ = self._agent.step(s, a)
                trajectories.append(Trajectory(s, a, v.detach().item()))
            else:
                s = torch.from_numpy(self._env.reset()).float()
                done = False

            episodes.append(trajectories)
            rewards.append(reward)

        return episodes, rewards


# pylint: disable=too-many-locals
def a2c(
    agent: ActorCriticAgent, actor_learners: List[ActorLearner], epochs: int, alpha: float, gamma: float, verbose: bool
) -> List[float]:
    """Trains an agent using advantage actor-critic (A2C) algorithm

    :param agent: The learning agent
    :type agent: ActorCriticAgent
    :param actor_learners: A list of actor-learners to collect experience
    :type actor_learners: List[ActorLearner]
    :param epochs: The number of epochs to train the agent for
    :type epochs: int
    :param alpha: The learning rate
    :type alpha: float
    :param gamma: The discount factor
    :type gamma: float
    :param verbose: Whether to run in verbose mode or not
    :type verbose: bool
    :return: The total reward per episode (averaged over all actor-learners)
    :rtype: List[float]
    """
    pi_optimizer = optim.Adam(agent.pi.parameters(), lr=alpha)
    v_optimizer = optim.Adam(agent.v.parameters(), lr=alpha)
    total_rewards = []

    for _ in tqdm(range(epochs), disable=not verbose):
        # Execute each actor learner synchronously collecting their experience
        total_episodes = []
        rewards_per_learner = []
        for actor_learner in actor_learners:
            episodes, rewards = actor_learner.collect()
            total_episodes.extend(episodes)
            rewards_per_learner.append(rewards)

        # Process experience to generate the data for learning
        data = _process_episodes(total_episodes, agent, gamma)

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

        # Collect the average reward per episode across all learners
        total_rewards.extend(np.array(rewards_per_learner).mean(axis=0).tolist())

    return total_rewards


def _process_episodes(episodes: List[List[Trajectory]], agent: ActorCriticAgent, gamma: float) -> Data:
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
    parser = argparse.ArgumentParser(description="Execute Advantage Actor Critic against CartPole-v1 environment")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train")
    parser.add_argument("--learners", type=int, default=16, help="Number of actor-learners to use")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes to sample per learner")
    parser.add_argument("--max-steps", type=int, default=20, help="Max length of experience to use by the learners")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes to use for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    parser.add_argument("--save-gif", action="store_true", help="Save a GIF of an interaction after training")
    args = parser.parse_args()

    agent = ActorCriticAgent(num_features=4, num_actions=2, device=device)

    actor_learners = []
    for _ in range(args.learners):
        # Each learners gets its own independent copy of the environment
        env = gym.make("CartPole-v1")
        # For reproducibility (each copy with its own seed)
        env.seed(random.randint(0, 100))
        actor_learners.append(ActorLearner(env, agent, num_episodes=args.episodes, max_steps=args.max_steps))

    print(f"Training agent with the following args\n{args}")
    rewards = a2c(
        agent,
        actor_learners,
        epochs=args.epochs,
        alpha=args.alpha,
        gamma=args.gamma,
        verbose=args.verbose,
    )

    plot_rewards(rewards, title="A2C on CartPole-v1", output_dir="a2c", filename="CartPole-v1")

    # Make a copy of the environment for evaluation only
    env = gym.make("CartPole-v1")
    env.seed(24)

    print("Evaluating agent")
    evaluate(env, agent, args.eval_episodes, args.verbose)

    if args.save_gif:
        print("Rendering interaction")
        render_interaction(env, agent, output_dir="a2c", filename="CartPole-v1")
