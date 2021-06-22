from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn


class Agent(ABC):
    """A class that defines the interface for Deep RL agents"""

    @property
    @abstractmethod
    def pi(self) -> nn.Module:
        """The policy function approximator

        :raises NotImplementedError: Property must be implemented by concrete agent classes
        :return: The policy approximator as a PyTorch module
        :rtype: nn.Module
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def v(self) -> nn.Module:
        """The value function approximator

        :raises NotImplementedError: Property must be implemented by concrete agebnt classes
        :return: The value approximator as a PyTorch module
        :rtype: nn.Module
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, s: torch.Tensor) -> Union[int, float]:
        """Selects an action for the given state

        :param s: The state to select an action for
        :type s: torch.Tensor
        :raises NotImplementedError: Method must be implemented by concrete agent classes
        :return: An action (either discrete or continuous)
        :rtype: Union[int, float]
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, s: torch.Tensor, a: Union[int, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs one single step

        Computes the value of the given state and the log probability, pi(a|s), of the given action

        :param s: The state to compute value for
        :type s: torch.Tensor
        :param a: The action to compute log probability for
        :type a: Union[int, float]
        :raises NotImplementedError: Method must be implemented by concrete learner classes
        :return: A tuple of V(s) and pi(a|s) (batch_size, action_size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError


class DiscreteAgent(Agent):
    """An agent that acts on discrete action spaces"""

    def __init__(self, num_features: int, num_actions: int, device: torch.device) -> None:
        # Architecture suggested in the paper "Benchmarking Deep Reinforcement Learning for Continuous Control"
        # https://arxiv.org/pdf/1604.06778.pdf
        self._pi = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(in_features=num_features, out_features=400),
            nn.ReLU(inplace=True),
            # Hidden Layer 2
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Linear(in_features=300, out_features=num_actions),
            nn.Softmax(dim=-1),
        ).to(device)

        self._v = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(in_features=num_features, out_features=400),
            nn.ReLU(inplace=True),
            # Hidden Layer 2
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Linear(in_features=300, out_features=1),
        ).to(device)

        self._device = device

    @property
    def pi(self) -> nn.Module:
        return self._pi

    @property
    def v(self) -> nn.Module:
        return self._v

    def act(self, s: torch.Tensor) -> Union[int, float]:
        return torch.distributions.Categorical(self._pi(s.to(self._device))).sample().item()

    def step(self, s: torch.Tensor, a: Union[int, float]) -> Tuple[torch.Tensor, float]:
        s = s.to(self._device)
        probs = self._pi(s)
        distribution = torch.distributions.Categorical(probs)
        values = self._v(s)

        # pylint: disable=not-callable
        return values, distribution.log_prob(torch.as_tensor(a).to(self._device))
