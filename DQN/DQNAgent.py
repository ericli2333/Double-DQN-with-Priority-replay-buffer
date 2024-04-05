from utility.NetWork import NetWork
from utility.ReplayBuffer import ReplayBuffer
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class Agent:
    """
    The Agent class represents a Deep Q-Network (DQN) agent for reinforcement learning.

    Args:
        in_channels (int): Number of input channels.
        num_actions (int): Number of possible actions.
        lr (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for epsilon-greedy action selection.
        replay_size (int): Size of the replay buffer.

    Attributes:
        num_actions (int): Number of possible actions.
        device (torch.device): Device (CPU or GPU) for running computations.
        gamma (float): Discount factor for future rewards.
        q_network (DQN): Q-network for estimating action values.
        target_network (DQN): Target network for estimating target action values.
        optimizer (torch.optim.RMSprop): Optimizer for updating the Q-network.

    Methods:
        greedy(state, epsilon): Selects an action using epsilon-greedy policy.
        calculate_loss(states, actions, rewards, next_states, dones): Calculates the loss for a batch of experiences.
        reset(): Resets the target network to match the Q-network.
        learn(batch_size): Performs a single learning step using a batch of experiences.
    """

    def __init__(self, in_channels, num_actions, reset_network_interval, lr, gamma, epsilon, replay_size):
        self.num_actions = num_actions
        self.replay_buffer = ReplayBuffer(replay_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.reset_network_interval = reset_network_interval
        self.gamma = gamma
        self.q_network = NetWork(in_channels, num_actions).to(self.device)
        self.target_network = NetWork(in_channels, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=lr, eps=epsilon, alpha=alpha)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, eps=epsilon)

    def get_action(self, state, epsilon):
        """
        Selects an action using epsilon-greedy policy.

        Args:
            state (torch.Tensor): Current state.
            epsilon (float): Exploration rate.

        Returns:
            int: Selected action.
        """
        if random.random() < epsilon:
            action = random.randrange(self.num_actions)
        else:
            q_values = self.q_network(state).detach().cpu().numpy()
            action = np.argmax(q_values)
            del q_values
        return action

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Calculates the loss for a batch of experiences.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.
            rewards (torch.Tensor): Batch of rewards.
            next_states (torch.Tensor): Batch of next states.
            dones (torch.Tensor): Batch of done flags.

        Returns:
            torch.Tensor: Loss value.
        """
        tmp = self.q_network(states)
        rewards = rewards.to(self.device)
        q_values = tmp[range(states.shape[0]), actions.long()]
        q_predict_action = self.q_network(next_states).detach().argmax(1)
        target_output = self.gamma * self.target_network(next_states)
        target_output = target_output[::, q_predict_action]
        default = rewards + target_output
        target = torch.where(dones.to(self.device), rewards, default).to(self.device).detach()
        return F.mse_loss(target, q_values)

    def reset(self):
        """
        Resets the target network to match the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, batch_size):
        """
        Performs a single learning step using a batch of experiences.

        Args:
            batch_size (int): Size of the batch.

        Returns:
            float: Loss value.
        """
        if batch_size < len(self.replay_buffer):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            loss = self.calculate_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()
            return loss.item()
        return 0
