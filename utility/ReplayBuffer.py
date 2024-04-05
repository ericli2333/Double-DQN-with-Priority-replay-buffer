import random
import torch


class ReplayBuffer:
    """
    A replay buffer class for storing and sampling experiences for reinforcement learning.

    Args:
        size (int): The maximum size of the replay buffer.

    Attributes:
        size (int): The maximum size of the replay buffer.
        buffer (list): A list to store the experiences.
        cur (int): The current index in the buffer.
        device (torch.device): The device to use for tensor operations.

    Methods:
        __len__(): Returns the number of experiences in the buffer.
        transform(lazy_frame): Transforms a lazy frame into a tensor.
        push(state, action, reward, next_state, done): Adds an experience to the buffer.
        sample(batch_size): Samples a batch of experiences from the buffer.

    """

    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.cur = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.buffer)

    def transform(self, lazy_frame):
        state = torch.from_numpy(lazy_frame.__array__()[None] / 255).float()
        return state.to(self.device)

    def push(self, state, action, reward, next_state, done):
        """
        Adds an experience to the replay buffer.

        Args:
            state (numpy.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (numpy.ndarray): The next state.
            done (bool): Whether the episode is done.

        """
        if len(self.buffer) == self.size:
            self.buffer[self.cur] = (state, action, reward, next_state, done)
        else:
            self.buffer.append((state, action, reward, next_state, done))
        self.cur = (self.cur + 1) % self.size

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            tuple: A tuple containing the batch of states, actions, rewards, next states, and dones.

        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(batch_size):
            frame, action, reward, next_frame, done = self.buffer[random.randint(0, len(self.buffer) - 1)]
            state = self.transform(frame)
            next_state = self.transform(next_frame)
            state = torch.squeeze(state, 0)
            next_state = torch.squeeze(next_state, 0)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (torch.stack(states).to(self.device), torch.tensor(actions).to(self.device),
                torch.tensor(rewards).to(self.device),
                torch.stack(next_states).to(self.device), torch.tensor(dones).to(self.device))
