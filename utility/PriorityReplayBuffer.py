import random
import torch
import queue


class MPriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item):
        self.heap.append(item)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return root

    def _heapify_up(self, index):
        parent_index = (index - 1) // 2
        if index > 0 and self.heap[index] > self.heap[parent_index]:  # 自定义对象的比较关系
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            self._heapify_up(parent_index)

    def _heapify_down(self, index):
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        largest = index

        if (left_child_index < len(self.heap) and
                self.heap[left_child_index] > self.heap[largest]):
            largest = left_child_index

        if (right_child_index < len(self.heap) and
                self.heap[right_child_index] > self.heap[largest]):
            largest = right_child_index

        if largest != index:
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            self._heapify_down(largest)


class Transition(object):
    def __init__(self, priority=0, index=0):
        self.priority = priority
        self.index = index

    def __lt__(self, other):
        return self.priority < other.priority

    def __gt__(self, other):
        return self.priority > other.priority


class PriorityReplayBuffer:
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
        self.max_priority = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = MPriorityQueue()

    def __len__(self):
        return len(self.buffer)

    def transform(self, lazy_frame):
        state = torch.from_numpy(lazy_frame.__array__()[None] / 255).float()
        return state.to(self.device)

    def push(self, td_error, state, action, reward, next_state, done):
        """
        Adds an experience to the replay buffer.

        Args:
            state (numpy.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (numpy.ndarray): The next state.
            done (bool): Whether the episode is done.

        """
        trans = Transition(priority=td_error, index=self.cur)
        self.q.push(trans)

        if len(self.buffer) == self.size:
            self.buffer[self.cur] = (state, action, reward, next_state, done)
        else:
            self.buffer.append((state, action, reward, next_state, done))
        self.cur = (self.cur + 1) % self.size

    def _get_index(self, batch_size):
        weight = [1 / (i + 1) for i in range(len(self.q.heap))]
        return random.choices(self.q.heap, weight, k=batch_size)

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            tuple: A tuple containing the batch of states, actions, rewards, next states, and dones.

        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for index in self._get_index(batch_size):
            frame, action, reward, next_frame, done = self.buffer[index.index]
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
