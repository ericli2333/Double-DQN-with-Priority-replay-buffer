import argparse
from utility.EnvConfig import make_env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import DQN.DQNAgent as AG
import math
import torch
from datetime import datetime

batch_size = 32
learning_rate = 3e-4
epsilon_begin = 1.0
epsilon_end = 0.2
epsilon_decay = 200000
epsilon_min = 0.001
alpha = 0.95
replay_start_size = 10000
update = 1000
print_interval = 1000


def epsilon(cur):
    return epsilon_end + (epsilon_begin - epsilon_end) * math.exp(-1.0 * cur / epsilon_decay)


def train(env_name='PongNoFrameskip-v4', learning_rate=3e-4, gamma=0.99, memory_size=100000, total_frame=2000000):
    env = make_env(env_name)
    DQNAgent = AG.Agent(in_channels=env.observation_space.shape[0], num_actions=env.action_space.n,
                        reset_network_interval=update,
                        lr=learning_rate, gamma=gamma, epsilon=epsilon_min, replay_size=memory_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = env.reset()[0]
    total_reward = 0
    Loss = []
    Reward = []
    episodes = 0
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y %m %d %H %M %S")
    writer = SummaryWriter(
        log_dir=f'./logs/{formatted_time}-env{env_name}-lr{learning_rate}-gamma{gamma}-memory_size{memory_size}-total_frame{total_frame}-max_eps{epsilon_begin}-min_eps{epsilon_end}-eps_decay{epsilon_decay}')

    for frame_num in range(total_frame):
        eps = epsilon(frame_num)
        state = DQNAgent.replay_buffer.transform(frame)
        action = DQNAgent.get_action(state, epsilon=eps)
        next_frame, reward, terminated, trunacated, _ = env.step(action)
        DQNAgent.replay_buffer.push(frame, action, reward, next_frame, terminated)
        total_reward += reward
        frame = next_frame
        loss = 0

        if len(DQNAgent.replay_buffer) > replay_start_size:
            loss = DQNAgent.train(batch_size=batch_size)
            Loss.append(loss)

        if frame_num % DQNAgent.reset_network_interval == 0:
            DQNAgent.reset()

        if frame_num % print_interval == 0:
            cur_reward = -22
            if len(Reward) > 0:
                cur_reward = np.mean(Reward[-10:])
            print('frame : {}, loss : {:.8f}, reward : {}'.format(frame_num, loss, cur_reward))
            writer.add_scalar('loss', loss, frame_num)
            writer.add_scalar('reward', cur_reward, frame_num)
            writer.add_scalar('epsilon', eps, frame_num)
            if len(DQNAgent.replay_buffer) > replay_start_size:
                for name, param in DQNAgent.q_network.named_parameters():
                    writer.add_histogram(tag=name + '_grad', values=param.grad, global_step=frame_num // 1000)
                    writer.add_histogram(tag=name + '_data', values=param.data, global_step=frame_num // 1000)

        if terminated:
            episodes += 1
            Reward.append(total_reward)
            print('episode {}: total reward {}'.format(episodes, total_reward))
            frame = env.reset()[0]
            total_reward = 0

        if frame_num % 1000 == 0:
            torch.cuda.empty_cache()

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='PongNoFrameskip-v4', help="Name of the environment")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--memory_size", type=int, default=100000, help="Size of the replay buffer")
    parser.add_argument("--total_frame", type=int, default=5000000, help="Total number of frames to train")
    parser.add_argument("--eps-max", type=float, default=1, help="Max epsilon value")
    parser.add_argument("--eps-min", type=float, default=0.02, help="Min epsilon value")
    args = parser.parse_args()
    epsilon_begin = args.eps_max
    epsilon_end = args.eps_min
    train(env_name=args.env_name,
          learning_rate=args.lr,
          gamma=args.gamma,
          memory_size=args.memory_size,
          total_frame=args.total_frame)
