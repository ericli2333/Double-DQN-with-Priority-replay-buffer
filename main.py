import utility.envconfig as env

if __name__ == "__main__":
    Env = env.DQNenv('PongNoFrameskip-v4')
    Env.testEnv()