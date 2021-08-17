import gym
from Agent import Agent
import argparse


def test(args):
    agent = Agent(4, 2, epsilon=0)
    agent.load(args.file)

    env = gym.make('CartPole-v0')
    env._max_episode_steps = args.max_steps

    acc_reward = 0
    
    for episode in range(args.episodes):
        state = env.reset()
        done =False
        n_steps =0
        while not done:
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            n_steps += 1
            acc_reward += reward
        if args.verbose:
            print(f' - episode {episode + 1} ended in {n_steps} steps.')

    print(f'Test ended with average reward {acc_reward / args.episodes}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-m", "--max-steps", type=int, default=500)
    parser.add_argument("-e", "--episodes", type=int, default=1)
    parser.add_argument("-f", "--file", type=str, required=True)

    args = parser.parse_args()

    test(args)