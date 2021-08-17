from matplotlib import artist
import torch
import gym
from torch import random
from Agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import argparse

def train(args):
    env = gym.make('CartPole-v0')
    env._max_episode_steps = args.max_steps
    agent = Agent(4, 2)
    
    rewards = []
    avg_rewards = []

    for episode in range(args.episodes):
        state = env.reset()

        acc_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            
            next_state, reward, done, _ = env.step(action)

            reward += abs(next_state[0]) * -1

            agent.store(state, action, next_state, reward, done)
            
            agent.learn()
            

            acc_reward += reward
            state = next_state

        rewards.append(acc_reward)
        avg_rewards.append(np.mean(rewards[-100:]))
        
        if not args.quiet:
            print(f' - episode {episode + 1}, reward: {acc_reward}, avg reward: {np.mean(rewards[-100:])}, eps: {agent.epsilon}')
    
    if args.plot:
        plt.plot(avg_rewards)
        plt.show()
    
    if args.save is not None:
        agent.save(args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-s", "--save", type=str)
    parser.add_argument("-m", "--max-steps", type=int, default=200)
    parser.add_argument("-e", "--episodes", type=int, required=True)

    args = parser.parse_args()

    train(args)