import torch
from ReplayMemory import ReplayBuffer
from Model import DQN
import numpy as np


class Agent():
    def __init__(self,
                 input_dim,
                 output_dim,
                 epsilon = 1,
                 min_epsilon = 0.01,
                 epsilon_decay_rate = 8e-5,
                 gamma = 0.99,
                 replay_memory_size = 100000,
                 batch_size = 256,
                 learning_rate = 0.0005,
                 replace_target_network = 10,
                ) -> None:

        self.replay_buffer = ReplayBuffer(replay_memory_size)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.replace_target_network = replace_target_network
        self.action_space = np.arange(output_dim)

        self.n_learn = 0

        self.online_network = DQN(input_dim, output_dim)
        self.target_network = DQN(input_dim, output_dim)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate, weight_decay=0.005)
        self.criterion = torch.nn.MSELoss()


    def get_action(self, state):
        if np.random.rand() > self.epsilon:
            state = torch.tensor([state], dtype=torch.float)
            q_vals = self.online_network(state)
            return torch.argmax(q_vals, dim=1).item()
        else:
            return np.random.choice(self.action_space)

    def dec_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.min_epsilon)

    def store(self, state, action, next_state, reward, done):
        self.replay_buffer.store(state, action, next_state, reward, done)

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        if len(self.replay_buffer.replay_buffer) < self.batch_size:
            return
            
        self.optimizer.zero_grad()

        if self.n_learn % self.replace_target_network == 0:
            self.update_target_network()
        
        self.dec_epsilon()

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = batch
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)

        indecies = np.arange(self.batch_size)

        q_pred = self.online_network(states)
        q_pred = q_pred[indecies, actions]

        q_target = self.target_network(next_states)
        q_target = torch.max(q_target, dim = 1).values
        q_target[dones] = 0

        q_eval = rewards + self.gamma * q_target

        loss = self.criterion(q_eval, q_pred)
        loss.backward()
        self.optimizer.step()

        self.n_learn += 1

        return loss.item()

    def save(self, file):
        torch.save(self.online_network.state_dict(), file)

    def load(self, file):
        self.online_network.load_state_dict(torch.load(file))




