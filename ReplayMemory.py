import random

class ReplayBuffer:
    def __init__(self, mem_size) -> None:
        
        self.replay_buffer = []
        self.mem_size = mem_size
        self.mem_pointer = 0

    def store(self, state, action, next_state, reward, done):
        if len(self.replay_buffer) < self.mem_size:
            self.replay_buffer.append(None)

        self.replay_buffer[self.mem_pointer] = (state, action, next_state, reward, done)

        self.mem_pointer = (self.mem_pointer + 1) % self.mem_size
    
    def sample(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)

        states, actions, next_states, rewards, dones = [], [], [], [], []

        for sample in batch:
            state, action, next_state, reward, done = sample
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        
        return states, actions, next_states, rewards, dones