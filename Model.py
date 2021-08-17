import torch


class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.FC = torch.nn.Sequential(torch.nn.Linear(input_dim, 24),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(24, 48),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(48, 96),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(96, 48),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(48, 24),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(24, output_dim)
                                      )
        
    
    def forward(self, X):
        return self.FC(X)
