import torch

class AutoEncoder(torch.nn.Module):

    def __init__(self):
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * 3, 512),
            torch.relu(),
            torch.nn.Linear(512, 128),
            torch.relu(),
            torch.nn.Linear(128, 32),
            torch.relu(),
            torch.nn.Linear(32, 8)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.relu(),
            torch.nn.Linear(32, 128),
            torch.relu(),
            torch.nn.Linear(128, 512),
            torch.relu(),
            torch.nn.Linear(512, 32 * 32 * 3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

