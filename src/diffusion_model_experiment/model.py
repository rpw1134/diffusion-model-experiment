from torch import nn

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.output_head = nn.Linear(128, 2)

    def forward(self, x):
        x = self.model(x)
        x = self.output_head(x)
        return x
