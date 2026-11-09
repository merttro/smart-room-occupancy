import torch
import torch.nn as nn

class OccupancyCalc(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5,16)
        self.layer2 = nn.Linear(16,8)
        self.layer3 = nn.Linear(8,1)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)

        return x