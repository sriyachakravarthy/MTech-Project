import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.nn.parameter import Parameter
import numpy as np
import os

class UniDeepFsmn(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)

        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim, bias=False)

    def forward(self, input):

        f1 = F.relu(self.linear(input))

        p1 = self.project(f1)

        x = th.unsqueeze(p1, 1)

        x_per = x.permute(0, 3, 2, 1)

        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])

        out = x_per + self.conv1(y)

        out1 = out.permute(0, 3, 2, 1)

        return input + out1.squeeze()
