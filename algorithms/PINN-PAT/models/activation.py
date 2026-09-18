import torch
from torch import nn
from enum import Enum
from typing import Type
class Activation(Enum):
    TANH = "Tanh"
    SINUSOIDAL = "Sin"
    @classmethod
    def func(cls, activation: "Activation") -> Type[nn.Module]:
        if activation == cls.TANH:
            return nn.Tanh
        elif activation == cls.SINUSOIDAL:
            return Sinusoidal
        else:
            assert False

class Sinusoidal(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.sin(x)
