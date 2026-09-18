import torch
from torch import nn
from torch.nn import init
from models.resblock import ResBlock
from models.activation import Activation

class MLP(nn.Module):
    def __init__(self, n_dim:int, n_hidden:int, l_layers:int, activation:Activation):
        super(MLP, self).__init__()

        self.input_layer = nn.Linear(n_dim, n_hidden)
        self.res_blocks = nn.Sequential(*[
            ResBlock(n_hidden, activation) for _ in range(l_layers)
        ])
        self.output_layer = nn.Linear(n_hidden, 1)
        # self.initialize()
    
    def initialize(self):
        
        # for module in self.modules():
        #     if isinstance(module, nn.Linear):
        #         # init.xavier_normal(module.weight)
        #         # init.xavier_normal_(module.weight, gain=0.1)
        #         init.kaiming_uniform_(module.weight, a=torch.sqrt(5))
        #         # init.zeros_(module.bias)
        #         init.kaiming_uniform_(module.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data *= 0.1
                module.bias.data *= 0.1

    def forward(self, inputs):
        # h = torch.stack(list(args), dim=-1)
        h = self.input_layer(inputs)
        h = self.res_blocks(h)
        return self.output_layer(h).squeeze()
