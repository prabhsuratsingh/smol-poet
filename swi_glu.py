import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.w1 = nn.Linear(embed_size, hidden_size, bias=False)
        self.w2 = nn.Linear(embed_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, embed_size, bias=False)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))
