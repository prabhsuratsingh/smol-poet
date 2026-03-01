import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        self.dim = dim

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)

        freqs = torch.einsum("i,j->ij", t, inv_freq)  

        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, positions):
        cos = self.cos[positions].to(device=x.device, dtype=x.dtype)
        sin = self.sin[positions].to(device=x.device, dtype=x.dtype)

        cos = cos.unsqueeze(0).unsqueeze(0)  
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)
        x_rot = x_rot.flatten(-2)

        return x_rot
