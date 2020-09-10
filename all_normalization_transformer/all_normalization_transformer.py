import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads = 8):
        super().__init__()
    def forward(self, x):
        return x
