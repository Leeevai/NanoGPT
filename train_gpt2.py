from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() 
        # batch size, sequence length, embedding dimension(n_embd)
        #calculate query, key, value for all heads in batch and move head forward to be the batch dimension
        #nh is "number of heads" and head size is the dimension of each head (n_embd // n_head)
        #hs is the head size, and we reshape q, k, v to have shape (B, nh, T, hs) where nh is the number of heads and hs is the head size   
        #C is the total embedding dimension, which is split into n_head heads, so each head has dimension C // n_head
        #e.g in GPT-2 (124M) n_embd is 768 and n_head is 12, so each head has dimension 64, nh*hs = n_embd, and we have 12 heads each with dimension 64 to make up the total embedding dimension of 768
        qkv = self.c_attn(x) # (B, T, 3*C)
        q,k,v = qkv.split(self.n_embd, dim=2) # tuple of 3 tensors each of shape (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y    
    
    #commit message: implement the multi-head self attention module for GPT-2, which includes the linear projections for query, key, value, the attention calculation, and the output projection. The attention calculation includes scaling the dot product of query and key by the square root of the head size, applying a causal mask to prevent attending to future tokens, and applying softmax to get the attention weights. Finally, we compute the output by multiplying the attention weights with the value and applying the output projection.
    
    
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size: int = 256 #context length
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        