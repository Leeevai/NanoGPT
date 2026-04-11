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
    block_size: int = 1024 #context length
    vocab_size: int = 50257 # 50k BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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
        
    def forward(self, idx, targets=None):
        #idx is (B, T) tensor of token indices
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        #forward the token and position embeddings
        pos = torch.arange(0,T, dtype=torch.long, device=idx.device) # (1, T)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (1, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        loss = None
        if targets is not None:
            #reshape x and targets to be (B*T, vocab_size) and (B*T) respectively so that we can compute the cross-entropy loss
            loss = F.cross_entropy(self.lm_head(x).view(-1, self.config.vocab_size), targets.view(-1))
        return self.lm_head(x), loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads a pre-trained GPT model from the Hugging Face Transformers library."""
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], f"Invalid model type: {model_type}. Must be one of ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']"
        from transformers import GPT2LMHeadModel
        print(f"Loading pre-trained GPT model: {model_type}")
        
        #n_layes, n_head, n_embd are determined by the model type (e.g. gpt2, gpt2-medium, gpt2-large, gpt2-xl)
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 355M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1.5B parameters
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # remove the bias buffer from the state dict keys since it is not a parameter and will not be loaded from the pre-trained model
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # remove the bias buffer from the state dict keys since it is not a parameter and will not be loaded from the pre-trained model
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # remove the bias buffer from the state dict keys since it is not a parameter and will not be loaded from the pre-trained model
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai implementation of GPT-2 has the weights of the linear layers transposed compared to the Hugging Face implementation, so we need to transpose the weights when loading them from the pre-trained model
        #this means that we have to transpose the weights of the linear layers when loading them from the pre-trained model, but we do not have to transpose the weights of the embedding layers since they are not transposed in either implementation
        
        assert len(sd_keys) == len(sd_keys_hf), f"Number of keys in state dict does not match number of keys in pre-trained model: {len(sd_keys)} vs {len(sd_keys_hf)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                #special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for key {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                #vanilla copy for the rest of the weights
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for key {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        print(f"Loaded pre-trained GPT model: {model_type}")
        return model

num_return_sequences = 5
max_length = 30     


# ------------------------------------------------------------------------------------

import tiktoken

class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T

        #at init laod tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from input.txt")
        print(f"1 epoch is {len(self.tokens) // (B*T)} iterations")
        
        #state 
        self.current_position = 0
        
    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_position += B*T
        if self.current_position + B*T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y



device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")


# ------------------------------------------------------------------------------------
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.to(device)
# logits, loss = model(x, targets=y)


# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for _ in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, targets=y)
    loss.backward()
    optimizer.step()
    print(f"step {_+1}, loss: {loss.item():.4f}")


# print(loss) # (B, T, vocab_size)
import sys; sys.exit(0)

torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (num_return_sequences, sequence_length, vocab_size)
        logits = logits[:, -1, :] # (num_return_sequences, vocab_size)
        probs = F.softmax(logits, dim=-1) # (num_return_sequences, vocab_size)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # (num_return_sequences, k)
        ix = torch.multinomial(topk_probs, num_samples=1) # (num_return_sequences, 1)
        xcol = torch.gather(topk_indices, dim=-1, index=ix) # (num_return_sequences, 1)
        x = torch.cat((x, xcol), dim=1) # (num_return_sequences, sequence_length + 1)

for i in range(num_return_sequences):
    token = x[i,:max_length].tolist()
    decoded = enc.decode(token)
    print(f"Generated text {i+1}: {decoded}")
