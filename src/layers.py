from dataclasses import dataclass
import torch.nn as nn
import torch
from torch import Tensor
@dataclass

class PocketConfig:
	vocab_size = 50257 # tiktoken
	seq_len = 256 # context window
	d_model = 384 # embedding dimension
	n_layers = 6 # no. stacked decoder blocks
	n_heads = 6 # Query heads
	n_kv_heads = 2 # kv heads
	dropout = 0.0 # regularization

	@property
	def d_k(self) ->  int:
		# dimension per attention
		return  self.d_model  // self.n_heads  # 64
	
	@property
	def n_groups(self) ->  int:
		# 3 query heads share each of 2 kv heads
		return  self.n_heads  //  self.n_kv_heads # 3

	@property 
	def ffn_hidden(self) -> int:
		# ffn hidden dimension, 4x d_model rounded to chunks
		multiple = 64
		return multiple * ((4 * self.d_model + multiple - 1) // multiple)

class RMSNorm(nn.Module):
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, tx: Tensor) -> Tensor:
        root_mean_square = tx.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps

        return tx / root_mean_square
	

class RoPE(nn.Module):
    def __init__(self, d_k, seq_len):
        super().__init__()
        theta = 1.0 / torch.pow(10000, torch.arange(0, d_k, 2).float() / d_k)
        positions = torch.arange(seq_len).float()
        
        angles = torch.outer(positions, theta)
        angles = torch.cat([angles, angles], dim=-1)

        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())

    def rotate_half(self, tx) -> Tensor:
        half = tx.shape[-1] // 2
        x1 = tx[..., :half]
        x2 = tx[..., half:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, tx: Tensor) -> Tensor:
        seq_len = tx.shape[2]
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]
        return tx * cos + self.rotate_half(tx) * sin
    

import math
class GQA(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__()
        self.cfg = cfg 
        self.Wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_k, bias=False)
        self.Wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_k, bias=False)
        self.Wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_k, bias=False)
        self.Wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rope = RoPE(d_k=cfg.d_k, seq_len=cfg.seq_len)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, tx: Tensor, mask = None) -> Tensor:
        batch, seq, _ = tx.shape 

        # first step: project
        Q: Tensor = self.Wq(tx) # (batch, seq, n_heads * d_k)
        K = self.Wk(tx)  # (batch, seq, n_kv_heads * d_k)
        V = self.Wv(tx)

        # second step: reshape into attention heads 
        Q = Q.view(batch, seq, self.cfg.n_heads, self.cfg.d_k).transpose(1, 2)  # (batch, 6, seq, 64)
        K = K.view(batch, seq, self.cfg.n_kv_heads, self.cfg.d_k).transpose(1, 2)
        V = V.view(batch, seq, self.cfg.n_kv_heads, self.cfg.d_k).transpose(1, 2)

        # apply rope 
        Q = self.rope(Q)  # (batch, 6, seq, 64)
        K: Tensor = self.rope(K) # (batch, 2, seq, 64)

        # expand K, V from n_kv_heads to n_heads
        K = K.repeat_interleave(self.cfg.n_groups, dim=1) # (batch, 6, seq, 64)
        V = V.repeat_interleave(self.cfg.n_groups, dim=1)
        
        # now attentino scores 
        scores = (torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.cfg.d_k))

        # casual mask 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax functoin 
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # weighted sum with value 
        output = torch.matmul(weights, V)

        # reshape back, should be contigous
        output = output.transpose(1, 2).contiguous().view(batch, seq, self.cfg.d_model)

        # output projection 
        return self.Wo(output)

import torch.nn.functional as F
class FFN(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__() 
        self.fc1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.fc2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, tx: Tensor) -> Tensor:
        x = self.fc1(tx) 
        x = F.relu(x).pow(2) 
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__()
        self.norm1 = RMSNorm()
        self.gqa = GQA(cfg)
        self.norm2 = RMSNorm()
        self.ffn = FFN(cfg)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, tx: Tensor, mask = None) -> Tensor:
        temp = tx 
        tx = self.norm1(tx)
        tx = self.gqa(tx, mask)
        tx = temp + self.dropout(tx)

        # ffn layer 
        temp = tx
        tx = self.norm2(tx)
        tx = self.ffn(tx)
        tx = temp + self.dropout(tx)

        return tx
    

class PocketTransformer(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.norm1 = RMSNorm()
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm2 = RMSNorm()
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tx: Tensor) -> Tensor:
        # apply embedding
        x = self.embedding(tx)
        x = self.norm1(x) 

        # casual mask 
        seq = x.shape[1]
        mask = torch.tril(torch.ones(seq, seq, device=x.device)).unsqueeze(0).unsqueeze(0)
        # (1, 1, seq, seq) 

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm2(x)
        logits = self.head(x)

        return logits