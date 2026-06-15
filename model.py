# Copyright 2026 Dmitry
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint as cp

class GPTConfig:
    def __init__(self, vocab_size=40960, embed_dim=1024, n_layers=16, n_heads=16,
                 n_kv_heads=None, intermediate_size=2560, max_seq_len=2048,
                 dropout=0.0, use_lora=False, num_experts=4, top_k=2, tie_word_embeddings=False, window_size=1024, anchor_size=64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_lora = use_lora
        self.num_experts = num_experts
        self.top_k = top_k
        self.tie_word_embeddings = tie_word_embeddings
        self.window_size = window_size
        self.anchor_size = anchor_size

def precompute_freqs_cis(dim: int, end: int, anchor_size: int = 64, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class VexNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.abs().mean(-1, keepdim=True)
        x = x / (variance + self.eps)
        return (x.to(input_dtype) * self.weight)

class DoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=32, alpha=16, dropout=0.05):
        super().__init__()
        self.linear = linear_layer
        in_features = linear_layer.weight.shape[1]
        out_features = linear_layer.weight.shape[0]
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.lora_A, std=1 / rank)
        nn.init.zeros_(self.lora_B)
        self.lora_m = nn.Parameter(self.linear.weight.data.norm(p=2, dim=1, keepdim=True))

    def forward(self, x):
        W = self.linear.weight
        lora_weight = (self.lora_A @ self.lora_B).T * self.scaling
        W_modified = W + lora_weight
        norm = W_modified.to(torch.float32).norm(p=2, dim=1, keepdim=True).to(W_modified.dtype)
        W_dora = self.lora_m * (W_modified / norm)
        return F.linear(self.dropout(x), W_dora, self.linear.bias)

class DiffCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.embed_dim
        self.head_dim = self.n_embd // self.n_heads
        
        self.window_size = getattr(config, 'window_size', 512) 
        self.anchor_size = getattr(config, 'anchor_size', 64)  
        
        self.c_attn = nn.Linear(self.n_embd, 5 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.lambda_noise = nn.Parameter(torch.zeros(self.n_heads, 1, 1))

        self.diff_norm = VexNorm(self.n_embd)

        max_len = config.max_seq_len
        mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, freqs_cis=None, use_cache=False, past_kv=None):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q1, q2, k1, k2, v = qkv.split(self.n_embd, dim=2)
        
        q1 = q1.view(B, T, self.n_heads, self.head_dim)
        q2 = q2.view(B, T, self.n_heads, self.head_dim)
        k1 = k1.view(B, T, self.n_heads, self.head_dim)
        k2 = k2.view(B, T, self.n_heads, self.head_dim)
        v  =  v.view(B, T, self.n_heads, self.head_dim)

        if freqs_cis is not None:
            q1, k1 = apply_rotary_emb(q1, k1, freqs_cis)
            q2, k2 = apply_rotary_emb(q2, k2, freqs_cis)

        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)
        v  =  v.transpose(1, 2)

        if use_cache:
            if past_kv is not None:
                past_k1, past_k2, past_v = past_kv
                k1 = torch.cat([past_k1, k1], dim=2)
                k2 = torch.cat([past_k2, k2], dim=2)
                v  = torch.cat([past_v, v], dim=2)
            past_kv = (k1, k2, v)
        else:
            past_kv = None

        seq_len_kv = k1.size(2)

        att1 = (q1 @ k1.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        if seq_len_kv > 1:
            q_pos = torch.arange(seq_len_kv - T, seq_len_kv, device=x.device).unsqueeze(1)
            k_pos = torch.arange(seq_len_kv, device=x.device).unsqueeze(0)
            
            window_mask = (k_pos >= q_pos - self.window_size) | (k_pos < self.anchor_size)
            
            precomputed_causal = self.causal_mask[:, :, :T, :seq_len_kv]
            valid_mask = (precomputed_causal > 0.5) & window_mask
            
            final_mask = torch.zeros_like(att1).masked_fill_(~valid_mask, float("-inf"))
            
            att1 = att1 + final_mask
            att2 = att2 + final_mask
        
        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)
        
        noise_canceller = torch.exp(self.lambda_noise).clamp(max=2.0)
        diff_att = att1 - (noise_canceller * att2)
        
        y = diff_att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.diff_norm(y)
        return self.c_proj(y), past_kv

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.embed_dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.embed_dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.intermediate_size, config.embed_dim, bias=False)
        self.w3.GPT_SCALE_INIT = True
        
        if config.use_lora:
            self.w1 = DoRALinear(self.w1)
            self.w2 = DoRALinear(self.w2)
            self.w3 = DoRALinear(self.w3)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class VexionMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = getattr(config, 'top_k', 2) 
        
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(self.num_experts)])
        self.router = nn.Linear(config.embed_dim, self.num_experts, bias=False)
        self.register_buffer('expert_usage_tracker', torch.zeros(self.num_experts))

    def forward(self, x):
        B, T, C = x.shape 
        x_flat = x.view(-1, C) 

        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        if self.training:
            with torch.no_grad(): 
                batch_usage = routing_weights.sum(dim=0) 
                self.expert_usage_tracker += batch_usage

        topk_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        tokens_per_expert = torch.bincount(selected_experts.flatten(), minlength=self.num_experts)
        route_fraction = tokens_per_expert.float() / selected_experts.numel()
        mean_probs = routing_weights.mean(dim=0) 
        aux_loss = self.num_experts * torch.sum(mean_probs * route_fraction)

        final_output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue 

            expert_tokens = x_flat[expert_mask]
            expert_out = expert(expert_tokens)

            idx_in_topk = (selected_experts[expert_mask] == i).nonzero(as_tuple=True)[1]
            token_weights = topk_weights[expert_mask, idx_in_topk].unsqueeze(-1)
            
            temp_output = torch.zeros_like(x_flat)
            temp_output[expert_mask] = expert_out * token_weights
            final_output = final_output + temp_output

        return final_output.view(B, T, C), aux_loss

    @torch.no_grad() 
    def mutate_dead_experts(self, optimizer=None, threshold_ratio=0.01, noise_factor=0.05):
        total_tokens = self.expert_usage_tracker.sum().item()
        if total_tokens == 0: return 0

        best_expert_idx = torch.argmax(self.expert_usage_tracker).item()
        best_expert = self.experts[best_expert_idx]
        mutated_count = 0
        
        for i in range(self.num_experts):
            usage_ratio = self.expert_usage_tracker[i].item() / total_tokens
            if usage_ratio < threshold_ratio and i != best_expert_idx:
                dead_expert = self.experts[i]
                for dead_param, best_param in zip(dead_expert.parameters(), best_expert.parameters()):
                    dead_param.data.copy_(best_param.data)
                    if optimizer is not None and dead_param in optimizer.state:
                        del optimizer.state[dead_param]
                    noise = torch.randn_like(dead_param.data)
                    scaled_noise = noise * noise_factor * best_param.data.std()
                    dead_param.data.add_(scaled_noise)
                self.router.weight.data[i].zero_() 
                mutated_count += 1
                
        self.expert_usage_tracker.zero_()
        return mutated_count

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = VexNorm(config.embed_dim)
        self.attn = DiffCausalSelfAttention(config)
        self.ln_2 = VexNorm(config.embed_dim)
        self.mlp = VexionMoE(config) 

    def forward(self, x, freqs_cis, use_cache=False, past_kv=None):
        attn_out, past_kv_out = self.attn(self.ln_1(x), freqs_cis, use_cache, past_kv)
        x = x + attn_out
        
        mlp_out, aux_loss = self.mlp(self.ln_2(x))
        x = x + mlp_out
        
        return x, aux_loss, past_kv_out

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_dim),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = VexNorm(config.embed_dim),
        ))
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

        self.register_buffer(
            "freqs_cis", 
            precompute_freqs_cis(
                config.embed_dim // config.n_heads, 
                config.max_seq_len * 2,
                anchor_size=config.anchor_size  
            ), 
            persistent=False
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT') and module.GPT_SCALE_INIT:
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache=False, past_key_values=None):
        b, t = idx.size()
        
        start_pos = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        freqs_cis = self.freqs_cis[start_pos : start_pos + t]

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)  

        if self.training and not x.requires_grad:
            x.requires_grad_(True)
        
        total_aux_loss = 0.0 
        new_past_key_values = () if use_cache else None
        
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if self.training:
                x, aux_loss, _ = cp.checkpoint(block, x, freqs_cis, False, None, use_reentrant=False)
            else:
                x, aux_loss, past_kv_out = block(x, freqs_cis, use_cache, past_kv)
                if use_cache:
                    new_past_key_values += (past_kv_out,)
                    
            total_aux_loss += aux_loss 
                
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
            if self.training:
                loss = loss + 0.01 * total_aux_loss
        else:
            logits = self.lm_head(x[:, [-1], :])  
            loss = None

        if use_cache:
            return logits, loss, new_past_key_values
        return logits, loss
