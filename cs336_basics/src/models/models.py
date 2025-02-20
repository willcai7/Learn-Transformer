import torch
import torch.nn as nn
import math 


def softmax(x: torch.Tensor, dim: int):
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return e_x / e_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(query, key, value, mask=None, pdrop=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attention = softmax(scores, -1)
    if pdrop is not None:
        attention = nn.Dropout(pdrop)(attention)
    return torch.matmul(attention, value), attention

class RMSNorm(nn.Module):
    def __init__(self,d_model,epsilon=1e-5):
        super(RMSNorm, self).__init__()
        self.epsilon = epsilon
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon) * self.weight

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x/math.sqrt(2)))

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        # self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = GELU()
        
    def forward(self, x):
        x = self.activation(self.w1(x))
        # x = self.dropout(x)
        x = self.w2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,attn_pdrop=None):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
    
        self.attn_dropout = attn_pdrop
        
    def forward(self, x):
        batch_size, dim ,_ = x.size()
        mask = torch.triu(torch.ones(dim, dim), diagonal=1).bool().to(x.device)
        query = self.q_proj(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        x, attention = scaled_dot_product_attention(query, key, value, mask, pdrop=self.attn_dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        self.attention = attention
        x = self.output_proj(x)
        return x
    
    def load_state_dict_test(self, state_dict):
        weights = state_dict
        for i in range(self.num_heads):
            self.q_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"q_heads.{i}.weight"]
            self.k_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"k_heads.{i}.weight"]
            self.v_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"v_heads.{i}.weight"]
        self.output_Proj.weight.data = weights["output_proj.weight"]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop=None, residual_pdrop=None):
        super(TransformerBlock, self).__init__()

        self.attn = MultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = FFN(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.residual_pdrop = residual_pdrop
        self.drop1 = nn.Dropout(residual_pdrop)

        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.drop2 = nn.Dropout(residual_pdrop)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

class TransformerLM(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, context_length, attn_pdrop=None, residual_pdrop=None):
        super(TransformerLM, self).__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(residual_pdrop)
        self.ln_final = RMSNorm(d_model)

    def forward(self, x):
        B,T = x.size()
        positions = torch.arange(T, device=x.device, dtype=x.dtype).expand(B, T)
        x = self.token_embeddings(x) + self.position_embeddings(positions)
        x = self.drop(x)

        for block in self.layers:
            x = block(x)
        
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, target):
        # compute the loss 
        logits = nn.functional.log_softmax(logits, dim=-1)
        loss = -logits[range(target.size(0)), target]
        return loss.mean()