import torch
from tensorflow.python.layers.core import dropout

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    mps_device = torch.device("cpu")

else:
    mps_device = torch.device("mps")
    print("MPS device available")


# Hyperparameters
batch_size = 64
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 2e-4
device = mps_device
eval_iters = 200
n_embd = 384
torch.manual_seed(42)
n_layer = 6
dropout = 0.3
n_head = 8

class Head(torch.nn.Module):
    """
    A simple Head of self attention
    """
    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril_mask', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1] * C**-0.5 # BTC @ BCT -> BTT
        # mask out the upper triangular part to make sure that tokens from the future do not affect the current token
        wei = wei.masked_fill(self.tril_mask[:T, :T] == 0, float('-inf'))
        wei = torch.nn.functional.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(torch.nn.Module):
    """
    A simple multi-head attention module
    """
    def __init__(self, n_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = torch.nn.Linear(n_embd, n_embd)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(torch.nn.Module):
    """
    A simple feed forward module
    """
    def __init__(self, n_embd):
        super(FeedForward, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, n_embd*4),
            torch.nn.ReLU(),
            torch.nn.Linear(n_embd*4, n_embd),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    """
    A simple transformer block with communication done by Multi Headed Self Attention
    followed by computation done by the feed forward network
    """
    def __init__(self, n_embd, n_head):
        super(Block, self).__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.sa(self.ln1(x)) + x # adding residual connection
        x = self.ffwd(self.ln2(x)) + x
        return x


class LanguageModel(torch.nn.Module):
    """
    A simple transformer language model
    """
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)
        self.blocks = torch.nn.Sequential(*[Block(n_embd, 4) for _ in range(n_layer)] + [torch.nn.LayerNorm(n_embd)])
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are of shape (B, T) tensor of integers where B is the batch size and T is the sequence length
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx