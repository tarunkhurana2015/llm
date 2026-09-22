import torch
import torch.nn as nn
from torch.nn import functional as F
import random

# Import the transformer block we created separately.
from transformer_block import Block

# ----------------------------
# Environment and runtime checks
# ----------------------------
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "No GPU available")

# ----------------------------
# Tiny toy corpus for language modeling practice
# ----------------------------
# Each sentence represents a small, human-readable example of how a model sees
# patterns in text. We add a special stopper token so the model learns boundaries.
corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]

corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
print(text)

# ----------------------------
# Tokenization: simple whitespace-based vocabulary
# ----------------------------
# For a tiny toy example, using whitespace as the split signal keeps the pipeline
# simple and visually easy to understand. Every unique token becomes a vocabulary item.
words = list(set(text.split()))
print("Unique words:", words)

vocab_size = len(words)
print("Vocabulary size:", vocab_size)

# Map words to integer IDs so the model can work with numeric tensors instead of raw strings.
word_to_idx = {word: idx for idx, word in enumerate(words)}
print("Word to index mapping:", word_to_idx)

# Reverse lookup for readability when inspecting outputs or debugging predictions.
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
print("Index to word mapping:", idx_to_word)

# Convert the entire text into a long integer tensor of token IDs.
data = torch.tensor([word_to_idx[word]
                    for word in text.split()], dtype=torch.long)
print("Data tensor:", data)

block_size = 6  # Number of tokens the model sees at once
embedding_dim = 32  # Size of the embedding vector for each token
n_heads = 2  # Number of attention heads in the multi-head attention mechanism
n_layers = 2  # Number of transformer blocks in the model
lr = 1e-3  # Learning rate for the optimizer
epochs = 1500  # Number of training epochs


def get_batch(batch_size=16):
    # Randomly sample starting indices for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


'''
Data tensor: tensor([41, 18, 11, 40, 22, 16, 34, 20,  6, 21, 29, 16,  4,  0,  6,  1, 16, 34,
         9, 19,  2, 40,  8, 16, 17,  6, 15,  5, 39, 16, 34, 32,  6, 28, 37, 16,
        13, 26, 38,  3, 24, 35, 20, 16, 12,  6,  4, 25, 23, 16, 27, 30, 36, 24,
        33, 16, 14,  7, 34, 31, 10, 16])


'''


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim)  # (42, 32)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        token_embeddings = self.token_embedding(x)  # (B, T, embedding_dim)
        position_embeddings = self.position_embedding(
            torch.arange(T))  # (T, embedding_dim)
        x = token_embeddings + position_embeddings  # (B, T, embedding_dim)
        x = self.blocks(x)  # (B, T, embedding_dim)
        x = self.ln_f(x)  # (B, T, embedding_dim)
        logits = self.head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = TinyTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    x_batch, y_batch = get_batch()
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"Step {step}, Loss: {loss.item(): .4f}")

context = torch.tensor([[word_to_idx["hello"]]], dtype=torch.long)
out = model.generate(context, max_new_tokens=15)

print("  ".join([idx_to_word[idx.item()] for idx in out[0]]))
