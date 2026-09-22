import sentencepiece as spm
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
# Tokenization using SentencePiece
# ----------------------------
# SentencePiece is a subword tokenizer that can handle out-of-vocabulary words and
# create a more compact vocabulary. It is particularly useful for languages with rich morphology or large vocabularies. Here, we train a SentencePiece model on our toy corpus to create a tokenizer.
# Train a SentencePiece model on the toy corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()
    print("Corpus text:", text)

spm.SentencePieceTrainer.train(
    input='corpus.txt', model_prefix='tokenizer', vocab_size=40, model_type='bpe')

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')

# Encode the text using the trained tokenizer
ids = sp.encode(text, out_type=int)
print("Encoded IDs:", ids)

# Convert the encoded IDs into a tensor for model input
data = torch.tensor(ids, dtype=torch.long)
print("Data tensor:", data)

vocab_size = sp.get_piece_size()
print(vocab_size)


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

# context = torch.tensor([[word_to_idx["the"]]], dtype=torch.long)
# out = model.generate(context, max_new_tokens=15)

# print("  ".join([idx_to_word[idx.item()] for idx in out[0]]))


sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

context = torch.tensor([sp.encode("hello")], dtype=torch.long)

out = model.generate(context, max_new_tokens=20)

print("\nGenerated text:\n")
# print(" ".join(idx2word[int(i)] for i in out[0]))

generated_ids = out[0].tolist()
print(sp.decode(generated_ids))
