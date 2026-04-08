import torch
import matplotlib.pyplot as plt
import seaborn as sns

from model import Llama
from bpe import BPE

device = "cuda"

tokenizer = BPE()
tokenizer.load_tokenizer("from_amd_gpu/vocab.json", "from_amd_gpu/bpe_merges.txt")

model = Llama(
    vocab_size=len(tokenizer.vocab),
    embed_size=512,
    num_layers=12,
    heads=8,
    kv=4,
).to(device)
model.load_state_dict(torch.load("models/smol_35M/base/smol_poet.pt"))

text = "Shall I compare thee to a summer's day"
tokens = tokenizer.encode(text)

x = torch.tensor(tokens).unsqueeze(0).to(device)

with torch.no_grad():
    logits, cache, attention = model(x, return_attention=True)

attn = attention[10][0].mean(0).cpu().numpy()

tokens_str = [tokenizer.decode([t]) for t in tokens]

plt.figure(figsize=(8,6))
sns.heatmap(
    attn,
    xticklabels=tokens_str,
    yticklabels=tokens_str,
    cmap="viridis"
)

plt.title("Attention Head Visualization")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")

plt.show()