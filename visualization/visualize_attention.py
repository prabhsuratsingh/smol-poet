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
model.load_state_dict(torch.load("from_amd_gpu/smol_poet.pt"))

text = "Shall I compare thee to a summer's day"
tokens = tokenizer.encode(text)

x = torch.tensor(tokens).unsqueeze(0).to(device)

logits, cache, attention = model(x)

attn = attention[0][0].cpu().numpy()

plt.figure(figsize=(8,6))
sns.heatmap(attn, cmap="viridis")

plt.title("Attention Head Visualization")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")

plt.show()