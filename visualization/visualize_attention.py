import torch
import matplotlib.pyplot as plt
import seaborn as sns

from model import Llama
from bpe import BPE

device = "cuda"

tokenizer = BPE()
tokenizer.load_tokenizer("vocab.json", "bpe_merges.txt")

model = Llama(vocab_size=len(tokenizer.vocab)).to(device)
model.load_state_dict(torch.load("smol_poet.pt"))

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