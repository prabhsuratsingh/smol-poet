import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from model import Llama
from bpe import BPE

device = "cuda"

tokenizer = BPE()
tokenizer.load_tokenizer("vocab.json", "bpe_merges.txt")

model = Llama(vocab_size=len(tokenizer.vocab)).to(device)
model.load_state_dict(torch.load("smol_poet.pt"))

embeddings = model.decoder.word_embedding.weight.detach().cpu()

pca = PCA(n_components=2)
proj = pca.fit_transform(embeddings[:500])

plt.figure(figsize=(8,6))
plt.scatter(proj[:,0], proj[:,1], alpha=0.5)

plt.title("Token Embedding PCA Projection")
plt.show()