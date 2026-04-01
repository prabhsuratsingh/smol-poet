import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from bpe import BPE
from model import Llama

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

embeddings = model.decoder.word_embedding.weight.detach().cpu()

pca = PCA(n_components=2)
proj = pca.fit_transform(embeddings[:500])

plt.figure(figsize=(8,6))
plt.scatter(proj[:,0], proj[:,1], alpha=0.5)

plt.title("Token Embedding PCA Projection")
plt.show()