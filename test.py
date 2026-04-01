import torch

from model import Llama, generate
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

print("Test Generation : ")
print(generate(model, tokenizer, "O gentle night", max_new_tokens=150))