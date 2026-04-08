import torch

from model import Llama
from bpe import BPE

DEVICE = "cuda"

VOCAB_PATH = "from_amd_gpu/vocab.json"
MERGES_PATH = "from_amd_gpu/bpe_merges.txt"

TEMPERATURE = 0.9
MAX_NEW_TOKENS = 150
K = 8
BETA_KL = 0.005
CLIP_EPS = 0.2
LR = 1e-6
RL_STEPS = 3000

torch.manual_seed(0)

tokenizer = BPE()
tokenizer.load_tokenizer("from_amd_gpu/vocab.json", "from_amd_gpu/bpe_merges.txt")

model_base = Llama(
    vocab_size=len(tokenizer.vocab),
    embed_size=512,
    num_layers=12,
    heads=8,
    kv=4,
).to(DEVICE)

model_post_rl = Llama(
    vocab_size=len(tokenizer.vocab),
    embed_size=512,
    num_layers=12,
    heads=8,
    kv=4,
).to(DEVICE)


model_base.load_state_dict(torch.load("from_amd_gpu/smol_poet.pt"))
model_post_rl.load_state_dict(torch.load("smol_poet_rl_final.pt"))

@torch.no_grad()
def sample(model, prompt):
    model.eval()

    ids = tokenizer.encode(prompt)
    tokens = torch.tensor(ids, device=DEVICE).unsqueeze(0)

    kv_cache = None
    logits, kv_cache = model(tokens, kv_cache)

    generated = []
    logprobs = []

    for _ in range(MAX_NEW_TOKENS):
        probs = torch.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)
        next_token = torch.multinomial(probs, 1)

        logprob = torch.log(probs.gather(-1, next_token))

        generated.append(next_token)
        logprobs.append(logprob)

        logits, kv_cache = model(next_token, kv_cache)

    gen = torch.cat(generated, dim=1)
    full = torch.cat([tokens, gen], dim=1)

    text = tokenizer.decode(full[0].tolist())

    return text

print("Test Generation : ")
print("1. Base 35M model : ")
print(sample(model_base, "Shall I compare thee"))
print()
print("2. PPO RL Fine Tune 35M model : ")
print(sample(model_post_rl, "Shall I compare thee"))