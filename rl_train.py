import copy
import random
import torch
import torch.nn.functional as F

from model import Llama
from bpe import BPE

torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = "cuda"
DTYPE = torch.bfloat16

PRETRAINED_PATH = "smol_poet.pt"
VOCAB_PATH = "vocab.json"
MERGES_PATH = "bpe_merges.txt"

TEMPERATURE = 0.8
MAX_NEW_TOKENS = 64
K = 4                 
BETA_KL = 0.02
LR = 1e-5
RL_STEPS = 3000

PROMPTS = [
    "ROMEO:",
    "JULIET:",
    "KING:",
    "First Citizen:",
    "O my lord,",
    "To be, or not to be",
]


tokenizer = BPE()
tokenizer.load_tokenizer(VOCAB_PATH, MERGES_PATH)
vocab_size = len(tokenizer.vocab)


policy = Llama(vocab_size=vocab_size).to(DEVICE)
policy.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
policy = policy.to(dtype=DTYPE)


ref_model = copy.deepcopy(policy).eval()
reward_model = copy.deepcopy(policy).eval()

for m in (ref_model, reward_model):
    for p in m.parameters():
        p.requires_grad = False


def sequence_logprob(model, tokens):
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    logits, _ = model(x)
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)

    return token_logp.sum(dim=1) 


@torch.no_grad()
def sample(model, prompt):
    model.eval()

    ids = tokenizer.encode(prompt)
    tokens = torch.tensor(ids, device=DEVICE).unsqueeze(0)

    kv_cache = None
    logits, kv_cache = model(tokens, kv_cache)

    generated = []

    for _ in range(MAX_NEW_TOKENS):
        probs = torch.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)
        next_token = torch.multinomial(probs, 1)

        generated.append(next_token)
        logits, kv_cache = model(next_token, kv_cache)

    gen = torch.cat(generated, dim=1)
    full = torch.cat([tokens, gen], dim=1)

    text = tokenizer.decode(full[0].tolist())
    return text, full


@torch.no_grad()
def reward_fn(text_tokens):
    return sequence_logprob(reward_model, text_tokens).item()


optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)


policy.train()

for step in range(1, RL_STEPS + 1):
    prompt = random.choice(PROMPTS)

    texts = []
    rewards = []
    token_batches = []

    for _ in range(K):
        txt, tokens = sample(policy, prompt)
        r = reward_fn(tokens)

        texts.append(txt)
        rewards.append(r)
        token_batches.append(tokens)

    baseline = sum(rewards) / K
    advantages = [r - baseline for r in rewards]

    loss = 0

    for A, tokens in zip(advantages, token_batches):
        logp = sequence_logprob(policy, tokens)

        with torch.no_grad():
            ref_logp = sequence_logprob(ref_model, tokens)

        kl = (logp - ref_logp)
        loss = loss + (-A * logp + BETA_KL * kl)

    loss = loss / K

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step:5d} | reward {baseline:.3f}")

    if step % 500 == 0:
        torch.save(policy.state_dict(), f"smol_poet_rl_step{step}.pt")

print("RL training complete.")
torch.save(policy.state_dict(), "smol_poet_rl_final.pt")
