import torch
import torch.nn.functional as F
import copy
from torch.utils.tensorboard import SummaryWriter
import csv
import os

from ppo.ppo_reward import compute_reward
from model import Llama
from bpe import BPE

def save_checkpoint(step, policy, optimizer, path):
    checkpoint = {
        "step": step,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state().clone().cpu(),
        "cuda_rng_state": [s.clone().cpu() for s in torch.cuda.get_rng_state_all()]
    }
    torch.save(checkpoint, path)

LOG_DIR = "ppo_logs"
os.makedirs(LOG_DIR, exist_ok=True)

writer = SummaryWriter(LOG_DIR)

csv_file = open("rl_metrics.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "step",
    "reward",
    "loss",
    "kl",
    "ratio"
])

torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = "cuda"
DTYPE = torch.float32

PRETRAINED_PATH = "from_amd_gpu/smol_poet.pt"
VOCAB_PATH = "from_amd_gpu/vocab.json"
MERGES_PATH = "from_amd_gpu/bpe_merges.txt"

TEMPERATURE = 0.9
MAX_NEW_TOKENS = 64
K = 8
BETA_KL = 0.02
CLIP_EPS = 0.2
LR = 1e-6
RL_STEPS = 3000


tokenizer = BPE()
tokenizer.load_tokenizer(VOCAB_PATH, MERGES_PATH)
vocab_size = len(tokenizer.vocab)


policy = Llama(
    vocab_size=vocab_size,
    embed_size=512,
    num_layers=12,
    heads=8,
    kv=4,
).to(DEVICE)
policy.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
policy = policy.to(dtype=DTYPE)


ref_model = copy.deepcopy(policy).eval()
reward_model = copy.deepcopy(policy).eval()

for m in (ref_model, reward_model):
    for p in m.parameters():
        p.requires_grad = False

prompts = [
    "O gentle night",
    "In the silent forest",
    "Love is a flame",
    "The ocean speaks",
    "Time drifts slowly",
]

#Sampling
@torch.no_grad()
def sample(model, tokenizer, prompt, max_new_tokens=64, temperature=0.8, top_k=50, device="cuda"):
    model.eval()

    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    kv_cache = None
    logits, kv_cache = model(tokens, kv_cache)

    generated = tokens

    for _ in range(max_new_tokens):
        logits_step = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits_step, top_k)
            logits_step[logits_step < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits_step, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        logits, kv_cache = model(next_token, kv_cache)
        generated = torch.cat([generated, next_token], dim=1)

    return generated

#calculate logporbs
def compute_logprobs(model, tokens):
    logits, _ = model(tokens)

    log_probs = F.log_softmax(logits, dim=-1)

    # shift
    log_probs = log_probs[:, :-1, :]
    target = tokens[:, 1:]

    selected = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)

    return selected.sum(dim=-1)  # sum over sequence

def batch_rewards(texts):
    rewards = []
    for t in texts:
        r = compute_reward(t)
        r = max(min(r, 5.0), -5.0)
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32)

# PPO Step
def ppo_step(model, ref_model, optimizer, tokenizer, prompts, beta=0.02, device="cuda"):

    model.train()

    batch_tokens = []
    texts = []

    # ---- 1. Generate ----
    for p in prompts:
        tokens = sample(model, tokenizer, p, device=device)
        batch_tokens.append(tokens)
        texts.append(tokenizer.decode(tokens[0].tolist()))

    batch_tokens = torch.cat(batch_tokens, dim=0)

    # ---- 2. Compute rewards (CPU ok, small cost) ----
    rewards = batch_rewards(texts).to(device)

    # ---- 3. Logprobs ----
    logp = compute_logprobs(model, batch_tokens)
    with torch.no_grad():
        logp_ref = compute_logprobs(ref_model, batch_tokens)

    # ---- 4. KL ----
    kl = logp - logp_ref

    # ---- 5. Final reward ----
    final_reward = rewards - beta * kl

    # ---- 6. PPO loss (simplified) ----
    loss = -(final_reward.detach() * logp).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "reward": rewards.mean().item(),
        "kl": kl.mean().item()
    }

#PPO Loop
def run_ppo(
    model,
    tokenizer,
    prompts,
    steps=1000,
    batch_size=8,
    lr=1e-5,
    device="cuda"
):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # frozen reference model
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    for step in range(steps):

        batch_prompts = [prompts[i % len(prompts)] for i in range(step*batch_size, (step+1)*batch_size)]

        stats = ppo_step(
            model,
            ref_model,
            optimizer,
            tokenizer,
            batch_prompts,
            device=device
        )

        if step % 10 == 0:
            print(f"step {step} | loss {stats['loss']:.4f} | reward {stats['reward']:.4f} | KL {stats['kl']:.4f}")

        if step % 100 == 0:
            sample_text = tokenizer.decode(
                sample(model, tokenizer, "O gentle night", device=device)[0].tolist()
            )
            print("\n--- SAMPLE ---\n", sample_text, "\n--------------\n")