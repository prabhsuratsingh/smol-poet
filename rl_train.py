import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import csv
import os

from model import Llama
from bpe import BPE

def save_checkpoint(step, policy, optimizer, path):
    checkpoint = {
        "step": step,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
    }
    torch.save(checkpoint, path)

# def save_checkpoint(step, policy, optimizer, path):
#     checkpoint = {
#         "step": step,
#         "model_state": policy.state_dict(),
#         "optimizer_state": optimizer.state_dict(),
#         "rng_state": torch.get_rng_state().clone().cpu(),
#         "cuda_rng_state": [s.clone().cpu() for s in torch.cuda.get_rng_state_all()]
#     }
#     torch.save(checkpoint, path)

LOG_DIR = "rl_logs"
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
DTYPE = torch.bfloat16

PRETRAINED_PATH = "from_amd_gpu/smol_poet.pt"
VOCAB_PATH = "from_amd_gpu/vocab.json"
MERGES_PATH = "from_amd_gpu/bpe_merges.txt"

TEMPERATURE = 0.9
MAX_NEW_TOKENS = 64
K = 8
BETA_KL = 0.005
CLIP_EPS = 0.2
LR = 1e-6
RL_STEPS = 3000

PROMPTS = [
    "O ",
    "Fair ",
    "Upon the night",
    "My love,",
    "Shall I compare thee",
    "When in disgrace",
    "If music be the food",
]


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

    old_logprob = torch.cat(logprobs).sum().detach()

    text = tokenizer.decode(full[0].tolist())

    return text, full, old_logprob

@torch.no_grad()
def reward_fn(tokens):
    text = tokenizer.decode(tokens[0].tolist())

    base = sequence_logprob(reward_model, tokens).item()

    keywords = ["thee", "thou", "thy", "doth", "hath"]
    style_bonus = sum(word in text for word in keywords)

    return base + 2.0 * style_bonus

optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

policy.train()

RESUME_PATH = "rl/checkpoint_step2500.pt"
start_step = 1

if RESUME_PATH and os.path.exists(RESUME_PATH):
    print(f"Loading checkpoint from {RESUME_PATH}")

    checkpoint = torch.load(
        RESUME_PATH,
        map_location=DEVICE,
        weights_only=True 
    )

    policy.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_step = checkpoint["step"] + 1

    rng_state = checkpoint["rng_state"]

    if not isinstance(rng_state, torch.Tensor):
        rng_state = torch.tensor(rng_state, dtype=torch.uint8)
    else:
        rng_state = rng_state.clone().detach()

    if rng_state.dtype != torch.uint8:
        rng_state = rng_state.to(torch.uint8)

    rng_state = rng_state.cpu()

    torch.set_rng_state(rng_state)

    cuda_rng_state = checkpoint["cuda_rng_state"]

    fixed_states = []
    for s in cuda_rng_state:
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.uint8)
        else:
            s = s.clone().detach()

        if s.dtype != torch.uint8:
            s = s.to(torch.uint8)

        fixed_states.append(s.cpu())

    torch.cuda.set_rng_state_all(fixed_states)

    print(f"Resuming from step {start_step}")

for step in range(start_step, RL_STEPS + 1):
    prompt = random.choice(PROMPTS)

    texts = []
    rewards = []
    token_batches = []
    old_logprobs = []

    for _ in range(K):

        txt, tokens, old_lp = sample(policy, prompt)

        r = reward_fn(tokens)

        texts.append(txt)
        rewards.append(r)
        token_batches.append(tokens)
        old_logprobs.append(old_lp)

    baseline = sum(rewards) / K
    advantages = torch.tensor(rewards, device=DEVICE)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    loss = 0
    total_kl = 0
    total_ratio = 0

    for A, tokens, old_lp in zip(advantages, token_batches, old_logprobs):

        new_logp = sequence_logprob(policy, tokens)

        ratio = torch.exp(new_logp - old_lp)
        clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)

        ppo_loss = -torch.min(
            ratio * A,
            clipped_ratio * A
        ).mean()

        with torch.no_grad():
            ref_logp = sequence_logprob(ref_model, tokens)

        kl = (new_logp - ref_logp).mean()

        loss = loss + (ppo_loss + BETA_KL * kl)

        total_kl += kl.item()
        total_ratio += ratio.item()

    loss = loss / K
    avg_kl = total_kl / K
    avg_ratio = total_ratio / K

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    writer.add_scalar("rl/reward", baseline, step)
    writer.add_scalar("rl/loss", loss.item(), step)
    writer.add_scalar("rl/kl", avg_kl, step)
    writer.add_scalar("rl/ratio", avg_ratio, step)

    csv_writer.writerow([
        step,
        baseline,
        loss.item(),
        avg_kl,
        avg_ratio
    ])

    if step % 50 == 0:
        print(f"step {step:5d} | reward {baseline:.3f}")

    if step % 200 == 0:
        sample_text, _, _ = sample(policy, "Shall I compare thee")
        writer.add_text("rl/sample", sample_text, step)

    if step % 500 == 0:
        save_checkpoint(step, policy, optimizer, f"rl/checkpoint_step{step}.pt")


print("RL training complete.")
torch.save(policy.state_dict(), "smol_poet_rl_final.pt")
writer.close()
csv_file.close()