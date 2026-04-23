import random
import torch
import torch.nn.functional as F
import copy
from torch.utils.tensorboard import SummaryWriter
import csv
import os
from transformers import AutoTokenizer
import yaml
from torch.nn.utils.rnn import pad_sequence

from ppo.ppo_reward import compute_reward
from model import Llama

def load_config(exp_name, path="config/config.yaml",):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    exp_cfg = cfg["experiments"][exp_name]

    return exp_cfg

PRETRAINED_PATH = "smol_poet_100M.pt"
FILE_SAVE_PATH = "smol_poet_100M"

def save_checkpoint(step, policy, optimizer, path):
    checkpoint = {
        "step": step,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state().clone().cpu(),
        "cuda_rng_state": [s.clone().cpu() for s in torch.cuda.get_rng_state_all()]
    }
    torch.save(checkpoint, path)

LOG_DIR = f"ppo_logs/{FILE_SAVE_PATH}"
os.makedirs(LOG_DIR, exist_ok=True)

writer = SummaryWriter(LOG_DIR)

csv_file = open(f"rl_metrics_{FILE_SAVE_PATH}.csv", "w", newline="")
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
# DTYPE = torch.float32
DTYPE = torch.bfloat16


TEMPERATURE = 0.7
MAX_NEW_TOKENS = 64
K = 8
# BETA_KL = 0.02
BETA_KL = 0.08
CLIP_EPS = 0.2
# LR = 1e-6
LR = 2e-7
RL_STEPS = 3000
steps = 3000


#Sampling
@torch.no_grad()
def sample(model, tokenizer, prompt, max_new_tokens=64, temperature=0.8, top_k=50, device="cuda"):
    model.eval()

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    prompt_len = tokens.shape[1]

    kv_cache = None
    logits, kv_cache = model(tokens, kv_cache)

    generated = tokens
    logprobs = []

    for _ in range(max_new_tokens):
        logits_step = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits_step, top_k)
            logits_step[logits_step < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits_step, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        logprob = torch.log(probs.gather(-1, next_token))

        logprobs.append(logprob)

        logits, kv_cache = model(next_token, kv_cache)
        generated = torch.cat([generated, next_token], dim=1)

    logprobs = torch.cat(logprobs, dim=1)  # (1, T)

    return generated, logprobs, prompt_len


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
    batch_logprobs_old = []
    texts = []
    prompt_lens = []

    # ---- 1. Sample ----
    for p in prompts:
        tokens, logprobs, prompt_len = sample(model, tokenizer, p, device=device)

        batch_tokens.append(tokens)
        batch_logprobs_old.append(logprobs)
        prompt_lens.append(prompt_len)

        texts.append(tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True))


    # remove batch dim
    batch_tokens = [t.squeeze(0) for t in batch_tokens]
    batch_logprobs_old = [lp.squeeze(0) for lp in batch_logprobs_old]

    # pad tokens
    batch_tokens = pad_sequence(
        batch_tokens,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    # pad logprobs
    batch_logprobs_old = pad_sequence(
        batch_logprobs_old,
        batch_first=True,
        padding_value=0.0
    )

    logprobs_old = batch_logprobs_old

    # ---- 2. Rewards ----
    rewards = batch_rewards(texts).to(device)
    rewards = rewards / 5.0   # scale to ~[-1, 1]

    # ---- 3. Recompute logprobs (new policy) ----
    logits, _ = model(batch_tokens)
    log_probs = F.log_softmax(logits, dim=-1)

    log_probs = log_probs[:, :-1, :]
    targets = batch_tokens[:, 1:]

    selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    # ---- 4. Mask prompt ----
    masks = []
    for i, pl in enumerate(prompt_lens):
        mask = torch.zeros_like(selected[i])
        mask[pl:] = 1.0
        masks.append(mask)

    masks = torch.stack(masks, dim=0)

    # get max generated length
    T_gen = logprobs_old.shape[1]

    # take only last T_gen tokens from new logprobs
    logprobs = selected[:, -T_gen:]

    # apply mask only on generated region
    mask_gen = masks[:, -T_gen:]

    logprobs = logprobs * mask_gen
    logprobs_old = logprobs_old * mask_gen

    # ---- 5. PPO ratio ----
    ratio = torch.exp(logprobs - logprobs_old)
    ratio = torch.clamp(ratio, 0, 5.0)

    # expand rewards to token level
    adv = rewards.unsqueeze(1).expand_as(logprobs)

    # ---- 6. Clipped objective ----
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv

    loss = -torch.min(surr1, surr2).mean()

    # ---- 7. KL penalty ----
    with torch.no_grad():
        logits_ref, _ = ref_model(batch_tokens)
        log_probs_ref = F.log_softmax(logits_ref, dim=-1)

    log_probs_ref = log_probs_ref[:, :-1, :]
    selected_ref = log_probs_ref.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    kl = (selected - selected_ref) * masks
    kl = kl.mean()

    loss = loss + beta * kl

    # ---- 8. optimize ----
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "reward": rewards.mean().item(),
        "kl": kl.item(),
        "ratio": ratio.mean().item()
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

    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    for step in range(steps):

        batch_prompts = random.choices(prompts, k=batch_size)

        stats = ppo_step(
            model,
            ref_model,
            optimizer,
            tokenizer,
            batch_prompts,
            beta=BETA_KL,
            device=device
        )

        writer.add_scalar("ppo/loss", stats["loss"], step)
        writer.add_scalar("ppo/reward", stats["reward"], step)
        writer.add_scalar("ppo/kl", stats["kl"], step)
        writer.add_scalar("ppo/ratio", stats["ratio"], step)

        csv_writer.writerow([
            step,
            stats["reward"],
            stats["loss"],
            stats["kl"],
            stats["ratio"]
        ])


        if step % 10 == 0:
            print(
                f"step {step} | "
                f"loss {stats['loss']:.4f} | "
                f"reward {stats['reward']:.4f} | "
                f"KL {stats['kl']:.4f} | "
                f"ratio {stats['ratio']:.4f}"
            )

        if step % 100 == 0:
            sample_tokens, _, _ = sample(
                model,
                tokenizer,
                "O gentle night",
                device=device
            )

            sample_text = tokenizer.decode(
                sample_tokens[0].tolist(),
                skip_special_tokens=True
            )

            print("\n--- SAMPLE ---\n", sample_text, "\n--------------\n")

            writer.add_text("ppo/sample", sample_text, step)


        if step % 500 == 0 and step > 0:
            save_checkpoint(
                step,
                model,
                optimizer,
                f"ppo_{FILE_SAVE_PATH}_ckpt_step_{step}.pt"
            )

    writer.close()
    csv_file.close()


if __name__ == "__main__":
    config = load_config("base_100M")
    exp_name = config["name"]
    PRETRAINED_PATH = config["pretrained_model"]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    special_tokens = {
        "additional_special_tokens": [
            "<|book|>",
            "<|endbook|>"
        ]
    }

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)


    policy = Llama(
        vocab_size=vocab_size,
        embed_size=config["model"]["embed_size"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        kv=config["model"]["kv"],
        dropout=config["model"]["dropout"],
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
    
    run_ppo(
        model=policy,
        tokenizer=tokenizer,
        prompts=prompts,
        steps=RL_STEPS,
        batch_size=K,
        lr=LR,
        device=DEVICE
    )