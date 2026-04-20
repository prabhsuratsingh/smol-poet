import torch
import yaml
from transformers import AutoTokenizer
from model import Llama

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(exp_name, path="config/config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["experiments"][exp_name]


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    special_tokens = {
        "additional_special_tokens": [
            "<|book|>",
            "<|endbook|>"
        ]
    }

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=150,
    temperature=0.8,
    top_k=50
):
    model.eval()

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor(ids, device=DEVICE).unsqueeze(0)

    kv_cache = None
    logits, kv_cache = model(tokens, kv_cache)

    generated = []

    for _ in range(max_new_tokens):
        logits_step = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits_step, top_k)
            logits_step[logits_step < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits_step, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token)

        logits, kv_cache = model(next_token, kv_cache)

    gen = torch.cat(generated, dim=1)
    full = torch.cat([tokens, gen], dim=1)

    return tokenizer.decode(full[0].tolist(), skip_special_tokens=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="experiment name from config")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()

    config = load_config(args.exp)

    tokenizer = build_tokenizer()

    model = Llama(
        vocab_size=len(tokenizer),
        embed_size=config["model"]["embed_size"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        kv=config["model"]["kv"],
        dropout=config["model"]["dropout"],
    ).to(DEVICE)

    print("Loading model:", config["pretrained_model"])
    model.load_state_dict(torch.load(config["pretrained_model"], map_location=DEVICE))

    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print("\n--- OUTPUT ---\n")
    print(output)
    print("\n--------------\n")


if __name__ == "__main__":
    main()