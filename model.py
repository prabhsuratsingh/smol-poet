import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
import re
from torch.utils.tensorboard.writer import SummaryWriter
import csv
import yaml

from bpe import BPE
from gqa_kv import GroupedQueryAttention
from swi_glu import SwiGLU

def load_config(exp_name, path="config/config.yaml",):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    exp_cfg = cfg["experiments"][exp_name]

    return exp_cfg
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, kv, dropout, device):
        super(TransformerBlock, self).__init__()

        self.attention = GroupedQueryAttention(embed_size, heads, kv=kv, dropout=dropout, device=device)
        self.norm1 = nn.RMSNorm(embed_size)
        self.norm2 = nn.RMSNorm(embed_size)

        hidden = int(2 * embed_size * 4 / 3)

        self.feed_forward = SwiGLU(embed_size, hidden)


        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, kv_cache=None):

        attn_out, kv_cache_out = self.attention(self.norm1(x), kv_cache)

        x = x + self.dropout(attn_out)

        x = x + self.dropout(
            self.feed_forward(self.norm2(x))
        )

        return x, kv_cache_out
    
class Decoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            target_vocab_size,
            embed_size,
            num_layers,
            heads, 
            kv,
            dropout,
            device,
    ):
        super(Decoder, self).__init__()

        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, kv, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.RMSNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, target_vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.fc_out.weight = self.word_embedding.weight


    def forward(self, x, kv_cache=None):
        x = self.dropout(self.word_embedding(x))
        new_cache = []

        for i, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[i]
            x, layer_cache_out = layer(x, layer_cache)
            new_cache.append(layer_cache_out)

        ln = self.dropout(self.norm(x))
        out = self.fc_out(ln)

        return out, new_cache


class Llama(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            kv=4,
            dropout=0,
            device="cuda",
    ):
        super(Llama, self).__init__()

        self.decoder = Decoder(
            vocab_size,
            vocab_size,
            embed_size,
            num_layers,
            heads,
            kv,
            dropout,
            device,
        )

        self.device = device
    
    def forward(self, target, kv_cache=None):

        out, new_cache = self.decoder(target, kv_cache)

        return out, new_cache
    
class LlamaDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.block_size = block_size

        tokens = tokenizer.encode(
            text,
            # allowed_special={"<s>", "</s>", "<unk>", "<|poem|>", "<|endpoem|>"}   this was 35M model trained only on poems dataset
            allowed_special={"<s>", "</s>", "<unk>", "<|book|>", "<|endbook|>"}
        )

        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size

        x = self.tokens[start:start+self.block_size]
        y = self.tokens[start+1:start+self.block_size+1]

        return x, y
    

'''
Generate Responses
'''
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    model.eval()

    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    kv_cache = None

    #prefill
    logits, kv_cache = model(tokens, kv_cache)

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        logits, kv_cache = model(next_token, kv_cache)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


'''
Utils
'''
def estimate_flops_per_token(model):
    return 6 * count_parameters(model)

def format_flops(flops):
    if flops > 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    if flops > 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    return f"{flops:.0f}"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, global_step, keep_last_n=3):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    filename = f"checkpoint_step_{global_step}.pt"
    filepath = os.path.join(CHECKPOINT_DIR, filename)

    # atomic save (safe)
    tmp_path = filepath + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, filepath)

    print(f"Checkpoint saved at step {global_step}")

    pattern = r"checkpoint_step_(\d+)\.pt"
    checkpoints = []

    for f in os.listdir(CHECKPOINT_DIR):
        match = re.match(pattern, f)
        if match:
            step = int(match.group(1))
            checkpoints.append((step, f))

    checkpoints.sort()

    if len(checkpoints) > keep_last_n:
        to_delete = checkpoints[:-keep_last_n]
        for _, fname in to_delete:
            os.remove(os.path.join(CHECKPOINT_DIR, fname))
            print(f"Deleted old checkpoint {fname}")

def get_latest_checkpoint():
    pattern = r"checkpoint_step_(\d+)\.pt"
    checkpoints = []

    for f in os.listdir(CHECKPOINT_DIR):
        match = re.match(pattern, f)
        if match:
            step = int(match.group(1))
            checkpoints.append((step, f))

    if not checkpoints:
        return None

    checkpoints.sort()
    latest_file = checkpoints[-1][1]
    return os.path.join(CHECKPOINT_DIR, latest_file)


'''
Main
'''
if __name__ == "__main__":
    config = load_config("small_35M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    exp_name = config["name"]

    CHECKPOINT_DIR = f"checkpoints/{exp_name}"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    LOG_DIR = f"logs/{exp_name}"
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    csv_file = open("training_metrics.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step",
        "epoch",
        "loss",
        "tokens_per_sec",
        "tflops"
    ])
    
    with open("poetry_corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPE()
    tokenizer.load_tokenizer(vocab_path="vocab.json", merges_path="bpe_merges.txt")

    vocab_size = len(tokenizer.vocab)

    model = Llama(
        vocab_size=vocab_size,
        embed_size=config["model"]["embed_size"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        kv=config["model"]["kv"],
        dropout=config["model"]["dropout"],
    ).to(device)

    dataset = LlamaDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=config["train"]["block_size"]
    )

    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"])
    criterion = nn.CrossEntropyLoss()

    latest_ckpt = get_latest_checkpoint()

    global_step = 0
    start_epoch = 0

    if latest_ckpt is not None:
        print(f"Resuming from {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"] + 1

    flops_per_token = estimate_flops_per_token(model)

    num_epochs = config["train"]["num_epochs"]
    log_interval = 100

    model.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        running_loss = 0.0

        t0 = time.time()

        for step, (x, y) in enumerate(loader):
            global_step += 1

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            step_start = time.time()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

            loss.backward()
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            writer.add_scalar("train/grad_norm", total_norm, global_step)

            optimizer.step()

            if global_step % 50000 == 0 and global_step > 0:
                save_checkpoint(model, optimizer, epoch, global_step, keep_last_n=3)

            step_time = time.time() - step_start

            tokens = x.numel()
            flops = flops_per_token * tokens
            tflops = flops / step_time / 1e12

            loss_val = loss.item()
            running_loss += loss_val
            epoch_loss += loss_val
            num_batches += 1

            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                tok_per_sec = tokens / step_time

                print(
                    f"step {global_step:6d} | "
                    f"epoch {epoch+1} | "
                    f"loss {avg_loss:.4f} | "
                    f"{tok_per_sec:8.0f} tok/s | "
                    f"{tflops:.2f} TFLOPs"
                )

                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/tokens_per_sec", tok_per_sec, global_step)
                writer.add_scalar("train/tflops", tflops, global_step)

                csv_writer.writerow([
                    global_step,
                    epoch+1,
                    avg_loss,
                    tok_per_sec,
                    tflops
                ])

                running_loss = 0.0

            if global_step % 10000 == 0:
                sample = generate(model, tokenizer, "O gentle night")
                writer.add_text("generation/sample", sample, global_step)

        avg_epoch_loss = epoch_loss / num_batches
        print(f"epoch {epoch+1}/{num_epochs} | avg loss {avg_epoch_loss:.4f}")


    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    torch.save(model.state_dict(), config["pretrained_model"])
    print("Model saved.")

    print("Test Generation : ")
    print(generate(model, tokenizer, "Once upon a time"))

    writer.close()
    csv_file.close()