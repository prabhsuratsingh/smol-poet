import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bpe import BPE
from gqa_kv import GroupedQueryAttention
from swi_glu import SwiGLU

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
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.tokens = tokenizer.encode(
            text,
            allowed_special={"<s>", "</s>", "<unk>"}

        )

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(
            self.tokens[idx : idx + self.block_size],
            dtype=torch.long
        )
        y = torch.tensor(
            self.tokens[idx + 1 : idx + self.block_size + 1],
            dtype=torch.long
        )
        return x, y
    
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPE()
    tokenizer.load_tokenizer(vocab_path="vocab.json", merges_path="bpe_merges.txt")

    vocab_size = len(tokenizer.vocab)

    model = Llama(
        vocab_size=vocab_size,
        embed_size=256,
        num_layers=6,
        heads=8,
        kv=4,
    ).to(device)

    dataset = LlamaDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=128
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    flops_per_token = estimate_flops_per_token(model)

    num_epochs = 5
    log_interval = 100
    global_step = 0

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        running_loss = 0.0

        t0 = time.time()

        for step, (x, y) in enumerate(loader):
            global_step += 1

            x = x.to(device)
            y = y.to(device)

            step_start = time.time()

            logits, _ = model(x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

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
                running_loss = 0.0

        avg_epoch_loss = epoch_loss / num_batches
        print(f"epoch {epoch+1}/{num_epochs} | avg loss {avg_epoch_loss:.4f}")


    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    torch.save(model.state_dict(), "llama_shakespeare.pt")
    print("Model saved.")

    print("Test Generation : ")
    print(generate(model, tokenizer, "Once upon a time"))   