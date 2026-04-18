import torch
from bpe import HeapBPE
from tqdm import tqdm
import os

INPUT_FILE = "E:/gutenberg_books/books_corpus_cleaned_final_2.txt"
OUTPUT_FILE = "E:/triple_experiments/tokens.pt"
TEMP_FILE = "E:/triple_experiments/tokens.tmp"

CHUNK_SIZE = 5_000_000

tokenizer = HeapBPE()
tokenizer.load_tokenizer(
    vocab_path="triple_experiments/vocab.json",
    merges_path="triple_experiments/bpe_merges.txt"
)

file_size = os.path.getsize(INPUT_FILE)

with open(INPUT_FILE, "r", encoding="utf-8") as f, \
     open(TEMP_FILE, "wb") as out:

    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Processing") as pbar:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break

            tokens = tokenizer.encode(chunk)

            # write tokens directly (no RAM accumulation)
            tensor = torch.tensor(tokens, dtype=torch.int32)
            tensor.numpy().tofile(out)

            pbar.update(len(chunk))

# ---- convert raw binary → torch tensor ----
print("Finalizing tensor...")

tokens = torch.fromfile(TEMP_FILE, dtype=torch.int32)
torch.save(tokens, OUTPUT_FILE)

os.remove(TEMP_FILE)

print("Saved tokens to", OUTPUT_FILE)