import torch
from tqdm import tqdm
import os
from transformers import AutoTokenizer

INPUT_FILE = "E:/gutenberg_books/books_corpus_cleaned_final_2.txt"
OUTPUT_FILE = "E:/triple_experiments/tokens.pt"
TEMP_FILE = "E:/triple_experiments/tokens.tmp"

CHUNK_SIZE = 500_000

tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens = {
    "additional_special_tokens": [
        "<|book|>",
        "<|endbook|>"
    ]
}
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token

file_size = os.path.getsize(INPUT_FILE)

with open(INPUT_FILE, "r", encoding="utf-8") as f, \
     open(TEMP_FILE, "wb") as out:

    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Processing") as pbar:
        while True:
            print("Reading chunk...")
            chunk = f.read(CHUNK_SIZE)
            print("Read done, size:", len(chunk))
            if not chunk:
                break

            print("Tokenizing...")
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            print("Tokenized:", len(tokens))
            pbar.update(len(chunk))

            # write tokens directly (no RAM accumulation)
            tensor = torch.tensor(tokens, dtype=torch.int32)
            tensor.numpy().tofile(out)


# ---- convert raw binary → torch tensor ----
print("Finalizing tensor...")

tokens = torch.fromfile(TEMP_FILE, dtype=torch.int32)
torch.save(tokens, OUTPUT_FILE)

os.remove(TEMP_FILE)

print("Saved tokens to", OUTPUT_FILE)