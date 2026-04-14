import json
import random
import heapq
from collections import defaultdict, Counter
from tqdm import tqdm


class HeapBPE:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.bpe_ranks = {}

    def train(self, text, vocab_size, allowed_special=None):
        if allowed_special is None:
            allowed_special = set()

        text = text.replace("\r\n", "\n")

        # --- Build base vocab ---
        # base_symbols = [bytes([i]).decode("latin1") for i in range(256)]
        base_symbols = list("abcdefghijklmnopqrstuvwxyz \n.,")
        extra = sorted(set(text) - set(base_symbols))
        symbols = base_symbols + extra

        self.vocab = {i: s for i, s in enumerate(symbols)}
        self.inverse_vocab = {s: i for i, s in self.vocab.items()}

        for token in allowed_special:
            if token not in self.inverse_vocab:
                idx = len(self.vocab)
                self.vocab[idx] = token
                self.inverse_vocab[token] = idx

        # --- Encode text ---
        tokens = [self.inverse_vocab[ch] for ch in text]

        # --- Build initial pair stats ---
        pair_freq = Counter()
        pair_positions = defaultdict(set)

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] += 1
            pair_positions[pair].add(i)

        # --- Max heap (negative freq for max behavior) ---
        heap = [(-freq, pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)

        merges = []
        num_merges = vocab_size - len(self.vocab)

        pbar = tqdm(range(num_merges), desc="Training BPE (heap)")

        for _ in pbar:
            # --- get best pair ---
            while heap:
                freq, pair = heapq.heappop(heap)
                freq = -freq
                if pair_freq.get(pair, 0) == freq:
                    break
            else:
                break

            if freq < 2:
                break

            new_id = len(self.vocab) + len(merges)
            merges.append((pair, new_id))

            positions = sorted(pair_positions[pair])
            pair_positions[pair].clear()
            pair_freq[pair] = 0

            new_positions = []

            for pos in positions:
                if pos >= len(tokens) - 1:
                    continue
                if tokens[pos] is None or tokens[pos + 1] is None:
                    continue
                if (tokens[pos], tokens[pos + 1]) != pair:
                    continue

                # --- neighbors before merge ---
                if pos > 0 and tokens[pos - 1] is not None and tokens[pos] is not None:
                    left_pair = (tokens[pos - 1], tokens[pos])
                    pair_freq[left_pair] -= 1
                    pair_positions[left_pair].discard(pos - 1)

                if pos < len(tokens) - 2 and tokens[pos + 1] is not None and tokens[pos + 2] is not None:
                    right_pair = (tokens[pos + 1], tokens[pos + 2])
                    pair_freq[right_pair] -= 1
                    pair_positions[right_pair].discard(pos + 1)

                # --- merge ---
                tokens[pos] = new_id
                tokens[pos + 1] = None  # mark removed

                # --- neighbors after merge ---
                if pos > 0 and tokens[pos - 1] is not None:
                    new_left = (tokens[pos - 1], new_id)
                    pair_freq[new_left] += 1
                    pair_positions[new_left].add(pos - 1)
                    heapq.heappush(heap, (-pair_freq[new_left], new_left))

                if pos < len(tokens) - 2 and tokens[pos + 2] is not None:
                    new_right = (new_id, tokens[pos + 2])
                    pair_freq[new_right] += 1
                    pair_positions[new_right].add(pos)
                    heapq.heappush(heap, (-pair_freq[new_right], new_right))

            # --- compact tokens (remove None) ---
            # tokens = [t for t in tokens if t is not None]

            if len(merges) % 50 == 0:
                pbar.set_postfix({
                    "tokens": len(tokens),
                    "pairs": len(pair_freq)
                })

        # --- finalize ---
        for rank, (pair, new_id) in enumerate(merges):
            self.bpe_merges[pair] = new_id
            self.bpe_ranks[pair] = rank

            merged_str = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab[new_id] = merged_str
            self.inverse_vocab[merged_str] = new_id

    def encode(self, text):
        text = text.replace("\r\n", "\n")
        text = text.lower()
        tokens = [self.inverse_vocab.get(ch, self.inverse_vocab["<unk>"]) for ch in text]

        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            candidate = None
            best_rank = float("inf")

            for p in pairs:
                if p in self.bpe_ranks and self.bpe_ranks[p] < best_rank:
                    best_rank = self.bpe_ranks[p]
                    candidate = p

            if candidate is None:
                break

            new_id = self.bpe_merges[candidate]

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == candidate:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids).replace("_", " ")
    
    def save_tokenizer(self, vocab_path, merges_path):
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        merges_sorted = sorted(self.bpe_ranks.items(), key=lambda x: x[1])

        with open(merges_path, "w", encoding="utf-8") as f:
            for (id0, id1), rank in merges_sorted:
                new_id = self.bpe_merges[(id0, id1)]
                f.write(f"{id0} {id1} {new_id}\n")


    def load_tokenizer(self, vocab_path, merges_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            loaded_vocab = json.load(f)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in self.vocab.items()}

        self.bpe_merges = {}
        self.bpe_ranks = {}

        with open(merges_path, "r", encoding="utf-8") as f:
            for rank, line in enumerate(f):
                id0, id1, new_id = map(int, line.strip().split())
                pair = (id0, id1)
                self.bpe_merges[pair] = new_id
                self.bpe_ranks[pair] = rank

# --- Sampling with tqdm ---
def chunk_sample(file_path, target_chars=12_000_000, keep_prob=0.1):
    result = []
    total = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Sampling"):
            if total >= target_chars:
                break

            if random.random() < keep_prob:
                result.append(line)
                total += len(line)

    return "".join(result)


if __name__ == "__main__":
    text = chunk_sample("E:/gutenberg_books/books_corpus_cleaned_final_2.txt")

    bpe = HeapBPE()
    bpe.train(
        text,
        vocab_size=16000,
        allowed_special={"<s>", "</s>", "<unk>", "<|book|>", "<|endbook|>"}
    )

    print("VOCAB Size : ")
    print(len(bpe.vocab))
    print("BPE Merges Size : ")
    print(len(bpe.bpe_merges))

    input_text = "Jack embraced beauty through art and life."
    token_ids = bpe.encode(input_text)
    print(token_ids)

    print(bpe.decode(
        bpe.encode("This is some text.")
    ))

    bpe.save_tokenizer(vocab_path="triple_experiments/vocab.json", merges_path="triple_experiments/bpe_merges.txt")
