import json
from typing import Counter

class BPE:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.bpe_ranks = {}
    
    def train(
            self,
            text,
            vocab_size,
            allowed_special={"<s>", "</s>", "<unk>"}
    ):
        if allowed_special is None:
            allowed_special = set()
        
        text = text.replace("\r\n", "\n")
        text = text.replace(" ", "_")

        base_symbols = [bytes([i]).decode("latin1") for i in range(256)]

        extra = sorted(set(text) - set(base_symbols))
        symbols = base_symbols + extra

        self.vocab = {i : s for i, s in enumerate(symbols)}
        self.inverse_vocab = {s: i for i, s in self.vocab.items()}

        for token in allowed_special:
            if token not in self.inverse_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = token
                self.inverse_vocab[token] = new_id
        
        token_ids = [self._symbol_id(ch) for ch in text]
        merges = []

        while len(self.vocab) + len(merges) < vocab_size:
            pair = self._most_freq_pair(token_ids)
            if pair is None:
                break

            new_id = len(self.vocab) + len(merges)
            token_ids = self._merge_pair(token_ids, pair, new_id)
            merges.append((pair, new_id))

        for rank, (pair, new_id) in enumerate(merges):
            self.bpe_merges[pair] = new_id
            self.bpe_ranks[pair] = rank
            merged_str = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab[new_id] = merged_str
            self.inverse_vocab[merged_str] = new_id


    def encode(self, text, allowed_special=None):
        text = text.replace("\r\n", "\n")
        text = text.replace(" ", "_")

        if allowed_special is None:
            allowed_special = set()

        ids = []
        i = 0
        while i < len(text):
            matched = False

            # check special tokens
            for tok in allowed_special:
                if text.startswith(tok, i):
                    ids.append(self.inverse_vocab[tok])
                    i += len(tok)
                    matched = True
                    break

            if matched:
                continue

            ch = text[i]
            ids.extend(self._encode_symbol(ch))
            i += 1

        return ids

    
    def _encode_symbol(self, ch):
        if ch not in self.inverse_vocab:
            utf8 = ch.encode("utf-8")
            return [self.inverse_vocab[bytes([b]).decode("latin1")] for b in utf8]
        
        token_ids = [self.inverse_vocab[ch]]

        while True:
            best_rank = None
            best_pos = None

            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i+1])
                if pair in self.bpe_ranks:
                    r = self.bpe_ranks[pair]
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_pos = i
            
            if best_pos is None:
                break

            pair = (token_ids[best_pos], token_ids[best_pos + 1])
            merged = self.bpe_merges[pair]
            token_ids = (
                token_ids[:best_pos]
                + [merged]
                + token_ids[best_pos + 2:]
            )

        return token_ids

    def decode(self, token_ids):
        pieces = [self.vocab[i] for i in token_ids]
        text = "".join(pieces)

        return text.replace("_", " ")

    def _symbol_id(self, ch):
        if ch in self.inverse_vocab:
            return self.inverse_vocab[ch]

        utf8 = ch.encode("utf-8")
        return [self.inverse_vocab[bytes([b]).decode("latin1")] for b in utf8]
    
    @staticmethod
    def _most_freq_pair(ids):
        if len(ids) < 2:
            return None

        counts = Counter(zip(ids, ids[1:]))
        return max(counts, key=counts.get)
    
    @staticmethod
    def _merge_pair(ids, pair, new_id):
        out = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        
        return out

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



if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPE()
    # tokenizer.train(
    #     text,
    #     vocab_size=1000,
    #     allowed_special={"<s>", "</s>", "<unk>"}
    # )

    # print("VOCAB Size : ")
    # print(len(tokenizer.vocab))
    # print("BPE Merges Size : ")
    # print(len(tokenizer.bpe_merges))

    # input_text = "Jack embraced beauty through art and life."
    # token_ids = tokenizer.encode(input_text)
    # print(token_ids)

    # print(tokenizer.decode(
    #     tokenizer.encode("This is some text.")
    # ))

    # tokenizer.save_tokenizer(vocab_path="vocab.json", merges_path="bpe_merges.txt")

    tokenizer.load_tokenizer("vocab.json", "bpe_merges.txt")

    s = "To be, or not to be."
    ids = tokenizer.encode(s)
    print(ids)
    print(tokenizer.decode(ids))