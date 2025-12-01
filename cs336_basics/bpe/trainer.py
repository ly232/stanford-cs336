from collections import Counter
from itertools import pairwise
from cs336_basics.bpe.pretokenizer import Pretokenizer

import pickle
import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BpeTrainer:

    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self.vocabs: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []

        self.pretokenizer = Pretokenizer(special_tokens)

    def persist(
            self, 
            vocab_path: str | os.PathLike, 
            merges_path: str | os.PathLike):
        with open(file=vocab_path, mode='wb') as f:
            pickle.dump(self.vocabs, f)
        with open(file=merges_path, mode='wb') as f:
            pickle.dump(self.merges, f)

    def __str__(self):
        debug_str = f"""
        vocabs: {self.vocabs},

        merges: {self.merges}
        """
        return debug_str

    def train(self, input_path: str | os.PathLike):
        with open(input_path, 'r') as f:
            text = f.read()
        text_sequence, pretokens_counter = self.pretokenizer.pretokenize(text)

        from collections import defaultdict
        import time
        profile = defaultdict(list)

        vocabs = {i: bytes([i]) for i in range(256)}
        for st in self.special_tokens:
            vocabs[len(vocabs)] = st.encode('utf-8')
        merges = []
        while len(vocabs) < self.vocab_size:
            counter = Counter()

            # Optimization to avoid recomputing the pairwise sliding in a pretoken.
            visited_pretokens = set()
            start_time = time.time()
            for pretoken in text_sequence:
                if pretoken in visited_pretokens:
                    continue
                visited_pretokens.add(pretoken)
                for tok1, tok2 in pairwise(text_sequence[pretoken]):
                    counter[(tok1, tok2)] += \
                        pretokens_counter[pretoken]
            end_time = time.time()
            profile['pretoken_pairwise_scan'].append(end_time - start_time)

            start_time = time.time()
            tok1, tok2 = \
                sorted(counter.keys(), key=lambda pair: (counter[pair], pair))[-1]
            end_time = time.time()
            profile['counter_key_sort'].append(end_time - start_time)

            idx = len(vocabs)
            vocabs[idx] = tok1 + tok2
            merges.append((tok1, tok2))

            new_text_sequence = {}
            start_time = time.time()
            for pretoken in text_sequence:
                if pretoken in new_text_sequence:
                    continue
                i = 0
                new_pretoken = []  # new pretoken bytes sequence
                old_pretoken = text_sequence[pretoken]  # old pretoken bytes seq
                while i < len(old_pretoken):
                    if old_pretoken[i:i+2] == [tok1, tok2]:
                        new_pretoken.append(tok1 + tok2)
                        i += 2
                    else:
                        new_pretoken.append(old_pretoken[i])
                        i += 1
                new_text_sequence[pretoken] = new_pretoken
            end_time = time.time()
            profile['new_text_sequence'].append(end_time - start_time)
            text_sequence = new_text_sequence

        import numpy as np
        for k, v in profile.items():
            print(f'avg latency for "{k}": {np.mean(v)} seconds; total: {np.sum(v)}')

        self.vocabs, self.merges = vocabs, merges
        return vocabs, merges

    
