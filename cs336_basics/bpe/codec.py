from collections import deque
from cs336_basics.bpe.pretokenizer import Pretokenizer
from itertools import pairwise
from typing import Iterable, Iterator

import pickle

class BpeCodec:

    def __init__(
            self, 
            vocabs: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None):
        self.vocabs = vocabs
        self.inverted_vocabs = {
            token: idx
            for idx, token in vocabs.items()
        }
        self.merges = set(merges)
        self.special_tokens = special_tokens

        self.pretokenizer = Pretokenizer(special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocabs = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return BpeCodec(vocabs, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pretokens = self.pretokenizer.split(text)
        ids = []
        for pretoken in pretokens:
            tokens = deque([bytes([b]) for b in pretoken.encode('utf-8')])
            while tokens:
                if len(tokens) == 1:
                    token = tokens.popleft()
                    ids.append(self.inverted_vocabs[token])
                    break
                tok1, tok2 = tokens.popleft(), tokens.popleft()
                if (tok1, tok2) in self.merges:
                    tokens.appendleft(tok1 + tok2)
                else:
                    tokens.appendleft(tok2)
                    ids.append(self.inverted_vocabs[tok1])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        return b''\
            .join([self.vocabs[i] for i in ids])\
            .decode('utf-8', errors='replace')
