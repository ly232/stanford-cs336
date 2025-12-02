from collections import deque
from cs336_basics.bpe.pretokenizer import Pretokenizer
from itertools import pairwise
from typing import Any, Generator, Iterable, Iterator

import pickle


class CodecIterator(Iterator[int]):

    def __init__(self, iterable: Iterable[str], codec: 'BpeCodec'):
        self.iterator = iter(iterable)
        self.ids: deque[int] = deque([])
        self.codec = codec

    def __iter__(self):
        return self
    
    def __next__(self) -> int:
        while True:
            if self.ids:
                return self.ids.popleft()
            self.ids = deque(self.codec.encode(text=next(self.iterator)))


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
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.pretokenizer = Pretokenizer(special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocabs = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return BpeCodec(vocabs, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pretokens = self.pretokenizer.pretokenize_for_inference(text)
        ids = []
        for pretoken in pretokens:
            if pretoken in self.special_tokens:
                ids.append(self.inverted_vocabs[pretoken.encode()])
                continue
            tokens = [bytes([b]) for b in pretoken.encode('utf-8')]
            for merge in self.merges:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i == len(tokens) - 1:
                        new_tokens.append(tokens[i])
                        break
                    if (tokens[i], tokens[i+1]) == merge:
                        new_tokens.append(tokens[i] + tokens[i+1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            ids.extend([self.inverted_vocabs[t] for t in tokens])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return CodecIterator(iterable, self)

    def decode(self, ids: list[int]) -> str:
        return b''\
            .join([self.vocabs[i] for i in ids])\
            .decode('utf-8', errors='replace')
