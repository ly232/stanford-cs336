from typing import Iterable, Iterator

import pickle

class BpeCodec:

    def __init__(
            self, 
            vocabs: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None):
        self.vocabs = vocabs
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocabs = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return BpeCodec(vocabs, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ...

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...