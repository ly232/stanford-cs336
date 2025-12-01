from collections import Counter

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Pretokenizer:

    def __init__(self, special_tokens: list[str] | None, pattern=PAT):
        self.special_tokens = \
            special_tokens if special_tokens is not None else []
        self.pattern = pattern

    def split(self, text):
        return re.findall(self.pattern, text)

    def pretokenize(self, text: str) -> tuple[dict[str, list[bytes]], Counter]:
        """Pretokenizes to mapping pretoken string to token bytes split."""
        chunks = [text]
        for st in self.special_tokens:
            new_chunks = []
            while chunks:
                chunk = chunks.pop()
                new_chunks.extend(chunk.split(st))
            chunks = new_chunks

        # Pre-tokenization.
        pretokens = []
        for chunk in chunks:
            pretokens.extend(re.findall(self.pattern, chunk))

        # Optimization to avoid repeating the pairwise sliding for the same pretoken
        pretokens_counter = Counter(pretokens)

        # maps pretoken string to token bytes split.
        text_sequence = {
            pretoken: [bytes([b]) for b in pretoken.encode('utf-8')]
            for pretoken in pretokens
        }

        return text_sequence, pretokens_counter
