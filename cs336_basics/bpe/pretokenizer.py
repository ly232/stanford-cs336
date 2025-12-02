from collections import Counter

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Pretokenizer:

    def __init__(self, special_tokens: list[str] | None, pattern=PAT):
        self.special_tokens = \
            special_tokens if special_tokens is not None else []
        self.special_tokens = sorted(self.special_tokens, reverse=True)
        self.pattern = pattern

    def split(self, text):
        chunks = [text]

        def _split_chunk(chunk, special_token):
            i = 0
            buffer = ''
            splitted = []
            while i < len(chunk):
                if chunk[i:i+len(special_token)] == special_token:
                    if buffer != '':
                        splitted.append(buffer)
                    buffer = ''
                    splitted.append(special_token)
                    i += len(special_token)
                else:
                    buffer += chunk[i]
                    i += 1
            if buffer != '':
                splitted.append(buffer)
            return splitted

        for st in self.special_tokens:
            new_chunks = []
            for chunk in chunks:
                if chunk in self.special_tokens:
                    new_chunks.append(chunk)
                else:
                    new_chunks.extend(_split_chunk(chunk, st))
            chunks = new_chunks

        # Pre-tokenization.
        pretokens = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                pretokens.append(chunk)
            else:
                pretokens.extend(re.findall(self.pattern, chunk))
        return pretokens

    def pretokenize(self, text: str) -> tuple[dict[str, list[bytes]], Counter]:
        """Pretokenizes to mapping pretoken string to token bytes split."""
        pretokens = self.split(text)

        # Optimization to avoid repeating the pairwise sliding for the same pretoken
        pretokens_counter = Counter(pretokens)

        # maps pretoken string to token bytes split.
        text_sequence = {
            pretoken: [bytes([b]) for b in pretoken.encode('utf-8')]
            for pretoken in pretokens
        }

        return text_sequence, pretokens_counter
