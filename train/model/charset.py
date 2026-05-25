from __future__ import annotations

DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def char_to_index(alphabet: str = DEFAULT_ALPHABET) -> dict[str, int]:
    return {char: idx for idx, char in enumerate(alphabet)}


def index_to_char(alphabet: str = DEFAULT_ALPHABET) -> dict[int, str]:
    return {idx: char for idx, char in enumerate(alphabet)}


def encode_text(text: str, alphabet: str = DEFAULT_ALPHABET) -> list[int]:
    lookup = char_to_index(alphabet)
    return [lookup[char] for char in text]


def decode_indices(indices: list[int], alphabet: str = DEFAULT_ALPHABET) -> str:
    lookup = index_to_char(alphabet)
    return "".join(lookup[idx] for idx in indices)
